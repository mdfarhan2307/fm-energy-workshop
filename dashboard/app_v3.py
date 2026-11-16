"""
Energy Dashboard v3 - Live Simulation + Prediction
====================================================
Based on v2. All simulation behaviour is identical:
  - Starts with all January 2016 readings loaded.
  - Every real minute, fetches the next recorded hour from Neon.
  - Stops when the last February 2016 record is reached.
  - Restart simulation button resets to January baseline.

New in v3
---------
  Prediction section now has a "Predict next hour" button.
  Clicking it uses models/gbr_model.joblib (GradientBoostingRegressor,
  already trained on METER_01) to predict the next hour for each meter
  independently, using only the readings revealed so far.

  Features required per prediction (all from the target timestamp T):
    hour, day_of_week, lag_1h, lag_2h, lag_3h, lag_24h, lag_168h

  Minimum history needed: 168 readings per meter (7 days).
  If a meter does not yet have enough history, a clear message is shown.

Run from the project root:
    ..\\vibe\\Scripts\\streamlit.exe run dashboard/app_v3.py --server.headless true
"""

import json
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Dashboard v3",
    page_icon="⚡",
    layout="wide",
)

TICK_SECONDS = 60
JAN_END      = pd.Timestamp("2016-01-31 23:00:00", tz="UTC")
DATA_END     = pd.Timestamp("2016-02-29 23:00:00", tz="UTC")
MIN_HISTORY  = 168   # lag_168h requires at least 168 prior hourly readings

METER_COLORS = {
    "METER_01": "#00C2FF",
    "METER_02": "#FF6B35",
    "METER_03": "#7FBA00",
    "TOTAL":    "#A259FF",
}

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.abspath(os.path.join(_HERE, ".."))
_MODEL_PATH    = os.path.join(_ROOT, "models", "gbr_model.joblib")
_FEATURES_PATH = os.path.join(_ROOT, "models", "feature_order.json")

# ── Load model once (module-level, not inside session state) ──────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model    = joblib.load(_MODEL_PATH)
    with open(_FEATURES_PATH) as f:
        features = json.load(f)
    return model, features

# ── Neon helpers ──────────────────────────────────────────────────────────────

def _load_env(path: str) -> dict:
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^([A-Z_]+)\s*=\s*"?([^"#\n]+)"?', line)
            if m:
                env[m.group(1)] = m.group(2).strip()
    return env


def _connection() -> tuple[str, dict]:
    # On Streamlit Cloud use st.secrets; locally fall back to .env
    if "DATABASE_URL" in st.secrets:
        conn = st.secrets["DATABASE_URL"]
    else:
        env = _load_env(os.path.join(_ROOT, ".env"))
        conn = env["DATABASE_URL"]
    conn = re.sub(r"[&?]channel_binding=require", "", conn)
    m = re.match(r"postgresql://[^@]+@([^/?]+)", conn)
    if not m:
        raise ValueError("Cannot parse host from DATABASE_URL")
    sql_url = f"https://{m.group(1)}/sql"
    headers = {
        "Content-Type": "application/json",
        "Neon-Connection-String": conn,
    }
    return sql_url, headers


def _run_query(query: str, params: list | None = None) -> list[dict]:
    sql_url, headers = _connection()
    payload: dict = {"query": query}
    if params:
        payload["params"] = params
    resp = requests.post(sql_url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json().get("rows", [])


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading January baseline from Neon...")
def load_january() -> pd.DataFrame:
    rows = _run_query("""
        SELECT meter_id, timestamp, electricity_kwh
        FROM   meter_readings_source
        WHERE  timestamp >= '2016-01-01 00:00:00+00'
          AND  timestamp <= '2016-01-31 23:00:00+00'
        ORDER  BY timestamp, meter_id
    """)
    df = pd.DataFrame(rows)
    df["timestamp"]       = pd.to_datetime(df["timestamp"], utc=True)
    df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"])
    return df


def fetch_next_hour(after: pd.Timestamp) -> pd.DataFrame:
    rows = _run_query("""
        SELECT meter_id, timestamp, electricity_kwh
        FROM   meter_readings_source
        WHERE  timestamp = (
            SELECT MIN(timestamp)
            FROM   meter_readings_source
            WHERE  timestamp > $1
        )
        ORDER  BY meter_id
    """, params=[after.isoformat()])
    if not rows:
        return pd.DataFrame(columns=["meter_id", "timestamp", "electricity_kwh"])
    df = pd.DataFrame(rows)
    df["timestamp"]       = pd.to_datetime(df["timestamp"], utc=True)
    df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"])
    return df


# ── Prediction helper ─────────────────────────────────────────────────────────

def predict_next_hour(meter_df: pd.DataFrame, meter_id: str) -> dict:
    """
    Given a single meter's readings (sorted, UTC timestamps), predict the
    next hour using the saved GBR model.

    Returns a dict with keys:
      ok        bool
      meter_id  str
      target_ts pd.Timestamp | None
      kwh       float | None
      message   str
    """
    model, features = load_model()

    meter_df = meter_df.sort_values("timestamp").reset_index(drop=True)
    n = len(meter_df)

    if n < MIN_HISTORY:
        return {
            "ok": False,
            "meter_id": meter_id,
            "target_ts": None,
            "kwh": None,
            "message": (
                f"Need at least {MIN_HISTORY} hourly readings to populate lag_168h. "
                f"Currently have {n}."
            ),
        }

    last_ts   = meter_df["timestamp"].iloc[-1]
    target_ts = last_ts + pd.Timedelta(hours=1)

    # Build lag lookup from the meter's own history
    kwh_index = meter_df.set_index("timestamp")["electricity_kwh"]

    def get_lag(hours: int) -> float | None:
        key = target_ts - pd.Timedelta(hours=hours)
        return float(kwh_index[key]) if key in kwh_index.index else None

    lag_values = {
        "lag_1h":   get_lag(1),
        "lag_2h":   get_lag(2),
        "lag_3h":   get_lag(3),
        "lag_24h":  get_lag(24),
        "lag_168h": get_lag(168),
    }

    missing = [k for k, v in lag_values.items() if v is None]
    if missing:
        return {
            "ok": False,
            "meter_id": meter_id,
            "target_ts": target_ts,
            "kwh": None,
            "message": f"Missing lag values: {', '.join(missing)}.",
        }

    X = pd.DataFrame([{
        "hour":        target_ts.hour,
        "day_of_week": target_ts.day_of_week,
        **lag_values,
    }])[features]

    pred = float(model.predict(X)[0])
    return {
        "ok": True,
        "meter_id": meter_id,
        "target_ts": target_ts,
        "kwh": pred,
        "message": "",
    }


# ── Chart builders ────────────────────────────────────────────────────────────

def _base_layout(title: str, height: int) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title="Time", showgrid=False),
        yaxis=dict(title="kWh", showgrid=True, gridcolor="#2a2a2a"),
        margin=dict(l=40, r=10, t=40, b=40),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        hovermode="x unified",
        height=height,
    )


def meter_chart(df: pd.DataFrame, meter_id: str) -> go.Figure:
    sub = df[df["meter_id"] == meter_id].sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["timestamp"],
        y=sub["electricity_kwh"],
        mode="lines",
        name=meter_id,
        line=dict(color=METER_COLORS[meter_id], width=1.5),
    ))
    fig.update_layout(**_base_layout(meter_id, 280))
    return fig


def total_chart(df: pd.DataFrame) -> go.Figure:
    total = (
        df.groupby("timestamp", as_index=False)["electricity_kwh"]
        .sum()
        .sort_values("timestamp")
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=total["timestamp"],
        y=total["electricity_kwh"],
        mode="lines",
        name="Total",
        line=dict(color=METER_COLORS["TOTAL"], width=2),
    ))
    fig.update_layout(**_base_layout("Total - All Meters Combined", 320))
    return fig


# ── Session-state bootstrap ───────────────────────────────────────────────────

def _init_state():
    jan = load_january()
    st.session_state.display_df    = jan.copy()
    st.session_state.last_ts       = JAN_END
    st.session_state.sim_done      = False
    st.session_state.last_tick     = time.time()
    st.session_state.tick_count    = 0
    st.session_state.predictions   = []   # list of result dicts from predict_next_hour


if "display_df" not in st.session_state:
    _init_state()

# ── Title & controls ──────────────────────────────────────────────────────────
st.title("Energy Dashboard v3 - Live Simulation")
st.caption(
    "Data source: **historical** readings from `meter_readings_source` (Neon PostgreSQL). "
    "Simulation advances one recorded hour per real minute."
)

col_btn, col_status = st.columns([1, 5])
with col_btn:
    if st.button("Restart simulation", use_container_width=True):
        load_january.clear()
        _init_state()
        st.rerun()

# ── Tick logic ────────────────────────────────────────────────────────────────
if not st.session_state.sim_done:
    now     = time.time()
    elapsed = now - st.session_state.last_tick

    if elapsed >= TICK_SECONDS:
        new_rows = fetch_next_hour(st.session_state.last_ts)

        if new_rows.empty:
            st.session_state.sim_done = True
        else:
            new_ts = new_rows["timestamp"].iloc[0]
            st.session_state.display_df = pd.concat(
                [st.session_state.display_df, new_rows], ignore_index=True
            )
            st.session_state.last_ts    = new_ts
            st.session_state.last_tick  = now
            st.session_state.tick_count += 1
            # Clear cached predictions when new data arrives
            st.session_state.predictions = []

            if new_ts >= DATA_END:
                st.session_state.sim_done = True

# ── Status line ───────────────────────────────────────────────────────────────
df   = st.session_state.display_df
last = st.session_state.last_ts

with col_status:
    if st.session_state.sim_done:
        st.success(
            f"Simulation complete - all data shown through "
            f"{last.strftime('%d %b %Y %H:%M')} UTC.",
            icon=None,
        )
    else:
        ticks        = st.session_state.tick_count
        next_tick_in = max(0, int(TICK_SECONDS - (time.time() - st.session_state.last_tick)))
        st.info(
            f"Showing up to **{last.strftime('%d %b %Y %H:%M')} UTC** "
            f"({ticks} hour{'s' if ticks != 1 else ''} streamed) - "
            f"next fetch in ~{next_tick_in}s",
            icon=None,
        )

# ── Charts - Row 1: individual meters ────────────────────────────────────────
st.subheader("Hourly Electricity Usage - Individual Meters")
meter_ids = sorted(df["meter_id"].unique())
cols = st.columns(3)
for col, meter_id in zip(cols, meter_ids):
    with col:
        st.plotly_chart(meter_chart(df, meter_id), use_container_width=True)

st.divider()

# ── Charts - Row 2: total ─────────────────────────────────────────────────────
st.subheader("Total Electricity Usage - All Meters")
st.plotly_chart(total_chart(df), use_container_width=True)

st.divider()

# ── Bottom row: Prediction + LLM Analysis ────────────────────────────────────
left, right = st.columns(2)

# ── Prediction ────────────────────────────────────────────────────────────────
with left:
    st.subheader("Prediction")

    if st.button("Predict next hour", use_container_width=True):
        results = []
        for mid in meter_ids:
            meter_df = (
                df[df["meter_id"] == mid]
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            results.append(predict_next_hour(meter_df, mid))
        st.session_state.predictions = results

    if st.session_state.predictions:
        for res in st.session_state.predictions:
            color = METER_COLORS.get(res["meter_id"], "#FFFFFF")
            if res["ok"]:
                target_str = res["target_ts"].strftime("%d %b %Y %H:%M UTC")
                st.markdown(
                    f"""
<div style="
    border-left: 4px solid {color};
    padding: 10px 16px;
    margin-bottom: 10px;
    background-color: #1a1a2e;
    border-radius: 4px;
">
  <span style="color:{color}; font-weight:bold; font-size:1rem;">{res['meter_id']}</span><br>
  <span style="font-size:1.5rem; font-weight:bold;">{res['kwh']:.2f} kWh</span><br>
  <span style="color:#aaa; font-size:0.85rem;">Target: {target_str}</span>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.warning(
                    f"**{res['meter_id']}** - Cannot predict: {res['message']}",
                    icon="⚠️",
                )
    else:
        st.info(
            "Press **Predict next hour** to generate a one-hour-ahead forecast "
            "for each meter using the saved GradientBoostingRegressor model. "
            "Predictions use only the readings revealed so far.",
            icon="🔮",
        )

# ── LLM Analysis ─────────────────────────────────────────────────────────────
with right:
    st.subheader("LLM Analysis")
    st.info(
        "**Coming soon.** This section will display an AI-generated commentary on "
        "recent energy patterns, powered by the OpenAI API.",
        icon="🤖",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption(
    f"Showing **{len(df):,}** readings across **{len(meter_ids)}** meters "
    f"({df['timestamp'].min().date()} to {df['timestamp'].max().date()})."
)

# ── Auto-rerun while simulation is running ────────────────────────────────────
if not st.session_state.sim_done:
    time.sleep(1)
    st.rerun()
