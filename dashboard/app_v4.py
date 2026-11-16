"""
Energy Dashboard v4 - Live Simulation + Prediction + LLM Analysis
==================================================================
Based on v3. All simulation and prediction behaviour is identical.

New in v4
---------
  The LLM Analysis placeholder is replaced with a working panel.

  Fixed prompt (read-only):
    "Give me a summary of the energy usage for METER_02 for the past 2 days"

  When Send is pressed:
    1. The latest 48 hourly readings for METER_02 are taken from whatever
       has been revealed so far in the simulation.
    2. Python calculates total, average, min and max kWh from those readings.
    3. A JSON object is built containing the meter name, time range,
       the four statistics and the full list of 48 hourly readings.
    4. The fixed prompt plus that JSON is sent to OpenAI as a single user
       message.  The API is called ONCE per button press and the response
       is stored in session state so simulation reruns and auto-ticks never
       fire a second request.

  The panel shows:
    - The LLM response text
    - "Based on readings up to <latest timestamp in the 48-hour window>"
    - A counter of how many LLM requests have been made this session
    - A collapsed expander containing the JSON that was sent

  OPENAI_API_KEY and OPENAI_MODEL are read from .env.
  The key is never displayed.

Run from the project root:
    ..\\vibe\\Scripts\\streamlit.exe run dashboard/app_v4.py --server.headless true
"""

import json
import os
import re
import time

import joblib
import openai
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Dashboard v4",
    page_icon="⚡",
    layout="wide",
)

TICK_SECONDS  = 60
JAN_END       = pd.Timestamp("2016-01-31 23:00:00", tz="UTC")
DATA_END      = pd.Timestamp("2016-02-29 23:00:00", tz="UTC")
MIN_HISTORY   = 168
LLM_METER     = "METER_02"
LLM_WINDOW_H  = 48          # hours of history sent to the LLM
FIXED_PROMPT  = "Give me a summary of the energy usage for METER_02 for the past 2 days"

METER_COLORS = {
    "METER_01": "#00C2FF",
    "METER_02": "#FF6B35",
    "METER_03": "#7FBA00",
    "TOTAL":    "#A259FF",
}

_HERE          = os.path.dirname(os.path.abspath(__file__))
_ROOT          = os.path.abspath(os.path.join(_HERE, ".."))
_MODEL_PATH    = os.path.join(_ROOT, "models", "gbr_model.joblib")
_FEATURES_PATH = os.path.join(_ROOT, "models", "feature_order.json")

# ── Load model once ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load(_MODEL_PATH)
    with open(_FEATURES_PATH) as f:
        features = json.load(f)
    return model, features

# ── Environment ───────────────────────────────────────────────────────────────
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

@st.cache_resource(show_spinner=False)
def _env() -> dict:
    # On Streamlit Cloud, secrets are injected via st.secrets.
    # Locally, fall back to reading the .env file.
    if "DATABASE_URL" in st.secrets:
        return {k: st.secrets[k] for k in st.secrets}
    return _load_env(os.path.join(_ROOT, ".env"))

# ── Neon helpers ──────────────────────────────────────────────────────────────
def _connection() -> tuple[str, dict]:
    env = _env()
    conn = re.sub(r"[&?]channel_binding=require", "", env["DATABASE_URL"])
    m = re.match(r"postgresql://[^@]+@([^/?]+)", conn)
    if not m:
        raise ValueError("Cannot parse host from DATABASE_URL")
    sql_url = f"https://{m.group(1)}/sql"
    headers = {"Content-Type": "application/json", "Neon-Connection-String": conn}
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
            SELECT MIN(timestamp) FROM meter_readings_source
            WHERE  timestamp > $1
        )
        ORDER BY meter_id
    """, params=[after.isoformat()])
    if not rows:
        return pd.DataFrame(columns=["meter_id", "timestamp", "electricity_kwh"])
    df = pd.DataFrame(rows)
    df["timestamp"]       = pd.to_datetime(df["timestamp"], utc=True)
    df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"])
    return df

# ── LLM helper ────────────────────────────────────────────────────────────────
def build_llm_payload(display_df: pd.DataFrame) -> tuple[dict, str] | tuple[None, str]:
    """
    Take the latest LLM_WINDOW_H rows for LLM_METER from display_df.
    Return (json_payload, error_message).  error_message is empty on success.
    """
    meter_df = (
        display_df[display_df["meter_id"] == LLM_METER]
        .sort_values("timestamp")
        .tail(LLM_WINDOW_H)
        .reset_index(drop=True)
    )
    if len(meter_df) < LLM_WINDOW_H:
        return None, (
            f"Only {len(meter_df)} readings available for {LLM_METER}. "
            f"Need at least {LLM_WINDOW_H} (48 hours)."
        )

    kwh = meter_df["electricity_kwh"]
    readings = [
        {"timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
         "electricity_kwh": round(float(row["electricity_kwh"]), 2)}
        for _, row in meter_df.iterrows()
    ]
    payload = {
        "meter":      LLM_METER,
        "time_range": {
            "from": meter_df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M UTC"),
            "to":   meter_df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC"),
        },
        "statistics": {
            "total_kwh":   round(float(kwh.sum()),  2),
            "average_kwh": round(float(kwh.mean()), 2),
            "min_kwh":     round(float(kwh.min()),  2),
            "max_kwh":     round(float(kwh.max()),  2),
        },
        "hourly_readings": readings,
    }
    return payload, ""


def call_openai(prompt: str, data_payload: dict) -> str:
    """Send prompt + JSON context to OpenAI and return the reply text."""
    env = _env()
    model_name = env.get("OPENAI_MODEL", "gpt-3.5-turbo")

    user_message = (
        f"{prompt}\n\n"
        f"Here is the data:\n```json\n{json.dumps(data_payload, indent=2)}\n```"
    )

    client = openai.OpenAI(api_key=env["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an energy analyst assistant. "
                    "Analyse the meter data provided and give a clear, concise summary."
                ),
            },
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content

# ── Prediction helper (unchanged from v3) ────────────────────────────────────
def predict_next_hour(meter_df: pd.DataFrame, meter_id: str) -> dict:
    model, features = load_model()
    meter_df = meter_df.sort_values("timestamp").reset_index(drop=True)
    n = len(meter_df)
    if n < MIN_HISTORY:
        return {"ok": False, "meter_id": meter_id, "target_ts": None, "kwh": None,
                "message": f"Need at least {MIN_HISTORY} readings. Currently have {n}."}
    last_ts   = meter_df["timestamp"].iloc[-1]
    target_ts = last_ts + pd.Timedelta(hours=1)
    kwh_index = meter_df.set_index("timestamp")["electricity_kwh"]
    def get_lag(h):
        key = target_ts - pd.Timedelta(hours=h)
        return float(kwh_index[key]) if key in kwh_index.index else None
    lag_values = {f"lag_{h}h": get_lag(h) for h in [1, 2, 3, 24, 168]}
    missing = [k for k, v in lag_values.items() if v is None]
    if missing:
        return {"ok": False, "meter_id": meter_id, "target_ts": target_ts, "kwh": None,
                "message": f"Missing lag values: {', '.join(missing)}."}
    X = pd.DataFrame([{"hour": target_ts.hour, "day_of_week": target_ts.day_of_week,
                        **lag_values}])[features]
    return {"ok": True, "meter_id": meter_id, "target_ts": target_ts,
            "kwh": float(model.predict(X)[0]), "message": ""}

# ── Chart builders ────────────────────────────────────────────────────────────
def _base_layout(title: str, height: int) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title="Time", showgrid=False),
        yaxis=dict(title="kWh", showgrid=True, gridcolor="#2a2a2a"),
        margin=dict(l=40, r=10, t=40, b=40),
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"), hovermode="x unified", height=height,
    )

def meter_chart(df: pd.DataFrame, meter_id: str) -> go.Figure:
    sub = df[df["meter_id"] == meter_id].sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["electricity_kwh"],
                             mode="lines", name=meter_id,
                             line=dict(color=METER_COLORS[meter_id], width=1.5)))
    fig.update_layout(**_base_layout(meter_id, 280))
    return fig

def total_chart(df: pd.DataFrame) -> go.Figure:
    total = df.groupby("timestamp", as_index=False)["electricity_kwh"].sum().sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=total["timestamp"], y=total["electricity_kwh"],
                             mode="lines", name="Total",
                             line=dict(color=METER_COLORS["TOTAL"], width=2)))
    fig.update_layout(**_base_layout("Total - All Meters Combined", 320))
    return fig

# ── Session-state bootstrap ───────────────────────────────────────────────────
def _init_state():
    jan = load_january()
    st.session_state.display_df   = jan.copy()
    st.session_state.last_ts      = JAN_END
    st.session_state.sim_done     = False
    st.session_state.last_tick    = time.time()
    st.session_state.tick_count   = 0
    st.session_state.predictions  = []
    # LLM state — persists across reruns, cleared only on Restart
    if "llm_response"    not in st.session_state:
        st.session_state.llm_response    = None   # str | None
    if "llm_data_sent"   not in st.session_state:
        st.session_state.llm_data_sent   = None   # dict | None
    if "llm_latest_ts"   not in st.session_state:
        st.session_state.llm_latest_ts   = None   # str | None
    if "llm_request_count" not in st.session_state:
        st.session_state.llm_request_count = 0
    if "llm_error"       not in st.session_state:
        st.session_state.llm_error       = None   # str | None

if "display_df" not in st.session_state:
    _init_state()

# ── Title & controls ──────────────────────────────────────────────────────────
st.title("Energy Dashboard v4 - Live Simulation")
st.caption(
    "Data source: **historical** readings from `meter_readings_source` (Neon PostgreSQL). "
    "Simulation advances one recorded hour per real minute."
)

col_btn, col_status = st.columns([1, 5])
with col_btn:
    if st.button("Restart simulation", use_container_width=True):
        load_january.clear()
        # Reset LLM state on restart too
        for key in ["llm_response", "llm_data_sent", "llm_latest_ts",
                    "llm_request_count", "llm_error"]:
            st.session_state.pop(key, None)
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
                [st.session_state.display_df, new_rows], ignore_index=True)
            st.session_state.last_ts   = new_ts
            st.session_state.last_tick = now
            st.session_state.tick_count += 1
            st.session_state.predictions = []
            if new_ts >= DATA_END:
                st.session_state.sim_done = True

# ── Status line ───────────────────────────────────────────────────────────────
df   = st.session_state.display_df
last = st.session_state.last_ts

with col_status:
    if st.session_state.sim_done:
        st.success(f"Simulation complete - all data shown through "
                   f"{last.strftime('%d %b %Y %H:%M')} UTC.", icon=None)
    else:
        ticks        = st.session_state.tick_count
        next_tick_in = max(0, int(TICK_SECONDS - (time.time() - st.session_state.last_tick)))
        st.info(f"Showing up to **{last.strftime('%d %b %Y %H:%M')} UTC** "
                f"({ticks} hour{'s' if ticks != 1 else ''} streamed) - "
                f"next fetch in ~{next_tick_in}s", icon=None)

# ── Charts - Row 1 ────────────────────────────────────────────────────────────
st.subheader("Hourly Electricity Usage - Individual Meters")
meter_ids = sorted(df["meter_id"].unique())
cols = st.columns(3)
for col, meter_id in zip(cols, meter_ids):
    with col:
        st.plotly_chart(meter_chart(df, meter_id), use_container_width=True)

st.divider()

# ── Charts - Row 2 ────────────────────────────────────────────────────────────
st.subheader("Total Electricity Usage - All Meters")
st.plotly_chart(total_chart(df), use_container_width=True)

st.divider()

# ── Bottom row ────────────────────────────────────────────────────────────────
left, right = st.columns(2)

# ── Prediction (unchanged from v3) ───────────────────────────────────────────
with left:
    st.subheader("Prediction")
    if st.button("Predict next hour", use_container_width=True):
        results = []
        for mid in meter_ids:
            mdf = df[df["meter_id"] == mid].sort_values("timestamp").reset_index(drop=True)
            results.append(predict_next_hour(mdf, mid))
        st.session_state.predictions = results

    if st.session_state.predictions:
        for res in st.session_state.predictions:
            color = METER_COLORS.get(res["meter_id"], "#FFFFFF")
            if res["ok"]:
                target_str = res["target_ts"].strftime("%d %b %Y %H:%M UTC")
                st.markdown(
                    f'<div style="border-left:4px solid {color};padding:10px 16px;'
                    f'margin-bottom:10px;background-color:#1a1a2e;border-radius:4px;">'
                    f'<span style="color:{color};font-weight:bold;">{res["meter_id"]}</span><br>'
                    f'<span style="font-size:1.5rem;font-weight:bold;">{res["kwh"]:.2f} kWh</span><br>'
                    f'<span style="color:#aaa;font-size:0.85rem;">Target: {target_str}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning(f"**{res['meter_id']}** - Cannot predict: {res['message']}", icon="⚠️")
    else:
        st.info("Press **Predict next hour** to generate a one-hour-ahead forecast "
                "for each meter using the saved GradientBoostingRegressor model. "
                "Predictions use only the readings revealed so far.", icon="🔮")

# ── LLM Analysis ─────────────────────────────────────────────────────────────
with right:
    st.subheader("LLM Analysis")

    # Read-only prompt display
    st.text_area(
        label="Prompt",
        value=FIXED_PROMPT,
        height=80,
        disabled=True,
        label_visibility="collapsed",
    )

    send_clicked = st.button("Send", use_container_width=True)

    if send_clicked:
        payload, err = build_llm_payload(df)
        if err:
            st.session_state.llm_error    = err
            st.session_state.llm_response = None
        else:
            with st.spinner("Asking the LLM..."):
                try:
                    reply = call_openai(FIXED_PROMPT, payload)
                    latest_ts_str = payload["time_range"]["to"]
                    st.session_state.llm_response      = reply
                    st.session_state.llm_data_sent     = payload
                    st.session_state.llm_latest_ts     = latest_ts_str
                    st.session_state.llm_request_count += 1
                    st.session_state.llm_error         = None
                except Exception as exc:
                    st.session_state.llm_error    = f"OpenAI API error: {exc}"
                    st.session_state.llm_response = None

    # ── Show results ──────────────────────────────────────────────────────────
    if st.session_state.llm_error:
        st.warning(st.session_state.llm_error, icon="⚠️")

    elif st.session_state.llm_response:
        count = st.session_state.llm_request_count
        latest = st.session_state.llm_latest_ts
        st.caption(
            f"Based on readings up to **{latest}** · "
            f"LLM requests this session: **{count}**"
        )
        st.markdown(st.session_state.llm_response)

        with st.expander("Data sent to the LLM"):
            st.code(
                json.dumps(st.session_state.llm_data_sent, indent=2),
                language="json",
            )
    else:
        st.info(
            "Press **Send** to ask the LLM to summarise the last 48 hours of "
            f"**{LLM_METER}** readings. The API is called only when you press Send — "
            "simulation ticks and page refreshes do not trigger new requests.",
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
