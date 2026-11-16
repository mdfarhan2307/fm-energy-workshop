"""
Energy Dashboard v2 — Live Simulation
======================================
Starts with all January 2016 readings already loaded.
Every real minute, fetches the next recorded hour from Neon (one reading per
meter, three rows total) and appends it to the charts.
Stops automatically when the last February 2016 record is reached.

Run from the project root:
    ..\\vibe\\Scripts\\streamlit.exe run dashboard/app_v2.py --server.headless true

Controls
--------
  Restart simulation  — resets to the January baseline and restarts the timer.
"""

import os
import re
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Dashboard v2",
    page_icon="⚡",
    layout="wide",
)

TICK_SECONDS = 60          # one real minute per simulated hour
JAN_END      = pd.Timestamp("2016-01-31 23:00:00", tz="UTC")
FEB_START    = pd.Timestamp("2016-02-01 00:00:00", tz="UTC")
DATA_END     = pd.Timestamp("2016-02-29 23:00:00", tz="UTC")

METER_COLORS = {
    "METER_01": "#00C2FF",
    "METER_02": "#FF6B35",
    "METER_03": "#7FBA00",
    "TOTAL":    "#A259FF",
}

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
    """Return (sql_url, headers) for the Neon HTTP SQL API."""
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    env = _load_env(env_path)
    conn = re.sub(r"[&?]channel_binding=require", "", env["DATABASE_URL"])
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

@st.cache_data(show_spinner="Loading January baseline from Neon…")
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
    """Return the three meter rows for the first timestamp strictly after `after`."""
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
    fig.update_layout(**_base_layout("Total — All Meters Combined", 320))
    return fig


# ── Session-state bootstrap ───────────────────────────────────────────────────
# `display_df`   — the DataFrame currently shown on the charts
# `last_ts`      — the latest timestamp already in display_df
# `sim_done`     — True once every February record has been shown
# `last_tick`    — wall-clock time when the most recent tick ran
# `tick_count`   — how many ticks have fired (for the status line)

def _init_state():
    jan = load_january()
    st.session_state.display_df  = jan.copy()
    st.session_state.last_ts     = JAN_END
    st.session_state.sim_done    = False
    st.session_state.last_tick   = time.time()
    st.session_state.tick_count  = 0


if "display_df" not in st.session_state:
    _init_state()

# ── Restart button ────────────────────────────────────────────────────────────
st.title("⚡ Energy Dashboard v2 — Live Simulation")
st.caption(
    "Data source: **historical** readings from `meter_readings_source` (Neon PostgreSQL). "
    "Simulation advances one recorded hour per real minute."
)

col_btn, col_status = st.columns([1, 5])
with col_btn:
    if st.button("🔄 Restart simulation", use_container_width=True):
        # Clear cached January load so it re-fetches cleanly, then re-init
        load_january.clear()
        _init_state()
        st.rerun()

# ── Tick logic ────────────────────────────────────────────────────────────────
# On every Streamlit re-run, check whether a minute has elapsed.
# If so, fetch the next hour and append it.

if not st.session_state.sim_done:
    now     = time.time()
    elapsed = now - st.session_state.last_tick

    if elapsed >= TICK_SECONDS:
        new_rows = fetch_next_hour(st.session_state.last_ts)

        if new_rows.empty:
            # No more data — simulation complete
            st.session_state.sim_done = True
        else:
            new_ts = new_rows["timestamp"].iloc[0]
            st.session_state.display_df = pd.concat(
                [st.session_state.display_df, new_rows], ignore_index=True
            )
            st.session_state.last_ts    = new_ts
            st.session_state.last_tick  = now
            st.session_state.tick_count += 1

            if new_ts >= DATA_END:
                st.session_state.sim_done = True

# ── Status line ───────────────────────────────────────────────────────────────
df   = st.session_state.display_df
last = st.session_state.last_ts

with col_status:
    if st.session_state.sim_done:
        st.success(
            f"✅ Simulation complete — all data shown through "
            f"{last.strftime('%d %b %Y %H:%M')} UTC.",
            icon=None,
        )
    else:
        ticks = st.session_state.tick_count
        next_tick_in = max(0, int(TICK_SECONDS - (time.time() - st.session_state.last_tick)))
        st.info(
            f"🕐 Showing up to **{last.strftime('%d %b %Y %H:%M')} UTC** "
            f"({ticks} hour{'s' if ticks != 1 else ''} streamed) — "
            f"next fetch in ~{next_tick_in}s",
            icon=None,
        )

# ── Charts — Row 1: individual meters ────────────────────────────────────────
st.subheader("Hourly Electricity Usage — Individual Meters")
meter_ids = sorted(df["meter_id"].unique())
cols = st.columns(3)
for col, meter_id in zip(cols, meter_ids):
    with col:
        st.plotly_chart(meter_chart(df, meter_id), use_container_width=True)

st.divider()

# ── Charts — Row 2: total ─────────────────────────────────────────────────────
st.subheader("Total Electricity Usage — All Meters")
st.plotly_chart(total_chart(df), use_container_width=True)

st.divider()

# ── Placeholders ─────────────────────────────────────────────────────────────
left, right = st.columns(2)
with left:
    st.subheader("🔮 Prediction")
    st.info(
        "**Coming soon.** This section will display a short-term energy usage forecast "
        "generated from the historical readings.",
        icon="🔮",
    )
with right:
    st.subheader("🤖 LLM Analysis")
    st.info(
        "**Coming soon.** This section will display an AI-generated commentary on "
        "recent energy patterns, powered by the OpenAI API.",
        icon="🤖",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption(
    f"Showing **{len(df):,}** readings across **{len(meter_ids)}** meters "
    f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()})."
)

# ── Auto-rerun while simulation is running ────────────────────────────────────
# Schedule the next rerun just before the next tick is due so the UI stays
# responsive without hammering the server.
if not st.session_state.sim_done:
    seconds_until_next = max(1, int(TICK_SECONDS - (time.time() - st.session_state.last_tick)))
    time.sleep(1)           # small delay so Streamlit does not spin at 100 %
    st.rerun()
