"""
Energy Dashboard
================
Reads historical meter readings from Neon PostgreSQL and displays:
  - Three side-by-side hourly electricity charts (one per meter)
  - A total electricity chart (all meters summed per timestamp)
  - Placeholder sections for Prediction and LLM Analysis

Run from the project root:
    ..\\vibe\\Scripts\\streamlit.exe run dashboard/app.py --server.headless true
"""

import os
import re

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Dashboard",
    page_icon="⚡",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_env(path: str) -> dict:
    """Parse key=value pairs from a .env file."""
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


def get_connection_string() -> str:
    # On Streamlit Cloud use st.secrets; locally fall back to .env
    if "DATABASE_URL" in st.secrets:
        url = st.secrets["DATABASE_URL"]
    else:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        env = load_env(os.path.abspath(env_path))
        url = env.get("DATABASE_URL", "")
    # Strip channel_binding=require — not supported by the HTTP API
    url = re.sub(r"[&?]channel_binding=require", "", url)
    return url


def neon_sql_endpoint(conn: str) -> tuple[str, dict]:
    """Return (sql_url, headers) for the Neon HTTP SQL API."""
    m = re.match(r"postgresql://[^@]+@([^/?]+)", conn)
    if not m:
        raise ValueError("Cannot parse host from DATABASE_URL")
    host = m.group(1)
    url = f"https://{host}/sql"
    headers = {
        "Content-Type": "application/json",
        "Neon-Connection-String": conn,
    }
    return url, headers


@st.cache_data(show_spinner="Loading data from Neon…")
def fetch_readings() -> pd.DataFrame:
    conn = get_connection_string()
    sql_url, headers = neon_sql_endpoint(conn)

    query = """
        SELECT meter_id, timestamp, electricity_kwh
        FROM   meter_readings_source
        ORDER  BY timestamp, meter_id
    """
    resp = requests.post(sql_url, json={"query": query}, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = data.get("rows", [])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"])
    return df


# ── Colour palette (one per meter + total) ────────────────────────────────────
METER_COLORS = {
    "METER_01": "#00C2FF",   # cyan
    "METER_02": "#FF6B35",   # orange
    "METER_03": "#7FBA00",   # green
    "TOTAL":    "#A259FF",   # purple
}

# ── Chart builders ────────────────────────────────────────────────────────────

def meter_chart(df: pd.DataFrame, meter_id: str) -> go.Figure:
    sub = df[df["meter_id"] == meter_id].sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["timestamp"],
        y=sub["electricity_kwh"],
        mode="lines",
        name=meter_id,
        line=dict(color=METER_COLORS[meter_id], width=1.5),
        fill="tozeroy",
        fillcolor=METER_COLORS[meter_id].replace("FF", "22").replace("#", "rgba(").rstrip(")") if False else None,
    ))
    fig.update_layout(
        title=dict(text=meter_id, font=dict(size=14)),
        xaxis=dict(title="Time", showgrid=False),
        yaxis=dict(title="kWh", showgrid=True, gridcolor="#2a2a2a"),
        margin=dict(l=40, r=10, t=40, b=40),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        hovermode="x unified",
        height=280,
    )
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
    fig.update_layout(
        title=dict(text="Total — All Meters Combined", font=dict(size=14)),
        xaxis=dict(title="Time", showgrid=False),
        yaxis=dict(title="kWh", showgrid=True, gridcolor="#2a2a2a"),
        margin=dict(l=40, r=10, t=40, b=40),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        hovermode="x unified",
        height=320,
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("⚡ Energy Dashboard")
st.caption("Data source: **historical** readings from `meter_readings_source` (Neon PostgreSQL)")

# Load data
df = fetch_readings()

meter_ids = sorted(df["meter_id"].unique())

# ── Row 1: three meter charts side by side ────────────────────────────────────
st.subheader("Hourly Electricity Usage — Individual Meters")
cols = st.columns(3)
for col, meter_id in zip(cols, meter_ids):
    with col:
        st.plotly_chart(meter_chart(df, meter_id), use_container_width=True)

st.divider()

# ── Row 2: total chart ────────────────────────────────────────────────────────
st.subheader("Total Electricity Usage — All Meters")
st.plotly_chart(total_chart(df), use_container_width=True)

st.divider()

# ── Row 3: placeholders ───────────────────────────────────────────────────────
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
