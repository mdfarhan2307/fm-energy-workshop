import csv
import os
import re
import requests

# ── Load DATABASE_URL from .env ───────────────────────────────────────────────
env_path = os.path.join(os.path.dirname(__file__), ".env")
database_url = None
with open(env_path) as f:
    for line in f:
        m = re.match(r'^DATABASE_URL\s*=\s*"?([^"\n]+)"?', line.strip())
        if m:
            database_url = m.group(1).strip()
            break

if not database_url:
    raise RuntimeError("DATABASE_URL not found in .env")

# Strip channel_binding=require (unsupported by HTTP API)
database_url = re.sub(r'[&?]channel_binding=require', '', database_url)
print(f"Using connection: {re.sub(r':([^:@]+)@', ':***@', database_url)}")

# ── Derive HTTPS SQL endpoint from connection string ──────────────────────────
# postgresql://user:pass@host/db?... → https://host/sql
m = re.match(r'postgresql://[^@]+@([^/?]+)', database_url)
if not m:
    raise RuntimeError("Cannot parse host from DATABASE_URL")
host = m.group(1)
sql_url = f"https://{host}/sql"
headers = {
    "Content-Type": "application/json",
    "Neon-Connection-String": database_url,
}

def run_sql(query, params=None):
    payload = {"query": query}
    if params:
        payload["params"] = params
    resp = requests.post(sql_url, json=payload, headers=headers, timeout=30)
    if not resp.ok:
        print(f"HTTP {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    return resp.json()

# ── Create table ──────────────────────────────────────────────────────────────
print("Creating table meter_readings_source …")
run_sql("""
CREATE TABLE IF NOT EXISTS meter_readings_source (
    meter_id        TEXT        NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    building_id     TEXT        NOT NULL,
    electricity_kwh NUMERIC     NOT NULL,
    CONSTRAINT meter_readings_source_pk PRIMARY KEY (meter_id, timestamp)
)
""")
print("Table ready.")

# ── Load CSV and upload in batches ────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "Data", "three_meters.csv")

rows = []
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append((
            row["meter_id"],
            row["timestamp"],
            row["building_id"],
            row["electricity_kwh"],
        ))

print(f"CSV rows read: {len(rows)}")

# Insert in batches using unnest for efficiency
BATCH = 500
inserted = 0
for start in range(0, len(rows), BATCH):
    batch = rows[start:start + BATCH]
    # Build a VALUES list
    placeholders = ", ".join(
        f"(${ (i*4)+1 }, ${ (i*4)+2 }, ${ (i*4)+3 }, ${ (i*4)+4 })"
        for i in range(len(batch))
    )
    flat_params = []
    for r in batch:
        flat_params.extend(r)
    query = f"""
        INSERT INTO meter_readings_source (meter_id, timestamp, building_id, electricity_kwh)
        VALUES {placeholders}
        ON CONFLICT (meter_id, timestamp) DO NOTHING
    """
    run_sql(query, flat_params)
    inserted += len(batch)
    print(f"  Uploaded {inserted}/{len(rows)} rows …")

print("Upload complete.")

# ── Summary query ─────────────────────────────────────────────────────────────
print("\nSummary per meter:")
result = run_sql("""
    SELECT
        meter_id,
        COUNT(*)            AS row_count,
        MIN(timestamp)      AS earliest,
        MAX(timestamp)      AS latest
    FROM meter_readings_source
    GROUP BY meter_id
    ORDER BY meter_id
""")

# Print table
cols = [c["name"] for c in result["fields"]]
print(f"{'meter_id':<12}  {'rows':>6}  {'earliest':^25}  {'latest':^25}")
print("-" * 75)
for r in result["rows"]:
    print(f"{r['meter_id']:<12}  {r['row_count']:>6}  {r['earliest']:^25}  {r['latest']:^25}")
