import pandas as pd

# Load and sort
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Keep only what we need
df = df[["timestamp", "electricity_kwh"]].copy()

# Calendar features
df["hour"]       = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_of_week  # Monday=0, Sunday=6

# Lag features
df["lag_1h"]   = df["electricity_kwh"].shift(1)
df["lag_2h"]   = df["electricity_kwh"].shift(2)
df["lag_3h"]   = df["electricity_kwh"].shift(3)
df["lag_24h"]  = df["electricity_kwh"].shift(24)
df["lag_168h"] = df["electricity_kwh"].shift(168)

# Drop rows that don't yet have full history (first 168 rows)
df = df.dropna().reset_index(drop=True)

print("=== Shape after dropping incomplete rows ===")
print(f"Rows: {len(df)}  (started with 1440, dropped 168 → {1440 - len(df)} dropped)")

print("\n=== Date range after drop ===")
print(f"First: {df['timestamp'].min()}")
print(f"Last:  {df['timestamp'].max()}")

print("\n=== Feature columns and first 3 rows ===")
feature_cols = ["hour", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
print(df[["timestamp", "electricity_kwh"] + feature_cols].head(3).to_string(index=False))

print("\n=== Any nulls remaining? ===")
print(df.isnull().sum().to_string())

print("\n=== Feature stats ===")
print(df[feature_cols].describe().round(2).to_string())
