import pandas as pd

# Load the data
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Basic shape and types
print("=== Shape ===")
print(f"Rows: {len(df)}, Columns: {list(df.columns)}")

print("\n=== Date range ===")
print(f"First: {df['timestamp'].min()}")
print(f"Last:  {df['timestamp'].max()}")
expected_hours = int((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600) + 1
print(f"Expected hourly rows: {expected_hours}, Actual rows: {len(df)}")
missing_hours = expected_hours - len(df)
print(f"Missing hours: {missing_hours}")

print("\n=== electricity_kwh stats ===")
print(df["electricity_kwh"].describe().round(2))

print("\n=== Null values ===")
print(df.isnull().sum())

print("\n=== Duplicate timestamps ===")
print(df.duplicated(subset="timestamp").sum())

print("\n=== Unique meter / building ===")
print(f"meter_id:    {df['meter_id'].unique()}")
print(f"building_id: {df['building_id'].unique()}")

print("\n=== Sample of first 3 and last 3 rows ===")
print(pd.concat([df.head(3), df.tail(3)]).to_string(index=False))
