import pandas as pd

# Load and sort
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Reproduce Step 2 preparation
df = df[["timestamp", "electricity_kwh"]].copy()
df["hour"]        = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_of_week
df["lag_1h"]      = df["electricity_kwh"].shift(1)
df["lag_2h"]      = df["electricity_kwh"].shift(2)
df["lag_3h"]      = df["electricity_kwh"].shift(3)
df["lag_24h"]     = df["electricity_kwh"].shift(24)
df["lag_168h"]    = df["electricity_kwh"].shift(168)
df = df.dropna().reset_index(drop=True)

# Split: hold back last two weeks (14 days = 336 hours) as test set
test_cutoff = df["timestamp"].max() - pd.Timedelta(weeks=2)
train = df[df["timestamp"] <= test_cutoff].copy()
test  = df[df["timestamp"] >  test_cutoff].copy()

feature_cols = ["hour", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
target_col   = "electricity_kwh"

X_train = train[feature_cols]
y_train = train[target_col]
X_test  = test[feature_cols]
y_test  = test[target_col]

print("=== Split summary ===")
print(f"Total usable rows : {len(df)}")
print(f"Train rows        : {len(train)}")
print(f"Test rows         : {len(test)}")
print(f"Train period      : {train['timestamp'].min()}  →  {train['timestamp'].max()}")
print(f"Test period       : {test['timestamp'].min()}  →  {test['timestamp'].max()}")
print(f"\nTrain % of data   : {len(train)/len(df)*100:.1f}%")
print(f"Test  % of data   : {len(test)/len(df)*100:.1f}%")

print("\n=== Target stats — train vs test ===")
print(f"{'':20s} {'train':>10} {'test':>10}")
for stat, t, v in zip(
    ["mean", "std", "min", "max"],
    [y_train.mean(), y_train.std(), y_train.min(), y_train.max()],
    [y_test.mean(),  y_test.std(),  y_test.min(),  y_test.max()],
):
    print(f"  {stat:18s} {t:>10.2f} {v:>10.2f}")
