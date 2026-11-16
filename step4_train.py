import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Load and sort
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Reproduce Steps 2 & 3
df = df[["timestamp", "electricity_kwh"]].copy()
df["hour"]        = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_of_week
df["lag_1h"]      = df["electricity_kwh"].shift(1)
df["lag_2h"]      = df["electricity_kwh"].shift(2)
df["lag_3h"]      = df["electricity_kwh"].shift(3)
df["lag_24h"]     = df["electricity_kwh"].shift(24)
df["lag_168h"]    = df["electricity_kwh"].shift(168)
df = df.dropna().reset_index(drop=True)

test_cutoff = df["timestamp"].max() - pd.Timedelta(weeks=2)
train = df[df["timestamp"] <= test_cutoff].copy()

feature_cols = ["hour", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
target_col   = "electricity_kwh"

X_train = train[feature_cols]
y_train = train[target_col]

# Train
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

print("=== Model trained ===")
print(f"Algorithm   : GradientBoostingRegressor")
print(f"random_state: 42")
print(f"n_estimators: {model.n_estimators}  (default)")
print(f"max_depth   : {model.max_depth}  (default)")
print(f"learning_rate: {model.learning_rate}  (default)")
print(f"Trained on  : {len(X_train)} rows")

print("\n=== Training set performance (in-sample) ===")
train_preds = model.predict(X_train)
train_mae  = (abs(y_train - train_preds)).mean()
train_rmse = ((y_train - train_preds) ** 2).mean() ** 0.5
print(f"MAE  : {train_mae:.3f} kWh")
print(f"RMSE : {train_rmse:.3f} kWh")

print("\n=== Feature importances ===")
for feat, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
    bar = "#" * int(imp * 50)
    print(f"  {feat:12s}  {imp:.3f}  {bar}")
