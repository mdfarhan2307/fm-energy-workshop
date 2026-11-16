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
test  = df[df["timestamp"] >  test_cutoff].copy()

feature_cols = ["hour", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
target_col   = "electricity_kwh"

X_train = train[feature_cols]
y_train = train[target_col]
X_test  = test[feature_cols]
y_test  = test[target_col]

# Train model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Model predictions on test set
test_preds = model.predict(X_test)

# Baseline: predict using the previous hour's reading (lag_1h)
baseline_preds = test["lag_1h"].values

# Metrics helper
def mae(actual, predicted):
    return (abs(actual - predicted)).mean()

def rmse(actual, predicted):
    return ((actual - predicted) ** 2).mean() ** 0.5

y = y_test.values

model_mae    = mae(y, test_preds)
model_rmse   = rmse(y, test_preds)
baseline_mae  = mae(y, baseline_preds)
baseline_rmse = rmse(y, baseline_preds)

print("=== Test set evaluation (336 hours, last 2 weeks) ===")
print(f"\n{'Metric':<8}  {'Baseline (lag_1h)':>20}  {'GBR Model':>12}  {'Improvement':>12}")
print("-" * 58)
print(f"{'MAE':<8}  {baseline_mae:>17.3f} kWh  {model_mae:>9.3f} kWh  {(baseline_mae - model_mae)/baseline_mae*100:>10.1f}%")
print(f"{'RMSE':<8}  {baseline_rmse:>17.3f} kWh  {model_rmse:>9.3f} kWh  {(baseline_rmse - model_rmse)/baseline_rmse*100:>10.1f}%")

print("\n=== Context: test set std dev ===")
print(f"electricity_kwh std on test set: {y_test.std():.3f} kWh")
print(f"Model MAE is {model_mae / y_test.std() * 100:.1f}% of test std dev")

print("\n=== Worst predictions (largest absolute errors) ===")
errors = abs(y - test_preds)
top5_idx = errors.argsort()[-5:][::-1]
print(f"{'Timestamp':<22} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
print("-" * 52)
for i in top5_idx:
    ts = test["timestamp"].iloc[i]
    print(f"{str(ts):<22} {y[i]:>8.2f} {test_preds[i]:>10.2f} {errors[i]:>8.2f}")
