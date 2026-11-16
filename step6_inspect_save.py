import json
import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves to file without needing a display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from sklearn.ensemble import GradientBoostingRegressor

# ── Reproduce Steps 2 & 3 ────────────────────────────────────────────────────
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df = df[["timestamp", "electricity_kwh"]].copy()
df["hour"]        = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_of_week
df["lag_1h"]      = df["electricity_kwh"].shift(1)
df["lag_2h"]      = df["electricity_kwh"].shift(2)
df["lag_3h"]      = df["electricity_kwh"].shift(3)
df["lag_24h"]     = df["electricity_kwh"].shift(24)
df["lag_168h"]    = df["electricity_kwh"].shift(168)
df = df.dropna().reset_index(drop=True)

test_cutoff  = df["timestamp"].max() - pd.Timedelta(weeks=2)
train        = df[df["timestamp"] <= test_cutoff].copy()
test         = df[df["timestamp"] >  test_cutoff].copy()

feature_cols = ["hour", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
target_col   = "electricity_kwh"

X_train = train[feature_cols]
y_train = train[target_col]
X_test  = test[feature_cols]
y_test  = test[target_col]

# ── Train ─────────────────────────────────────────────────────────────────────
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
test_preds = model.predict(X_test)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test["timestamp"], y_test.values,  label="Actual",    color="steelblue",  linewidth=1.2)
ax.plot(test["timestamp"], test_preds,     label="Predicted", color="darkorange", linewidth=1.2, linestyle="--")
ax.set_title("Actual vs Predicted — two-week test window (METER_01)", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Electricity (kWh)")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=30, ha="right")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = pathlib.Path("models/forecast_plot.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Plot saved  : {plot_path}")

# ── Save model and feature order ──────────────────────────────────────────────
model_path   = pathlib.Path("models/gbr_model.joblib")
feature_path = pathlib.Path("models/feature_order.json")

joblib.dump(model, model_path)
print(f"Model saved : {model_path}")

with open(feature_path, "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"Features saved: {feature_path}")

# ── Verify files ──────────────────────────────────────────────────────────────
print("\n=== models/ directory ===")
for p in sorted(pathlib.Path("models").iterdir()):
    size_kb = p.stat().st_size / 1024
    print(f"  {p.name:<30}  {size_kb:>7.1f} KB")

# ── Reload and sanity-check the saved model ───────────────────────────────────
loaded_model = joblib.load(model_path)
reload_preds = loaded_model.predict(X_test)
match = (reload_preds == test_preds).all()
print(f"\nReloaded model predictions match originals: {match}")

print("\n=== feature_order.json contents ===")
with open(feature_path) as f:
    print(f.read())
