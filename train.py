"""Train the concrete strength model.

Data: UCI Concrete Compressive Strength, 1,030 real samples,
loaded into data/concrete.db (see data/concrete.csv).
Features are mix components in kg per cubic metre, plus age in days.
water_cement_ratio is derived as water / cement, not a new measurement.
Target is compressive strength in MPa. PSI = MPa * 145.038.

Honest limits, stated in the README and the app:
- This dataset has NO temperature or humidity columns. Temperature
  and humidity are therefore NOT model inputs. If a user idea lists
  them, they are not claimed here, because the training data does
  not contain them.
- This is a prediction from historical lab mixes, not a safety
  certification. It does not replace a real 28-day test.
"""
import json
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).parent

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

FEATURES = ["cement", "slag", "fly_ash", "water", "superplasticizer",
            "coarse_agg", "fine_agg", "age_days", "water_cement_ratio"]
MPA_TO_PSI = 145.038

try:
    con = sqlite3.connect(ROOT / "data" / "concrete.db")
    df = pd.read_sql("SELECT * FROM samples", con)
    con.close()
except Exception:
    df = pd.read_csv(ROOT / "data" / "concrete.csv")
df["water_cement_ratio"] = df["water"] / df["cement"]

X = df[FEATURES]
y = df["strength_mpa"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

model = XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    reg_lambda=2.0, reg_alpha=0.1,  # regularized, as the idea asked
    random_state=42, n_jobs=-1)

t0 = time.perf_counter()
model.fit(X_train, y_train)
fit_s = time.perf_counter() - t0

pred = model.predict(X_test)
rmse = float(mean_squared_error(y_test, pred) ** 0.5)
mae = float(mean_absolute_error(y_test, pred))
r2 = float(r2_score(y_test, pred))

# Single-row inference latency, model only (not network, not cold start).
row = X_test.iloc[[0]]
for _ in range(20):
    model.predict(row)
times = []
for _ in range(200):
    s = time.perf_counter()
    model.predict(row)
    times.append((time.perf_counter() - s) * 1000)
times.sort()
metrics = {
    "n_samples": int(len(df)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "features": FEATURES,
    "rmse_mpa": rmse,
    "rmse_psi": rmse * MPA_TO_PSI,
    "mae_mpa": mae,
    "mae_psi": mae * MPA_TO_PSI,
    "r2": r2,
    "fit_seconds": fit_s,
    "inference_ms_p50_model_only": times[len(times) // 2],
    "inference_ms_p99_model_only": times[int(len(times) * 0.99)],
    "note": "Inference time is model-only. End-to-end phone time adds network and server time.",
}
joblib.dump(model, ROOT / "model.joblib")
with open(ROOT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
