"""Concrete Curing & Strength Predictor API.

FastAPI backend, plain HTML frontend in frontend/index.html.
POST /api/predict with mix components (kg per cubic metre) and age
in days. Returns strength in MPa and PSI, water-to-cement ratio,
and the model-only inference time in milliseconds.

Honest limits:
- Trained on UCI Concrete Compressive Strength, 1,030 samples.
  No temperature or humidity in that dataset, so they are not
  model inputs here.
- This is a prediction from historical mixes, not a safety
  certification and not a replacement for a real 28-day test.
- Inference time reported is model-only. End-to-end phone time
  adds network and server time.
"""
import json
import time
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

MPA_TO_PSI = 145.038
FEATURES = ["cement", "slag", "fly_ash", "water", "superplasticizer",
            "coarse_agg", "fine_agg", "age_days", "water_cement_ratio"]

app = FastAPI(title="Concrete Curing & Strength Predictor")
MODEL_PATH = Path(__file__).parent / "model.joblib"
if not MODEL_PATH.exists():
    # model.joblib is a binary artifact, not stored in the repo.
    # Regenerate it from data/concrete.csv on first start (about 1 second).
    import runpy
    runpy.run_path(str(Path(__file__).parent / "train.py"), run_name="__main__")
model = joblib.load(MODEL_PATH)
try:
    METRICS = json.loads((Path(__file__).parent / "metrics.json").read_text())
except Exception:
    METRICS = {}


class Mix(BaseModel):
    cement: float = Field(gt=0)
    slag: float = Field(ge=0, default=0)
    fly_ash: float = Field(ge=0, default=0)
    water: float = Field(gt=0)
    superplasticizer: float = Field(ge=0, default=0)
    coarse_agg: float = Field(gt=0)
    fine_agg: float = Field(gt=0)
    age_days: float = Field(ge=1, le=365)


def predict_row(m: Mix, age: float):
    row = [[m.cement, m.slag, m.fly_ash, m.water, m.superplasticizer,
            m.coarse_agg, m.fine_agg, age, m.water / m.cement]]
    t0 = time.perf_counter()
    mpa = float(model.predict(row)[0])
    dt_ms = (time.perf_counter() - t0) * 1000
    return mpa, dt_ms


@app.get("/api/health")
def health():
    return {"status": "ok", "model": "XGBoost, UCI concrete, 1030 samples",
            "metrics": METRICS}


@app.post("/api/predict")
def predict(m: Mix):
    mpa, dt_ms = predict_row(m, m.age_days)
    curve = []
    for age in [1, 3, 7, 14, 28, 56, 90, 180, 365]:
        c_mpa, _ = predict_row(m, float(age))
        curve.append({"age_days": age, "strength_mpa": round(c_mpa, 2),
                      "strength_psi": round(c_mpa * MPA_TO_PSI)})
    return {
        "strength_mpa": round(mpa, 2),
        "strength_psi": round(mpa * MPA_TO_PSI),
        "water_cement_ratio": round(m.water / m.cement, 3),
        "inference_ms_model_only": round(dt_ms, 3),
        "curve": curve,
        "note": "Prediction from historical mixes, not a safety certification.",
    }


class TargetMix(Mix):
    target_psi: float = Field(gt=0)


@app.post("/api/target")
def target(m: TargetMix):
    """Earliest age in the standard curve at which predicted strength
    reaches or exceeds target_psi. Returns null if the curve never
    reaches it by 365 days. This is a model prediction, not a
    safety certification, and it does not replace a real test."""
    curve = []
    earliest = None
    for age in [1, 3, 7, 14, 28, 56, 90, 180, 365]:
        c_mpa, _ = predict_row(m, float(age))
        psi = round(c_mpa * MPA_TO_PSI)
        curve.append({"age_days": age, "strength_mpa": round(c_mpa, 2),
                      "strength_psi": psi})
        if earliest is None and psi >= m.target_psi:
            earliest = age
    return {
        "target_psi": m.target_psi,
        "earliest_age_days": earliest,
        "reached_by_365_days": earliest is not None,
        "max_strength_psi": max(c["strength_psi"] for c in curve),
        "curve": curve,
        "note": "Prediction from historical mixes, not a safety certification.",
    }


app.mount("/", StaticFiles(directory=Path(__file__).parent / "frontend", html=True), name="frontend")
