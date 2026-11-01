# Concrete Curing & Strength Predictor

A web tool for a job-site supervisor: enter a concrete mix and its age,
get a predicted compressive strength in PSI and MPa, plus a strength
curve from 1 to 365 days for that same mix.

## Data, honestly
- Real data: UCI Concrete Compressive Strength, 1,030 samples.
  `data/concrete.csv` and `data/concrete.db` (table `samples`).
- Features: cement, blast furnace slag, fly ash, water,
  superplasticizer, coarse aggregate, fine aggregate, all in kg per
  cubic metre, plus age in days. Water-to-cement ratio is derived
  as water / cement.
- **No temperature or humidity.** That dataset has no weather
  columns, so temperature and humidity are NOT model inputs here,
  even though the original idea listed them. Adding them would
  require a dataset that actually contains them.
- This is a prediction from historical mixes, not a safety
  certification and not a replacement for a real 28-day test.

## Model
Regularized XGBoost (`reg_lambda=2.0, reg_alpha=0.1`, 300 trees,
depth 6, learning rate 0.05), train/test split 80/20, seed 42.
Saved to `model.joblib`. Metrics in `metrics.json`.

Verified on this machine, test set of 206 samples:
- R2 0.92
- RMSE 4.54 MPa / 658 PSI
- MAE 2.99 MPa / 434 PSI
- Model-only single-row inference p50 1.15 ms

That last number is model-only. End-to-end phone time adds network
and server time, so do not claim under 10 ms for the full round trip.

## Run locally
```
python3 -m venv .venv
.venv/bin/pip install --no-cache-dir -r requirements.txt
.venv/bin/python train.py
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

## Deploy on Replit
Import this repo into Replit, install from `requirements.txt`, and run
`uvicorn app:app --host 0.0.0.0 --port 8000`. Replit gives you a
public URL. No API key and no database server are needed.
Note: `model.joblib` is a binary artifact and is not stored in the
repo, so the app regenerates it from `data/concrete.csv` on first
start (about 1 second), or you can run `python train.py` first.

## API
- `GET /api/health` — status and test metrics
- `POST /api/predict` — mix + age, returns MPa, PSI,
  water-to-cement ratio, model-only inference time, and the age curve
