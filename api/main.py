from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.preprocess import clean_text

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
LABELS = {-1: "negative", 0: "neutral", 1: "positive"}

model = joblib.load(MODEL_PATH)

app = FastAPI(title="VibeChecker API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    comments: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.comments:
        return {"labels": [], "counts": {name: 0 for name in LABELS.values()}, "total": 0}

    cleaned = [clean_text(c) for c in req.comments]
    preds = model.predict(cleaned)
    labels = [LABELS[int(p)] for p in preds]
    counts = {name: labels.count(name) for name in LABELS.values()}
    return {"labels": labels, "counts": counts, "total": len(labels)}