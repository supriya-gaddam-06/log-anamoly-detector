import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from parse_logs import parse_line

MODEL_PATH = Path("models/iforest.pkl")
ENCODER_PATH = Path("models/encoder.pkl")
TEMPLATES_PATH = Path("models/templates.json")
FREQ_META = Path("data/processed/feature_meta.parquet")

app = FastAPI(title="Log Anomaly Detector", version="1.0")

_model = None
_encoder = None
_templates = None
_template_freq = None
_level_columns = None


def _lazy_load():
    global _model, _encoder, _templates, _template_freq, _level_columns
    if _model is not None:
        return
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists() or not TEMPLATES_PATH.exists():
        raise HTTPException(500, "Model artifacts missing. Run training first.")
    _model = joblib.load(MODEL_PATH)
    _encoder = joblib.load(ENCODER_PATH)
    _templates = set(json.loads(TEMPLATES_PATH.read_text()))

    meta = pd.read_parquet(FREQ_META)
    _template_freq = meta["template"].value_counts().to_dict()

    # fixed level ordering matching training one-hot
    _level_columns = ["lvl_UNKNOWN", "lvl_error", "lvl_notice", "lvl_warn"]


class LogLine(BaseModel):
    line: str


class LogBatch(BaseModel):
    lines: List[str]


def _assign_template(message: str) -> str:
    """Map a message to a known template string (exact match), else <UNKNOWN>."""
    if message in _templates:
        return message
    return "<UNKNOWN>"


def _build_features(lines: List[str]) -> np.ndarray:
    parsed = [parse_line(l) for l in lines]
    df = pd.DataFrame(parsed)
    df["raw"] = lines

    df["ts"] = pd.to_datetime(df["timestamp"], format="%a %b %d %H:%M:%S %Y", errors="coerce")
    df["hour"] = df["ts"].dt.hour
    df["hour_known"] = df["hour"].notna().astype(int)
    df["hour"] = df["hour"].fillna(0).astype(int)

    df["template"] = df["message"].apply(_assign_template)
    df["template_freq"] = df["template"].map(_template_freq).fillna(1).astype(np.float32)
    df["template_rarity"] = (1.0 / df["template_freq"]).astype(np.float32)
    df["template_freq_log"] = np.log1p(df["template_freq"]).astype(np.float32)

    # Template one-hot via the persisted encoder
    tmpl_ohe = _encoder.transform(df[["template"]]).astype(np.float32)

    numeric_cols = ["template_rarity", "template_freq_log", "hour", "hour_known"]
    numeric = df[numeric_cols].to_numpy(dtype=np.float32)

    # Level one-hot, fixed order
    level_map = {col: (df["level"] == col.replace("lvl_", "")).astype(np.float32).to_numpy()
                 for col in _level_columns}
    level = np.stack([level_map[c] for c in _level_columns], axis=1)

    return np.hstack([tmpl_ohe, numeric, level])


def _score_batch(lines: List[str]) -> List[Dict[str, Any]]:
    _lazy_load()
    X = _build_features(lines)
    labels = _model.predict(X)
    scores = -_model.decision_function(X)
    out = []
    for i, line in enumerate(lines):
        out.append({
            "line": line[:200],
            "is_anomaly": bool(labels[i] == -1),
            "anomaly_score": float(scores[i]),
        })
    return out


@app.get("/health")
def health():
    return {"status": "ok", "artifacts_present": MODEL_PATH.exists() and ENCODER_PATH.exists()}


@app.post("/score")
def score(batch: LogBatch):
    if not batch.lines:
        raise HTTPException(400, "Empty batch")
    return {"results": _score_batch(batch.lines)}


@app.post("/score_one")
def score_one(item: LogLine):
    return _score_batch([item.line])[0]