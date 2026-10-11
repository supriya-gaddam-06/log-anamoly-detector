import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest

FEATURES = Path("data/processed/features.npz")
META = Path("data/processed/feature_meta.parquet")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "iforest.pkl"
SCORES_PATH = Path("data/processed/anomaly_scores.parquet")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train(X: np.ndarray, contamination: float = 0.01, seed: int = 42):
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    t0 = time.time()
    model.fit(X)
    print(f"Trained in {time.time() - t0:.1f}s  |  contamination={contamination}")
    return model


if __name__ == "__main__":
    data = np.load(FEATURES, allow_pickle=True)
    X = data["X"]
    names = data["names"].tolist()
    meta = pd.read_parquet(META)

    print(f"Loaded features: {X.shape}, {len(names)} names")
    print(f"Meta rows: {len(meta):,}")

    model = train(X, contamination=0.01)

    labels = model.predict(X)
    scores = model.decision_function(X)
    anomaly_score = -scores

    meta = meta.copy()
    meta["pred_label"] = labels
    meta["pred_is_anomaly"] = (labels == -1).astype(int)
    meta["anomaly_score"] = anomaly_score

    n_anom = meta["pred_is_anomaly"].sum()
    print(f"\nFlagged anomalies: {n_anom:,} / {len(meta):,} ({n_anom / len(meta):.2%})")

    print("\n--- Top 15 most anomalous lines ---")
    top = meta.sort_values("anomaly_score", ascending=False).head(15)
    for _, row in top.iterrows():
        print(f"  score={row['anomaly_score']:.4f}  lvl={row['level']:<8}  {row['raw'][:110]}")

    print("\n--- Anomaly distribution by template (top 10) ---")
    by_tmpl = (
        meta[meta["pred_is_anomaly"] == 1]
        .groupby("template")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    for tmpl, cnt in by_tmpl.items():
        print(f"  {cnt:>6}  {tmpl[:110]}")

    joblib.dump(model, MODEL_PATH)
    meta.to_parquet(SCORES_PATH, index=False)
    print(f"\nSaved model  -> {MODEL_PATH}")
    print(f"Saved scores -> {SCORES_PATH}")