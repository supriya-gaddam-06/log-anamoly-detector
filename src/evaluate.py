import numpy as np
import pandas as pd
from pathlib import Path

SCORES = Path("data/processed/anomaly_scores.parquet")
df = pd.read_parquet(SCORES)

# --- 1. Score distribution ---
print("=== Anomaly score distribution ===")
print(df["anomaly_score"].describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.995, 0.999]))

# --- 2. Threshold sweep ---
print("\n=== Threshold sweep: how many lines flagged at each cutoff? ===")
for thr in [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
    n = (df["anomaly_score"] >= thr).sum()
    pct = n / len(df) * 100
    print(f"  score >= {thr:.2f}  ->  {n:>6,} lines ({pct:>5.2f}%)")

# --- 3. What does the "normal" tail look like? ---
print("\n=== 5 least-anomalous lines (most normal) ===")
bottom = df.sort_values("anomaly_score").head(5)
for _, r in bottom.iterrows():
    print(f"  score={r['anomaly_score']:.4f}  {r['raw'][:110]}")

# --- 4. Anomaly rate by hour ---
print("\n=== Anomaly rate by hour of day ===")
df["hour"] = pd.to_datetime(df["ts"], errors="coerce").dt.hour
by_hour = df.groupby("hour")["pred_is_anomaly"].agg(["sum", "count"])
by_hour["rate_pct"] = (by_hour["sum"] / by_hour["count"] * 100).round(2)
print(by_hour.to_string())

# --- 5. Anomaly rate by level ---
print("\n=== Anomaly rate by level ===")
by_lvl = df.groupby("level")["pred_is_anomaly"].agg(["sum", "count"])
by_lvl["rate_pct"] = (by_lvl["sum"] / by_lvl["count"] * 100).round(2)
print(by_lvl.to_string())