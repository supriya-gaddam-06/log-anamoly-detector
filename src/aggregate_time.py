import numpy as np
import pandas as pd
from pathlib import Path

SCORES = Path("data/processed/anomaly_scores.parquet")
OUT = Path("data/processed/minute_anomalies.parquet")

df = pd.read_parquet(SCORES)
df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
df = df.dropna(subset=["ts"]).sort_values("ts")

df["minute"] = df["ts"].dt.floor("min")

per_min = df.groupby("minute").agg(
    n_lines=("raw", "size"),
    n_anomalies=("pred_is_anomaly", "sum"),
    mean_score=("anomaly_score", "mean"),
    max_score=("anomaly_score", "max"),
)
per_min["anomaly_rate"] = per_min["n_anomalies"] / per_min["n_lines"]
per_min["low_confidence"] = per_min["n_lines"] < 5

# Rank minutes by number of flagged anomalies, tie-break by rate
per_min["rank_score"] = per_min["n_anomalies"] * per_min["anomaly_rate"]

print(f"Total minutes: {len(per_min):,}")
print(f"Minutes with >=1 anomaly: {(per_min['n_anomalies'] > 0).sum():,}")
print(f"Total flagged lines: {per_min['n_anomalies'].sum():,}")

print("\n=== Top 20 minutes by anomaly count (min 5 lines) ===")
top = (
    per_min.query("n_lines >= 5")
    .sort_values(["n_anomalies", "anomaly_rate"], ascending=False)
    .head(20)
)
for ts, row in top.iterrows():
    print(f"  {ts}  lines={int(row['n_lines']):>4}  anom={int(row['n_anomalies']):>4}  "
          f"rate={row['anomaly_rate']:>6.2%}  max_score={row['max_score']:.4f}")

print("\n=== Top 20 minutes by anomaly rate (min 5 lines) ===")
top_r = (
    per_min.query("n_lines >= 5")
    .sort_values(["anomaly_rate", "n_anomalies"], ascending=False)
    .head(20)
)
for ts, row in top_r.iterrows():
    print(f"  {ts}  lines={int(row['n_lines']):>4}  anom={int(row['n_anomalies']):>4}  "
          f"rate={row['anomaly_rate']:>6.2%}  max_score={row['max_score']:.4f}")

# Save without the broken z-score column
per_min = per_min.drop(columns=[c for c in ["rate_rolling_mean", "rate_rolling_std", "rate_zscore"] if c in per_min.columns])
per_min.to_parquet(OUT)
print(f"\nSaved -> {OUT}")
print(f"Columns: {list(per_min.columns)}")