import time
import pandas as pd
from pathlib import Path
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from parse_logs import load_and_parse

OUT = Path("data/processed/templates.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)


def get_miner() -> TemplateMiner:
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = 0.4          # similarity threshold
    cfg.drain_depth = 4             # parse tree depth
    cfg.drain_max_children = 100
    cfg.profiling_enabled = False
    return TemplateMiner(config=cfg)


def extract(df: pd.DataFrame) -> pd.DataFrame:
    miner = get_miner()

    template_ids = []
    templates = []

    t0 = time.time()
    for msg in df["message"].tolist():
        result = miner.add_log_message(msg)
        template_ids.append(result["cluster_id"])
        templates.append(result["template_mined"])
    elapsed = time.time() - t0

    df = df.copy()
    df["template_id"] = template_ids
    df["template"] = templates

    print(f"Processed {len(df):,} lines in {elapsed:.1f}s")
    print(f"Unique templates: {df['template'].nunique():,}")
    return df


if __name__ == "__main__":
    df = load_and_parse()
    df = extract(df)

    print("\n--- Top 10 templates by frequency ---")
    top = df["template"].value_counts().head(10)
    for tmpl, cnt in top.items():
        print(f"{cnt:>6}  {tmpl[:110]}")

    # Save the processed version
    df.to_parquet(OUT, index=False)
    print(f"\nSaved processed data -> {OUT}")