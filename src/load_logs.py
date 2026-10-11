import pandas as pd
from pathlib import Path

RAW = Path("data/raw/Apache.log")

def load_raw_log(path: Path = RAW) -> pd.DataFrame:
    """Load raw Apache log file as a single-column DataFrame."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    df = pd.DataFrame({"raw": [l.rstrip("\n") for l in lines]})
    return df

if __name__ == "__main__":
    df = load_raw_log()
    print(f"Total lines: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print("\n--- First 5 lines ---")
    for line in df["raw"].head(5):
        print(line[:120])
    print("\n--- Last 3 lines ---")
    for line in df["raw"].tail(3):
        print(line[:120])
    print("\n--- Line length stats ---")
    lengths = df["raw"].str.len()
    print(lengths.describe())