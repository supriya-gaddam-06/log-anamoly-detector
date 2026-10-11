import re
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/Apache.log")

# Regex for Apache error log format:
# [Thu Jun 09 06:07:04 2005] [notice] Message here
PATTERN = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\]\s+"
    r"\[(?P<level>[^\]]+)\]\s+"
    r"(?P<message>.*)$"
)

def parse_line(line: str) -> dict:
    m = PATTERN.match(line)
    if not m:
        return {"timestamp": None, "level": "UNKNOWN", "message": line, "parsed": False}
    d = m.groupdict()
    d["parsed"] = True
    return d

def load_and_parse(path: Path = RAW) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    rows = [parse_line(l) for l in lines]
    df = pd.DataFrame(rows)
    df["raw"] = lines
    return df

if __name__ == "__main__":
    df = load_and_parse()
    print(f"Total lines: {len(df):,}")
    print(f"Parsed successfully: {df['parsed'].sum():,} ({df['parsed'].mean():.1%})")
    print(f"\nLevel distribution:\n{df['level'].value_counts()}")
    print(f"\nSample parsed rows:")
    print(df[["timestamp", "level", "message"]].head(5).to_string())
    print(f"\nSample unparsed rows:")
    print(df[~df["parsed"]][["raw"]].head(5).to_string())