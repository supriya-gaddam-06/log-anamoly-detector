import pandas as pd

df = pd.read_parquet("data/processed/templates.parquet")
vc = df["template"].value_counts()
print(f"Total templates: {len(vc)}\n")
for tmpl, cnt in vc.items():
    print(f"{cnt:>6}  {tmpl[:120]}")