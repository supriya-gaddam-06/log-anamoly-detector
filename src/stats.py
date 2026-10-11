from parse_logs import load_and_parse

df = load_and_parse()
print(f"Total lines:       {len(df):,}")
print(f"Unique raw lines:  {df['raw'].nunique():,}")
print(f"Duplicates:        {len(df) - df['raw'].nunique():,}")
print(f"Unique messages:   {df['message'].nunique():,}")