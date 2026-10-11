import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

PROCESSED = Path("data/processed/templates.parquet")
FEATURES = Path("data/processed/features.npz")
FEATURE_META = Path("data/processed/feature_meta.parquet")
ENCODER_PATH = Path("models/encoder.pkl")
TEMPLATES_PATH = Path("models/templates.json")


def build(df: pd.DataFrame):
    df = df.copy()

    # --- 1. Parse timestamp into datetime ---
    df["ts"] = pd.to_datetime(df["timestamp"], format="%a %b %d %H:%M:%S %Y", errors="coerce")
    df["hour"] = df["ts"].dt.hour
    df["hour_known"] = df["hour"].notna().astype(int)
    df["hour"] = df["hour"].fillna(0).astype(int)

    # --- 2. Template rarity ---
    vc = df["template"].value_counts()
    df["template_freq"] = df["template"].map(vc).astype(np.float32)
    df["template_rarity"] = (1.0 / df["template_freq"]).astype(np.float32)
    df["template_freq_log"] = np.log1p(df["template_freq"]).astype(np.float32)

    # --- 3. Level one-hot ---
    level_dummies = pd.get_dummies(df["level"], prefix="lvl")

    # --- 4. Template one-hot ---
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    template_ohe = ohe.fit_transform(df[["template"]])
    template_cols = [f"tmpl_{i}" for i in range(template_ohe.shape[1])]

    # --- 5. Assemble ---
    numeric_cols = ["template_rarity", "template_freq_log", "hour", "hour_known"]
    numeric = df[numeric_cols].to_numpy(dtype=np.float32)
    level = level_dummies.to_numpy(dtype=np.float32)

    X = np.hstack([template_ohe.astype(np.float32), numeric, level])
    feature_names = template_cols + numeric_cols + list(level_dummies.columns)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Numeric features: {numeric_cols}")
    print(f"Level features:   {list(level_dummies.columns)}")

    # --- 6. Persist encoder + template list for inference ---
    ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ohe, ENCODER_PATH)
    TEMPLATES_PATH.write_text(json.dumps(sorted(df["template"].unique().tolist())))
    print(f"Saved encoder   -> {ENCODER_PATH}")
    print(f"Saved templates -> {TEMPLATES_PATH}")

    meta = df[["raw", "ts", "level", "template", "template_id",
               "template_freq", "template_rarity"]].copy()
    return X, feature_names, meta


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED)
    X, names, meta = build(df)

    np.savez_compressed(FEATURES, X=X, names=names)
    meta.to_parquet(FEATURE_META, index=False)

    print(f"\nSaved features -> {FEATURES}")
    print(f"Saved meta     -> {FEATURE_META}")
    print(f"\nFeature matrix stats:")
    print(f"  Mean:  {X.mean():.4f}")
    print(f"  Std:   {X.std():.4f}")
    print(f"  Min:   {X.min():.4f}")
    print(f"  Max:   {X.max():.4f}")