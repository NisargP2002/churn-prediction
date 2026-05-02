import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DIR, PROC_DIR, TARGET_COL, CUSTOMER_ID


def load_raw(filename="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}. Did you download it?")
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_schema(df):
    required = [CUSTOMER_ID, TARGET_COL, "tenure", "MonthlyCharges", "TotalCharges"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # TotalCharges has spaces for new customers — fix silently
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_fixed = df["TotalCharges"].isna().sum()
    if n_fixed > 0:
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
        print(f"Fixed {n_fixed} blank TotalCharges values")

    print(f"Schema valid. No missing required columns.")
    return df


def encode_target(df):
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)
    churn_rate = df[TARGET_COL].mean()
    churn_count = df[TARGET_COL].sum()
    print(f"Churn rate: {churn_rate:.1%}  ({churn_count} churners out of {len(df)})")
    return df


def quick_summary(df):
    print("\n--- Data Summary ---")
    print(f"Shape         : {df.shape}")
    print(f"Null values   : {df.isnull().sum().sum()}")
    print(f"Duplicates    : {df.duplicated().sum()}")
    print(f"Numeric cols  : {list(df.select_dtypes('number').columns)}")
    print(f"Object cols   : {list(df.select_dtypes('object').columns)}")
    print("--------------------\n")


def save_interim(df, name="interim.parquet"):
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROC_DIR / name
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    df = load_raw()
    df = validate_schema(df)
    df = encode_target(df)
    quick_summary(df)
    save_interim(df)
    print("\nDay 1 complete. Run: python src/data_loader.py")