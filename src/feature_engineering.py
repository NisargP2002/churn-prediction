import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    TARGET_COL, CUSTOMER_ID, TEST_SIZE,
    RANDOM_STATE, PROC_DIR, SMOTE_RATIO
)

# ── Constants ────────────────────────────────────────────────

BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "MultipleLines", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies"
]

PROTECTIVE_SERVICES = [
    "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport"
]

ORDINAL_CONTRACT = {
    "Month-to-month": 0,
    "One year":       1,
    "Two year":       2
}

ORDINAL_INTERNET = {
    "No":           0,
    "DSL":          1,
    "Fiber optic":  2
}

ORDINAL_PAYMENT = {
    "Electronic check":             0,
    "Mailed check":                 1,
    "Bank transfer (automatic)":    2,
    "Credit card (automatic)":      3
}


# ── Step 1: encode binary columns ───────────────────────────

def encode_binary(df):
    """
    Insight: Yes/No and Male/Female columns → 0/1
    Covers 12 service columns + gender + demographics
    """
    df = df.copy()
    mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    print(f"  Binary encoded: {len(BINARY_COLS)} columns")
    return df


# ── Step 2: encode categoricals ─────────────────────────────

def encode_categoricals(df):
    """
    Insight 3 from EDA: Contract has strong ordinal relationship
    with churn — Month-to-month worst, Two year best.
    Use ordinal encoding (not one-hot) to preserve that order.
    """
    df = df.copy()
    df["Contract"]       = df["Contract"].map(ORDINAL_CONTRACT)
    df["InternetService"] = df["InternetService"].map(ORDINAL_INTERNET)
    df["PaymentMethod"]  = df["PaymentMethod"].map(ORDINAL_PAYMENT)

    # Fill any unmapped values with 0
    for col in ["Contract", "InternetService", "PaymentMethod"]:
        df[col] = df[col].fillna(0).astype(int)

    print(f"  Ordinal encoded: Contract, InternetService, PaymentMethod")
    return df


# ── Step 3: engineer new features ───────────────────────────

def engineer_features(df):
    """
    All 6 insights from EDA become features here.
    """
    df = df.copy()

    # Insight 1 — new customers churn most
    df["IsNewCustomer"] = (df["tenure"] < 12).astype(int)

    # Insight 2 — tenure bucket (ordinal risk tier)
    df["TenureBucket"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=[4, 3, 2, 1, 0],   # higher = higher risk
        include_lowest=True
    ).astype(int)

    # Insight 4 — charge per month of tenure (value signal)
    df["ChargePerTenure"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]   # fallback for 0-tenure customers
    )

    # Insight 5 & 6 — number of ALL services (engagement)
    all_services = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    df["NumServices"] = df[all_services].sum(axis=1)

    # Insight 6 — protective services specifically reduce churn
    df["NumProtectiveServices"] = df[PROTECTIVE_SERVICES].sum(axis=1)

    # High risk flag — combines top 3 risk signals into one
    df["HighRiskFlag"] = (
        (df["Contract"]      == 0) &   # Month-to-month
        (df["IsNewCustomer"] == 1) &   # tenure < 12
        (df["InternetService"] == 2)   # Fiber optic
    ).astype(int)

    new_features = [
        "IsNewCustomer", "TenureBucket", "ChargePerTenure",
        "NumServices", "NumProtectiveServices", "HighRiskFlag"
    ]
    print(f"  Engineered {len(new_features)} new features: {new_features}")
    return df


# ── Step 4: drop columns we don't need ──────────────────────

def drop_unused(df):
    """
    customerID is an identifier — not a feature.
    TotalCharges is now captured in ChargePerTenure.
    tenure_bucket was a temp EDA column.
    """
    drop_cols = [CUSTOMER_ID, "TotalCharges"]
    # Drop EDA temp columns if they exist
    for col in ["tenure_bucket"]:
        if col in df.columns:
            drop_cols.append(col)

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"  Dropped: {drop_cols}")
    return df


# ── Step 5: split features and target ───────────────────────

def get_feature_matrix(df):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    print(f"  Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  Features: {list(X.columns)}")
    return X, y


# ── Step 6: train/test split ─────────────────────────────────

def split_data(X, y):
    """
    stratify=y ensures both splits have the same 73/27 ratio.
    Without this, your test set might accidentally have
    40% churners and your AUC looks artificially great.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y            # CRITICAL — don't remove this
    )
    print(f"  Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")
    print(f"  Train churn rate: {y_train.mean():.1%}")
    print(f"  Test  churn rate: {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test


# ── Step 7: SMOTE ────────────────────────────────────────────

def apply_smote(X_train, y_train):
    """
    Insight 1 from EDA: 73/27 split means the model sees
    2.7x more 'Stay' examples than 'Churn' during training.
    SMOTE creates synthetic minority (Churn) samples.

    sampling_strategy=0.6 means after SMOTE:
      Churn count = 60% of Stay count  (not 50/50 — intentional)
      Reason: real world IS imbalanced, we just reduce the gap.
    """
    before_0 = (y_train == 0).sum()
    before_1 = (y_train == 1).sum()

    sm = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    after_0 = (y_res == 0).sum()
    after_1 = (y_res == 1).sum()

    print(f"  Before SMOTE → Stay: {before_0}  Churn: {before_1}")
    print(f"  After  SMOTE → Stay: {after_0}  Churn: {after_1}")
    print(f"  Rows added by SMOTE: {len(X_res) - len(X_train)}")
    return X_res, y_res


# ── Step 8: save processed data ─────────────────────────────

def save_processed(X_train, X_test, y_train, y_test):
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(PROC_DIR / "X_train.parquet", index=False)
    X_test.to_parquet( PROC_DIR / "X_test.parquet",  index=False)
    y_train.to_frame().to_parquet(PROC_DIR / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet( PROC_DIR / "y_test.parquet",  index=False)

    print(f"  Saved 4 parquet files to {PROC_DIR}")


# ── Master pipeline ──────────────────────────────────────────

def run_pipeline(df=None):
    print("\n=== Feature Engineering Pipeline ===\n")

    if df is None:
        df = pd.read_parquet(PROC_DIR / "interim.parquet")
        print(f"Loaded interim data: {df.shape}\n")

    print("Step 1 — Binary encoding")
    df = encode_binary(df)

    print("\nStep 2 — Categorical encoding")
    df = encode_categoricals(df)

    print("\nStep 3 — Feature engineering")
    df = engineer_features(df)

    print("\nStep 4 — Drop unused columns")
    df = drop_unused(df)

    print("\nStep 5 — Feature matrix")
    X, y = get_feature_matrix(df)

    print("\nStep 6 — Train/test split")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nStep 7 — SMOTE")
    X_train, y_train = apply_smote(X_train, y_train)

    print("\nStep 8 — Save processed data")
    save_processed(X_train, X_test, y_train, y_test)

    print("\n=== Pipeline complete ===")
    print(f"  Final training set : {X_train.shape}")
    print(f"  Final test set     : {X_test.shape}")
    print(f"  Features           : {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_pipeline()