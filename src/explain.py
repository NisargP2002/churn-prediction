import sys
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_DIR, PROC_DIR, OUTPUT_DIR, SHAP_SAMPLE_SIZE


def load_artifacts():
    model  = joblib.load(MODEL_DIR / "churn_model.pkl")
    X_test = pd.read_parquet(PROC_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROC_DIR / "y_test.parquet").squeeze()
    print(f"Loaded model + test set: {X_test.shape}")
    return model, X_test, y_test


def build_explainer(model):
    """
    TreeExplainer is purpose-built for XGBoost.
    It runs in O(n) time vs KernelExplainer's O(n^2).
    For 1409 test rows this is the only sensible choice.
    """
    explainer = shap.TreeExplainer(model)
    print("TreeExplainer built.")
    return explainer


def global_beeswarm(explainer, X_test):
    """
    Beeswarm plot — the most informative SHAP plot.
    Each dot is one customer.
    X position = how much that feature pushed the prediction.
    Color       = feature value (red=high, blue=low).
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    X_sample = X_test.sample(
        min(SHAP_SAMPLE_SIZE, len(X_test)),
        random_state=42
    )
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sample,
        plot_type="dot",
        show=False
    )
    plt.title("SHAP beeswarm — what drives churn predictions",
              fontsize=12, pad=12)
    plt.tight_layout()
    out = OUTPUT_DIR / "shap_beeswarm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Beeswarm saved → {out}")
    plt.show()

    return shap_values, X_sample


def global_bar(explainer, X_test):
    """
    Bar plot — mean absolute SHAP value per feature.
    Simpler than beeswarm but cleaner for README screenshots.
    """
    X_sample = X_test.sample(
        min(SHAP_SAMPLE_SIZE, len(X_test)),
        random_state=42
    )
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X_sample,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP feature importance (mean |SHAP value|)",
              fontsize=12, pad=12)
    plt.tight_layout()
    out = OUTPUT_DIR / "shap_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Bar chart saved → {out}")
    plt.show()


def waterfall_single(explainer, X_test, y_test, customer_idx=None):
    """
    Waterfall plot for ONE customer.
    Shows exactly which features pushed the prediction
    up or down from the baseline expected value.
    This is your interview demo plot.
    """
    # Default: find a high-risk customer (actual churn=1)
    if customer_idx is None:
        churn_indices = y_test[y_test == 1].index
        customer_idx = churn_indices[0]

    customer = X_test.loc[[customer_idx]]
    shap_val = explainer.shap_values(customer)

    print(f"\n--- Customer {customer_idx} ---")
    print(customer.T.rename(columns={customer_idx: "value"}).to_string())

    prob = 1 / (1 + np.exp(-explainer.expected_value -
                            shap_val[0].sum()))
    print(f"\nPredicted churn probability: {prob:.1%}")

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_val[0],
            base_values=explainer.expected_value,
            data=customer.values[0],
            feature_names=customer.columns.tolist()
        ),
        show=False
    )
    plt.title(f"Why is customer {customer_idx} predicted to churn?",
              fontsize=11, pad=12)
    plt.tight_layout()
    out = OUTPUT_DIR / f"shap_waterfall_{customer_idx}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Waterfall saved → {out}")
    plt.show()

    return customer, shap_val


def what_if(model, explainer, customer_row):
    """
    The interview wow factor.
    Take a high-risk customer and simulate
    what happens if you change their contract.
    """
    original_prob = model.predict_proba(customer_row)[0][1]

    print("\n=== What-If Simulator ===")
    print(f"Original churn probability : {original_prob:.1%}")
    print(f"Contract value now         : {customer_row['Contract'].values[0]}"
          " (0=Month-to-month)")

    scenarios = [
        {"Contract": 1, "label": "Upgrade to One year contract"},
        {"Contract": 2, "label": "Upgrade to Two year contract"},
        {"Contract": 1, "NumProtectiveServices": min(
            customer_row["NumProtectiveServices"].values[0] + 2, 4),
         "label": "One year + 2 protective services"},
    ]

    for s in scenarios:
        label = s.pop("label")
        modified = customer_row.copy()
        for col, val in s.items():
            modified[col] = val
        new_prob = model.predict_proba(modified)[0][1]
        reduction = original_prob - new_prob
        print(f"\n  Scenario : {label}")
        print(f"  New prob : {new_prob:.1%}  "
              f"(risk reduced by {reduction:.1%})")


if __name__ == "__main__":
    print("\n=== SHAP Explainability ===\n")

    model, X_test, y_test = load_artifacts()
    explainer = build_explainer(model)

    print("\n[1/4] Global beeswarm plot...")
    shap_values, X_sample = global_beeswarm(explainer, X_test)

    print("\n[2/4] Global bar chart...")
    global_bar(explainer, X_test)

    print("\n[3/4] Single customer waterfall...")
    customer, shap_val = waterfall_single(
        explainer, X_test, y_test)

    print("\n[4/4] What-if simulator...")
    what_if(model, explainer, customer)

    print("\n=== SHAP complete ===")
    print(f"All plots saved to outputs/")