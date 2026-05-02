import sys
import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix
)

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_DIR, PROC_DIR, OUTPUT_DIR


def load_artifacts():
    model   = joblib.load(MODEL_DIR / "churn_model.pkl")
    X_test  = pd.read_parquet(PROC_DIR / "X_test.parquet")
    y_test  = pd.read_parquet(PROC_DIR / "y_test.parquet").squeeze()
    return model, X_test, y_test


def compute_metrics(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)

    print("\n=== Model Evaluation ===\n")
    print(classification_report(
        y_test, y_pred,
        target_names=["Stay", "Churn"]
    ))
    print(f"AUC-ROC            : {auc:.4f}")
    print(f"Average Precision  : {ap:.4f}")
    print(f"Baseline (no model): {1 - y_test.mean():.4f}")
    print(f"Your lift           : +{auc - (1 - y_test.mean()):.4f}")

    return y_pred, y_proba, auc, ap


def plot_results(y_test, y_pred, y_proba, auc, ap):
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    # ── Plot 1: ROC curve ─────────────────────────
    ax1  = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax1.plot(fpr, tpr, color="#534AB7", lw=2,
             label=f"AUC = {auc:.3f}")
    ax1.plot([0,1],[0,1], color="#B4B2A9",
             linestyle="--", label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Plot 2: Precision-Recall curve ───────────
    ax2 = fig.add_subplot(gs[0, 1])
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ax2.plot(recall, precision, color="#1D9E75", lw=2,
             label=f"AP = {ap:.3f}")
    ax2.axhline(y=y_test.mean(), color="#D85A30",
                linestyle="--", label=f"Baseline = {y_test.mean():.2f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Plot 3: Confusion matrix ─────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    cm  = confusion_matrix(y_test, y_pred)
    im  = ax3.imshow(cm, cmap="Blues")
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Stay", "Churn"])
    ax3.set_yticklabels(["Stay", "Churn"])
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]),
                     ha="center", va="center", fontsize=14,
                     color="white" if cm[i,j] > cm.max()/2 else "black")

    # ── Plot 4: Score distribution ───────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(y_proba[y_test == 0], bins=40, alpha=0.6,
             color="#1D9E75", label="Stay", density=True)
    ax4.hist(y_proba[y_test == 1], bins=40, alpha=0.6,
             color="#D85A30", label="Churn", density=True)
    ax4.axvline(x=0.5, color="#444441",
                linestyle="--", label="Threshold=0.5")
    ax4.set_xlabel("Predicted churn probability")
    ax4.set_ylabel("Density")
    ax4.set_title("Score Distribution")
    ax4.legend()
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # ── Plot 5: Feature importance ───────────────
    ax5 = fig.add_subplot(gs[1, 1:])
    model = joblib.load(MODEL_DIR / "churn_model.pkl")
    feat_names = pd.read_parquet(
        PROC_DIR / "X_test.parquet").columns.tolist()
    importances = model.feature_importances_
    idx = np.argsort(importances)
    ax5.barh([feat_names[i] for i in idx],
             importances[idx], color="#534AB7", alpha=0.8)
    ax5.set_xlabel("Importance score")
    ax5.set_title("Feature Importance (XGBoost)")
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    plt.suptitle("Churn Model Evaluation", fontsize=14,
                 fontweight="500", y=1.01)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "evaluation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()


def log_to_mlflow(auc, ap):
    mlflow.set_experiment("churn_prediction")
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("avg_precision", ap)
        mlflow.log_artifact(str(OUTPUT_DIR / "evaluation.png"))
    print("Metrics logged to MLflow.")


if __name__ == "__main__":
    model, X_test, y_test = load_artifacts()
    y_pred, y_proba, auc, ap = compute_metrics(model, X_test, y_test)
    plot_results(y_test, y_pred, y_proba, auc, ap)
    log_to_mlflow(auc, ap)