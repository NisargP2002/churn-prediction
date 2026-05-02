import sys
import joblib
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    MODEL_DIR, PROC_DIR, N_ESTIMATORS,
    MAX_DEPTH, LEARNING_RATE,
    SCALE_POS_WEIGHT, RANDOM_STATE
)


def load_data():
    X_train = pd.read_parquet(PROC_DIR / "X_train.parquet")
    y_train = pd.read_parquet(PROC_DIR / "y_train.parquet").squeeze()
    X_test  = pd.read_parquet(PROC_DIR / "X_test.parquet")
    y_test  = pd.read_parquet(PROC_DIR / "y_test.parquet").squeeze()
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train(X_train, y_train, X_test, y_test, run_name="xgb_baseline"):
    MODEL_DIR.mkdir(exist_ok=True)

    mlflow.set_experiment("churn_prediction")

    with mlflow.start_run(run_name=run_name) as run:

        params = {
            "n_estimators":      N_ESTIMATORS,
            "max_depth":         MAX_DEPTH,
            "learning_rate":     LEARNING_RATE,
            "scale_pos_weight":  SCALE_POS_WEIGHT,
            "random_state":      RANDOM_STATE,
            "eval_metric":       "auc",
            "use_label_encoder": False,
            "tree_method":       "hist",
        }
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50
        )

        # Save model locally
        model_path = MODEL_DIR / "churn_model.pkl"
        joblib.dump(model, model_path)
        print(f"\nModel saved → {model_path}")

        # Log model artifact to MLflow
        mlflow.xgboost.log_model(model, "model")

        print(f"MLflow run ID: {run.info.run_id}")
        return model


if __name__ == "__main__":
    print("\n=== Training XGBoost Churn Model ===\n")
    X_train, X_test, y_train, y_test = load_data()
    model = train(X_train, y_train, X_test, y_test)
    print("\nTraining complete. Run evaluate.py next.")