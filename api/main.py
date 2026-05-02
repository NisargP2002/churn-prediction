import sys
import shap
import joblib
import numpy as np
import pandas as pd
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_DIR, PROC_DIR, API_HOST, API_PORT
from api.schema import CustomerFeatures, PredictionResponse

model     = None
explainer = None
FEATURE_ORDER = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, explainer, FEATURE_ORDER

    print("Loading model...")
    model = joblib.load(MODEL_DIR / "churn_model.pkl")

    print("Building SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    # Lock feature order from training data
    X_ref = pd.read_parquet(PROC_DIR / "X_test.parquet")
    FEATURE_ORDER = X_ref.columns.tolist()
    print(f"Model ready. Features: {len(FEATURE_ORDER)}")
    yield
    model = None


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts telco customer churn with SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    customer: CustomerFeatures,
    customer_id: str = Query(default="unknown")
):
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded yet")
    try:
        # Build feature row in exact training order
        row = pd.DataFrame([customer.dict()])[FEATURE_ORDER]

        # Prediction
        prob = float(model.predict_proba(row)[0][1])

        # SHAP top reasons
        sv      = explainer.shap_values(row)[0]
        top_idx = np.argsort(np.abs(sv))[-3:][::-1]
        reasons = []
        for i in top_idx:
            direction = "increases" if sv[i] > 0 else "decreases"
            reasons.append(
                f"{FEATURE_ORDER[i]} {direction} churn risk"
                f" (impact: {sv[i]:+.3f})"
            )

        # Risk tier
        if prob > 0.6:
            risk = "high"
        elif prob > 0.3:
            risk = "medium"
        else:
            risk = "low"

        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=round(prob, 4),
            churn_prediction=prob > 0.5,
            risk_level=risk,
            top_churn_reasons=reasons
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
def model_info():
    if FEATURE_ORDER is None:
        raise HTTPException(status_code=503, detail="Not ready")
    return {
        "n_features": len(FEATURE_ORDER),
        "features":   FEATURE_ORDER,
        "model_type": "XGBoostClassifier",
        "version":    "1.0.0"
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app",
                host=API_HOST,
                port=API_PORT,
                reload=True)