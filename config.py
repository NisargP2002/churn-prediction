from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

TARGET_COL   = "Churn"
CUSTOMER_ID  = "customerID"
TEST_SIZE    = 0.2
RANDOM_STATE = 42

N_ESTIMATORS      = 300
MAX_DEPTH         = 6
LEARNING_RATE     = 0.05
SCALE_POS_WEIGHT  = 3
SMOTE_RATIO       = 0.6

SHAP_SAMPLE_SIZE  = 500

API_HOST = "0.0.0.0"
API_PORT = 8000