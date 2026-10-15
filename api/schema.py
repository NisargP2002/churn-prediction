from pydantic import BaseModel, Field, validator
from typing import Literal


class CustomerFeatures(BaseModel):
    tenure:          int   = Field(..., ge=0, le=72)
    MonthlyCharges:  float = Field(..., ge=0, le=200)
    TotalCharges:    float = Field(..., ge=0)
    Contract:        int   = Field(..., ge=0, le=2)
    InternetService: int   = Field(..., ge=0, le=2)
    PaymentMethod:   int   = Field(..., ge=0, le=3)
    NumServices:          int = Field(..., ge=0, le=8)
    NumProtectiveServices: int = Field(..., ge=0, le=4)
    IsNewCustomer:   int   = Field(..., ge=0, le=1)
    ChargePerTenure: float = Field(default=0.0)

    # All remaining features the model expects
    gender:          int = Field(default=0)
    Partner:         int = Field(default=0)
    Dependents:      int = Field(default=0)
    PhoneService:    int = Field(default=1)
    MultipleLines:   int = Field(default=0)
    OnlineSecurity:  int = Field(default=0)
    OnlineBackup:    int = Field(default=0)
    DeviceProtection: int = Field(default=0)
    TechSupport:     int = Field(default=0)
    StreamingTV:     int = Field(default=0)
    StreamingMovies: int = Field(default=0)
    PaperlessBilling: int = Field(default=1)
    SeniorCitizen:   int = Field(default=0)
    HighRiskFlag:    int = Field(default=0)
    TenureBucket:    int = Field(default=0)

    @validator("IsNewCustomer", always=True, pre=True)
    def compute_is_new_customer(cls, v, values):
        return int(values.get("tenure", 0) < 12)

    @validator("ChargePerTenure", always=True, pre=True)
    def compute_charge_per_tenure(cls, v, values):
        tenure = values.get("tenure", 0)
        total_charges = values.get("TotalCharges", 0)
        monthly_charges = values.get("MonthlyCharges", 0)
        return round(total_charges / tenure, 4) if tenure > 0 else monthly_charges

    @validator("TenureBucket", always=True, pre=True)
    def compute_tenure_bucket(cls, v, values):
        tenure = values.get("tenure", 0)
        if tenure <= 6:
            return 4
        if tenure <= 12:
            return 3
        if tenure <= 24:
            return 2
        if tenure <= 48:
            return 1
        return 0

    @validator("HighRiskFlag", always=True, pre=True)
    def compute_high_risk(cls, v, values):
        contract = values.get("Contract", 1)
        new_cust = values.get("IsNewCustomer", 0)
        internet = values.get("InternetService", 0)
        return int(contract == 0 and new_cust == 1 and internet == 2)

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 3,
                "MonthlyCharges": 89.5,
                "TotalCharges": 268.5,
                "Contract": 0,
                "InternetService": 2,
                "PaymentMethod": 0,
                "NumServices": 2,
                "NumProtectiveServices": 0,
                "IsNewCustomer": 1
            }
        }


class PredictionResponse(BaseModel):
    customer_id:        str
    churn_probability:  float
    churn_prediction:   bool
    risk_level:         Literal["low", "medium", "high"]
    top_churn_reasons:  list[str]