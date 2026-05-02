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

    @validator("ChargePerTenure", always=True, pre=True)
    def compute_charge_per_tenure(cls, v, values):
        if v == 0.0 and "tenure" in values:
            t  = values["tenure"]
            tc = values.get("TotalCharges", 0)
            return round(tc / t, 4) if t > 0 else 0.0
        return v

    @validator("HighRiskFlag", always=True, pre=True)
    def compute_high_risk(cls, v, values):
        if v == 0:
            contract = values.get("Contract", 1)
            new_cust = values.get("IsNewCustomer", 0)
            internet = values.get("InternetService", 0)
            return int(contract == 0 and new_cust == 1
                       and internet == 2)
        return v

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