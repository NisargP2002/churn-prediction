from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure": 3,
                "MonthlyCharges": 89.5,
                "TotalCharges": 268.5,
                "Contract": 0,
                "InternetService": 2,
                "PaymentMethod": 0,
                "NumServices": 2,
                "NumProtectiveServices": 0,
            }
        }
    )

    tenure:          int   = Field(..., ge=0, le=72)
    MonthlyCharges:  float = Field(..., ge=0, le=200)
    TotalCharges:    float = Field(..., ge=0)
    Contract:        int   = Field(..., ge=0, le=2)
    InternetService: int   = Field(..., ge=0, le=2)
    PaymentMethod:   int   = Field(..., ge=0, le=3)
    NumServices:          int = Field(..., ge=0, le=8)
    NumProtectiveServices: int = Field(..., ge=0, le=4)
    IsNewCustomer:   int   = Field(default=0, ge=0, le=1)
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

    @model_validator(mode="after")
    def derive_training_features(self):
        self.IsNewCustomer = int(self.tenure < 12)
        self.ChargePerTenure = (
            round(self.TotalCharges / self.tenure, 4)
            if self.tenure > 0
            else self.MonthlyCharges
        )

        if self.tenure <= 6:
            self.TenureBucket = 4
        elif self.tenure <= 12:
            self.TenureBucket = 3
        elif self.tenure <= 24:
            self.TenureBucket = 2
        elif self.tenure <= 48:
            self.TenureBucket = 1
        else:
            self.TenureBucket = 0

        self.HighRiskFlag = int(
            self.Contract == 0
            and self.IsNewCustomer == 1
            and self.InternetService == 2
        )
        return self


class PredictionResponse(BaseModel):
    customer_id:        str
    churn_probability:  float
    churn_prediction:   bool
    risk_level:         Literal["low", "medium", "high"]
    top_churn_reasons:  list[str]
