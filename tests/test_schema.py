from api.schema import CustomerFeatures


def test_engineered_features_match_training_transformations():
    customer = CustomerFeatures(
        tenure=0,
        MonthlyCharges=75.0,
        TotalCharges=0.0,
        Contract=0,
        InternetService=2,
        PaymentMethod=0,
        NumServices=2,
        NumProtectiveServices=0,
        IsNewCustomer=0,
    )

    assert customer.IsNewCustomer == 1
    assert customer.TenureBucket == 4
    assert customer.ChargePerTenure == 75.0
    assert customer.HighRiskFlag == 1


def test_tenure_bucket_boundaries():
    base = {
        "MonthlyCharges": 50.0,
        "TotalCharges": 50.0,
        "Contract": 1,
        "InternetService": 1,
        "PaymentMethod": 1,
        "NumServices": 1,
        "NumProtectiveServices": 1,
        "IsNewCustomer": 0,
    }

    assert CustomerFeatures(tenure=6, **base).TenureBucket == 4
    assert CustomerFeatures(tenure=12, **base).TenureBucket == 3
    assert CustomerFeatures(tenure=24, **base).TenureBucket == 2
    assert CustomerFeatures(tenure=48, **base).TenureBucket == 1
    assert CustomerFeatures(tenure=72, **base).TenureBucket == 0
