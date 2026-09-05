from pathlib import Path
import numpy as np
from src.data_pipeline import load_data, impute_categorical, neutralize_outliers, engineer_features, validate_business_rules

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Dataset_for_Data_Analytics.xlsx"

def test_business_rules():
    df = load_data(RAW)
    validate_business_rules(df)

def test_coupon_imputation_removes_missingness():
    df = impute_categorical(load_data(RAW))
    assert df["CouponCode_Imputed"].isna().sum() == 0

def test_outlier_winsorization_preserves_rows():
    df = load_data(RAW)
    out = neutralize_outliers(df)
    assert len(out) == len(df)
    assert out["TotalPrice_Winsorized"].notna().all()

def test_engineered_features_exist():
    df = engineer_features(neutralize_outliers(impute_categorical(load_data(RAW))))
    expected = {"HasCoupon", "OrderYear", "OrderMonth", "OrderQuarter", "OrderDayOfWeek", "IsWeekend", "BasketFillRatio", "OrderValuePerCartItem"}
    assert expected.issubset(df.columns)
