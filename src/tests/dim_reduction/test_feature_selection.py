import pytest
import numpy as np
import pandas as pd

from src.main.dim_reduction.feature_selection import FeatureSelector


@pytest.fixture
def df_numeric():
    # Simple df: two id columns + 3 numerics + some constants
    return pd.DataFrame({
        "Test": [1, 2, 3],
        "Category": [0, 1, 0],
        "kt_path": [0, 0, 0],
        "model": [1, 1, 1],
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [2.0, 3.0, 4.0],
        "feature3": [1.0, 1.0, 1.0],  # low variance
        "feature4": [10.0, 10.0, 10.0],  # constant
        "feature5": [100, 200, 300],  # correlated with feature1/2
    })


def test_numeric_df_removes_non_metrics(df_numeric):
    removed = []
    numeric = FeatureSelector.numeric_df(df_numeric, removed)
    # Should have only numeric cols and none of the special columns
    assert all(numeric.dtypes.apply(lambda t: np.issubdtype(t, np.number)))
    # All 'removed' special columns gone
    for col in ["Test", "Category", "kt_path", "model"]:
        assert col not in numeric.columns
    assert set(removed) == {"Test", "Category", "kt_path", "model"}


def test_drop_low_variance_removes_constant_and_lowvar(df_numeric):
    removed = []
    numeric = FeatureSelector.numeric_df(df_numeric, [])
    result = FeatureSelector.drop_low_variance(numeric, removed, threshold=0.01)
    # Should remove feature3 and feature4 (zero variance)
    assert "feature3" in removed
    assert "feature4" in removed
    assert "feature3" not in result.columns
    assert "feature4" not in result.columns
    # Keeps non-constant features
    assert "feature1" in result.columns

def test_drop_high_corr_removes_highly_corr(df_numeric):
    removed = []
    numeric = FeatureSelector.numeric_df(df_numeric, [])
    # Remove constant/lowvar first (feature3, feature4)
    no_lowvar = FeatureSelector.drop_low_variance(numeric, removed)
    # Now feature1/feature2/feature5: feature5 is linearly correlated with feature1 and feature2
    result = FeatureSelector.drop_high_corr(no_lowvar.copy(), removed, thresh=0.99)
    # At least one of the correlated features is removed
    corr = result.corr()
    for i in corr.index:
        for j in corr.columns:
            if i != j:
                assert abs(corr.loc[i, j]) < 0.99

def test_drop_high_corr_recursive_prefers_max_removal():
    # Setup: featureA=1,2,3; featureB=2,3,4; featureC=100,200,300 (C highly correlated with A/B)
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [2, 3, 4],
        "C": [100, 200, 300],
        "D": [1, 1, 2],  # unrelated
    })
    upper = df.corr().abs().where(np.triu(np.ones(df.shape[1]), k=1).astype(bool))
    res = FeatureSelector.drop_high_corr_recursive(df, upper, thresh=0.99)
    # It must drop one of (A, B, C) to eliminate the high correlation
    assert res['max_dropped'] >= 1
    assert any(f in res['to_drop'] for f in "ABC")

def test_drop_high_corr_does_not_modify_removed(df_numeric):
    removed = []
    numeric = FeatureSelector.numeric_df(df_numeric, [])
    result = FeatureSelector.drop_high_corr(numeric.copy(), removed, thresh=0.99)
    # All removed features are actually missing from the result
    for feat in removed:
        assert feat not in result.columns

def test_safe_corr_df_stops_early_and_invertible():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4],
        "f2": [2, 3, 4, 5],  # correlated with f1
        "f3": [7, 8, 9, 10], # correlated with f1/f2
        "f4": [0, 1, 0, 1],  # unrelated
        "f5": [1, 1, 2, 2],  # unrelated
    })
    removed = []
    df_clean = FeatureSelector.safe_corr_df(df, removed)
    # Should stop when fewer than 3 features left OR when matrix is invertible
    # Check that correlation matrix is non-singular or at most two features left
    assert df_clean.shape[1] < 3 or np.linalg.matrix_rank(np.corrcoef(df_clean.T)) == df_clean.shape[1]

def test_numeric_df_handles_no_special_columns():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [2, 3, 4]
    })
    removed = []
    out = FeatureSelector.numeric_df(df, removed)
    assert set(out.columns) == {"a", "b"}
    assert removed == []

def test_drop_low_variance_keeps_all_if_above_thresh():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [1.1, 2.2, 3.3]
    })
    removed = []
    out = FeatureSelector.drop_low_variance(df, removed, threshold=0.00001)
    assert set(out.columns) == {"x", "y"}
    assert removed == []

def test_drop_high_corr_handles_no_corr():
    df = pd.DataFrame({
        "x": [1, 20, 3],
        "y": [10, 12, 55]
    })
    removed = []
    out = FeatureSelector.drop_high_corr(df.copy(), removed, thresh=0.5)
    assert set(out.columns) == {"x", "y"}
    assert removed == []
