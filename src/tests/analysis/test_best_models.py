

from unittest import mock

import pandas as pd
import pytest
from global_config import FEATURES

from src.main.analysis.best_models import MetricProcessor


@pytest.fixture
def sample_metrics_df() -> pd.DataFrame:
    """
    Provides a sample metrics DataFrame for testing.
    """
    data = {
        'model': ['kt_source', 'model1'],
    }
    for i, feat in enumerate(FEATURES):
        data[feat] = [float(i), float(i+1)]
    return pd.DataFrame(data)


@pytest.fixture
def sample_comp_df() -> pd.DataFrame:
    """
    Provides a sample compilation DataFrame for testing.
    """
    data = {
        'field': ['model1'],
        'repo_count': [10]
    }
    return pd.DataFrame(data)


def test_validate_columns_pass(sample_metrics_df: pd.DataFrame) -> None:
    """
    Tests that validate_columns passes when required columns exist.
    """
    processor = MetricProcessor("metrics.csv", "comp.csv", "output.csv")
    processor.validate_columns(sample_metrics_df)


def test_validate_columns_fail() -> None:
    """
    Tests that validate_columns raises an error if required columns are missing.
    """
    processor = MetricProcessor("metrics.csv", "comp.csv", "output.csv")
    df_missing = pd.DataFrame({"model": ["kt_source"]})  # Missing features

    with pytest.raises(ValueError) as exc:
        processor.validate_columns(df_missing)

    assert "Missing columns" in str(exc.value)


def test_compute_reference_vector(sample_metrics_df: pd.DataFrame) -> None:
    """
    Tests computation of the reference vector and zeroing certain features.
    """
    processor = MetricProcessor("metrics.csv", "comp.csv", "output.csv")
    ref_vec: pd.Series = processor.compute_reference_vector(sample_metrics_df)

    assert isinstance(ref_vec, pd.Series)
    assert ref_vec['CondE'] == 0.0
    assert ref_vec['JSD'] == 0.0
    assert ref_vec['KL'] == 0.0


def test_compute_distances(sample_metrics_df: pd.DataFrame) -> None:
    """
    Tests computation of distances to the reference vector.
    """
    processor = MetricProcessor("metrics.csv", "comp.csv", "output.csv")
    ref_vec: pd.Series = processor.compute_reference_vector(sample_metrics_df)
    row = sample_metrics_df.iloc[1]
    coverage = 1.0

    distances: pd.Series = processor.compute_distances(row, ref_vec, coverage)

    assert isinstance(distances, pd.Series)
    for metric in ['euclidean', 'manhattan', 'cosine', 'chebyshev']:
        assert metric in distances
        assert metric + "_cov" in distances


def test_process_pipeline(monkeypatch: pytest.MonkeyPatch, sample_metrics_df: pd.DataFrame, sample_comp_df: pd.DataFrame) -> None:
    """
    Tests the main processing pipeline end-to-end, without actual file I/O.
    """
    processor = MetricProcessor("metrics.csv", "comp.csv", "output.csv")

    monkeypatch.setattr(processor, "load_data", lambda: (sample_metrics_df, sample_comp_df))
    monkeypatch.setattr(processor, "validate_columns", lambda df: None)
    monkeypatch.setattr(
        processor,
        "compute_reference_vector",
        lambda df: df.loc[df["model"] == "kt_source", FEATURES].iloc[0]
    )

    # Patch to_csv to capture the output DataFrame instead of writing to file
    with mock.patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
        processor.process()
        assert mock_to_csv.called


def test_load_data_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Tests load_data method for missing files, ensuring sys.exit is called.
    """
    processor = MetricProcessor("missing.csv", "missing.csv", "output.csv")

    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit):
        processor.load_data()
