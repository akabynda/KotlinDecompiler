import pytest
from pathlib import Path
from unittest import mock
import pandas as pd

from src.main.collect.metrics.metrics_for_synthetic_tests import (
    SyntheticMetricsCalculator,
)


@pytest.fixture
def dummy_collector(monkeypatch):
    # Patch MetricsCollector to return controlled, deterministic test data
    mock_metrics = mock.Mock()
    # Simulate collect_tests and build_pairs
    mock_metrics.collect_tests.return_value = {
        "TestA": {"orig": "orig_code", "decs": {"BytecodeChatGPT": "dec_code"}}
    }
    mock_metrics.build_pairs.return_value = [
        ("TestA", "BytecodeChatGPT", "dec_code", "orig_code")
    ]
    # Return fixed metric values
    mock_metrics.structural.return_value = {"strct1": 1.0}
    mock_metrics.entropy_metrics.return_value = {"KL": 0.5, "JSD": 0.9}
    mock_metrics.lm_metrics.return_value = {"LM_CE": 0.8}
    mock_metrics.load_lm.return_value = ({"u": 1}, {"b": 2}, {"l": 3})

    # Patch where SyntheticMetricsCalculator looks up MetricsCollector
    monkeypatch.setattr(
        "src.main.collect.metrics.metrics_for_synthetic_tests.MetricsCollector",
        mock.Mock(return_value=mock_metrics),
    )
    return mock_metrics


def test_init_loads_lm(monkeypatch):
    # Patch MetricsCollector at correct point
    mock_metrics = mock.Mock()
    mock_metrics.load_lm.return_value = ({"u": 1}, {"b": 2}, {"l": 3})
    monkeypatch.setattr(
        "src.main.collect.metrics.metrics_for_synthetic_tests.MetricsCollector",
        mock.Mock(return_value=mock_metrics),
    )
    calc = SyntheticMetricsCalculator(Path("tests"), Path("out.csv"))
    assert calc.p_uni == {"u": 1}
    assert calc.p_bi == {"b": 2}
    assert calc.p_left == {"l": 3}


def test_build_rows_calls_all_metrics(monkeypatch, dummy_collector, tmp_path):
    # Use the dummy_collector with all deterministic returns
    calc = SyntheticMetricsCalculator(tmp_path, tmp_path / "metrics.csv")
    calc.metrics_collector = dummy_collector
    calc.p_uni, calc.p_bi, calc.p_left = {}, {}, {}

    rows = calc.build_rows()
    # Should be exactly one row with expected values
    assert len(rows) == 1
    row = rows[0]
    assert row["Test"] == "TestA"
    assert row["Category"] == "BytecodeChatGPT"
    assert row["strct1"] == 1.0
    assert row["KL"] == 0.5
    assert row["JSD"] == 0.9
    assert row["LM_CE"] == 0.8


def test_run_saves_csv(monkeypatch, dummy_collector, tmp_path):
    calc = SyntheticMetricsCalculator(tmp_path, tmp_path / "out.csv")
    calc.metrics_collector = dummy_collector
    calc.p_uni, calc.p_bi, calc.p_left = {}, {}, {}

    calc.run()
    # Check the CSV was written and contains expected columns and row
    df = pd.read_csv(tmp_path / "out.csv")
    # One row, columns as above
    assert list(df.columns) == ["Test", "Category", "strct1", "KL", "JSD", "LM_CE"]
    assert df.iloc[0]["Test"] == "TestA"
    assert df.iloc[0]["Category"] == "BytecodeChatGPT"
    assert df.iloc[0]["strct1"] == 1.0
    assert df.iloc[0]["KL"] == 0.5
    assert df.iloc[0]["JSD"] == 0.9
    assert df.iloc[0]["LM_CE"] == 0.8


def test_build_rows_empty_pairs(monkeypatch, dummy_collector, tmp_path):
    # Simulate no pairs
    dummy_collector.build_pairs.return_value = []
    calc = SyntheticMetricsCalculator(tmp_path, tmp_path / "out.csv")
    calc.metrics_collector = dummy_collector
    calc.p_uni, calc.p_bi, calc.p_left = {}, {}, {}
    rows = calc.build_rows()
    assert rows == []


def test_build_rows_multiple_pairs(monkeypatch, dummy_collector, tmp_path):
    # Simulate multiple pairs
    dummy_collector.build_pairs.return_value = [
        ("TestA", "BytecodeChatGPT", "dec_code", "orig_code"),
        ("TestB", "BytecodeJ2K", "dec2", "orig2"),
    ]
    dummy_collector.structural.side_effect = [{"strct1": 1.0}, {"strct1": 2.0}]
    dummy_collector.entropy_metrics.side_effect = [
        {"KL": 0.5, "JSD": 0.9},
        {"KL": 0.8, "JSD": 1.1},
    ]
    dummy_collector.lm_metrics.side_effect = [
        {"LM_CE": 0.8},
        {"LM_CE": 1.2},
    ]
    calc = SyntheticMetricsCalculator(tmp_path, tmp_path / "out.csv")
    calc.metrics_collector = dummy_collector
    calc.p_uni, calc.p_bi, calc.p_left = {}, {}, {}
    rows = calc.build_rows()
    assert len(rows) == 2
    assert rows[0]["Test"] == "TestA"
    assert rows[1]["Test"] == "TestB"
    assert rows[1]["KL"] == 0.8
    assert rows[1]["LM_CE"] == 1.2
