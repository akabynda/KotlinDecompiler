import csv
import json
from unittest import mock

import pytest

from src.main.collect.metrics.metrics_for_models import ModelMetricsCalculator


@pytest.fixture
def dummy_metrics_for_models(monkeypatch):
    mock_coll = mock.Mock()
    mock_coll.structural.return_value = {"foo": 1.0}
    mock_coll.lm_metrics.return_value = {"bar": 2.0}
    mock_coll.entropy_metrics.return_value = {"baz": 3.0}
    monkeypatch.setattr("main.collect.metrics.metrics_for_models.MetricsCollector", mock.Mock(return_value=mock_coll))
    return mock_coll


def make_calc_with_dummy_collector(monkeypatch):
    mock_coll = mock.Mock()
    mock_coll.load_lm.return_value = ({"u": 1}, {"b": 2}, {"l": 3})
    mock_coll.structural.return_value = {"foo": 1.0}
    mock_coll.lm_metrics.return_value = {"bar": 2.0}
    mock_coll.entropy_metrics.return_value = {"baz": 3.0}
    monkeypatch.setattr("main.collect.metrics.metrics_for_models.MetricsCollector", mock.Mock(return_value=mock_coll))
    calc = ModelMetricsCalculator(workers=1)
    return calc, mock_coll


def test_init_sets_workers_and_loads_lm(monkeypatch):
    calc, mock_coll = make_calc_with_dummy_collector(monkeypatch)
    assert calc.workers == 1 or calc.workers == 2
    assert calc.p_uni == {"u": 1}
    assert calc.p_bi == {"b": 2}
    assert calc.p_left == {"l": 3}


def test_compute_row_returns_metric_row(monkeypatch):
    calc, mock_coll = make_calc_with_dummy_collector(monkeypatch)
    args = ("p1", "modelA", "CODE", "ORIG", ["foo", "bar", "baz"])
    out = calc.compute_row(args)
    assert out == ["p1", "modelA", 1.0, 2.0, 3.0]


def test_prepare_tasks_creates_tasks(tmp_path, monkeypatch):
    calc, mock_coll = make_calc_with_dummy_collector(monkeypatch)
    allowed = {"modelA": ["p1"]}
    (tmp_path / "allowed.json").write_text(json.dumps(allowed))
    records = [
        {"kt_path": "p1", "modelA": "CODE", "kt_source": "ORIG", "classes": None}
    ]
    with (tmp_path / "in.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    output_csv = tmp_path / "out.csv"
    tasks, metric_list, existing = calc.prepare_tasks(
        tmp_path / "in.jsonl", tmp_path / "allowed.json", output_csv
    )
    assert tasks
    t = tasks[0]
    assert t[:4] == ("p1", "modelA", "CODE", "ORIG")
    assert "foo" in metric_list or "bar" in metric_list or "baz" in metric_list
    assert existing == set()


def test_prepare_tasks_respects_existing_csv(tmp_path, monkeypatch):
    calc, mock_coll = make_calc_with_dummy_collector(monkeypatch)
    allowed = {"modelA": ["p1"]}
    (tmp_path / "allowed.json").write_text(json.dumps(allowed))
    records = [
        {"kt_path": "p1", "modelA": "CODE", "kt_source": "ORIG", "classes": None}
    ]
    with (tmp_path / "in.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    output_csv = tmp_path / "out.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kt_path", "model", "foo", "bar", "baz"])
        w.writerow(["p1", "modelA", 1, 2, 3])
    tasks, metric_list, existing = calc.prepare_tasks(
        tmp_path / "in.jsonl", tmp_path / "allowed.json", output_csv
    )
    assert not tasks
    assert ("p1", "modelA") in existing


def test_process_metrics_runs_and_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("main.collect.metrics.metrics_for_models.ProcessPoolExecutor",
                        lambda max_workers=None: DummyExecutor())
    monkeypatch.setattr("main.collect.metrics.metrics_for_models.as_completed", lambda futs: futs)

    mock_coll = mock.Mock()
    mock_coll.load_lm.return_value = ({"u": 1}, {"b": 2}, {"l": 3})
    mock_coll.structural.return_value = {"foo": 1.0}
    mock_coll.lm_metrics.return_value = {"bar": 2.0}
    mock_coll.entropy_metrics.return_value = {"baz": 3.0}
    monkeypatch.setattr("main.collect.metrics.metrics_for_models.MetricsCollector", mock.Mock(return_value=mock_coll))
    from src.main.collect.metrics.metrics_for_models import ModelMetricsCalculator

    calc = ModelMetricsCalculator(workers=1)
    allowed = {"modelA": ["p1"]}
    (tmp_path / "allowed.json").write_text(json.dumps(allowed))
    records = [
        {"kt_path": "p1", "modelA": "CODE", "kt_source": "ORIG", "classes": None}
    ]
    with (tmp_path / "in.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    output_csv = tmp_path / "out.csv"

    class DummyExecutor:
        def __enter__(self): return self

        def __exit__(self, *a): pass

        def submit(self, fn, arg):
            class DummyFuture:
                def result(self_): return fn(arg)

            return DummyFuture()

    calc.process_metrics(tmp_path / "in.jsonl", output_csv, tmp_path / "allowed.json")
    with output_csv.open("r", encoding="utf-8") as f:
        lines = list(csv.reader(f))
    assert lines[0][:2] == ["kt_path", "model"]
    assert lines[1][:2] == ["p1", "modelA"]
