from unittest import mock

import pytest

from src.main.collect.metrics.metrics_collector import MetricsCollector


@pytest.fixture
def collector():
    return MetricsCollector(language="kotlin")


def test_read_kt_reads_file(tmp_path):
    f = tmp_path / "foo.kt"
    f.write_text("hello")
    assert MetricsCollector.read_kt(f) == "hello"


def test_read_kt_reads_dir(tmp_path):
    (tmp_path / "a.kt").write_text("hi")
    (tmp_path / "b.kt").write_text("there")
    assert "hi" in MetricsCollector.read_kt(tmp_path)
    assert "there" in MetricsCollector.read_kt(tmp_path)


def test_read_kt_flat_reads_flat_dir(tmp_path):
    (tmp_path / "c.kt").write_text("code1")
    (tmp_path / "d.kt").write_text("code2")
    assert "code1" in MetricsCollector.read_kt_flat(tmp_path)
    assert "code2" in MetricsCollector.read_kt_flat(tmp_path)


def test_structural_calls_registry(monkeypatch):
    tree_obj = object()
    called = {}
    # Patch parse and registry
    monkeypatch.setattr("main.utils.kotlin_parser.parse", lambda src: tree_obj)
    monkeypatch.setattr("main.metrics.registry", {"foo": lambda tree: 1.23, "bar": lambda tree: 4.56})
    result = MetricsCollector.structural("some code")
    assert result == {'Abrupt Control Flow': 0,
                      'Chapin Q': 0.0,
                      'Conditional Complexity': 0.0,
                      'Conditional Statements': 0,
                      'Cyclomatic Complexity': 1,
                      'Halstead Bugs': 0,
                      'Halstead Difficulty': 0,
                      'Halstead Distinct Operands': 0,
                      'Halstead Distinct Operators': 0,
                      'Halstead Effort': 0,
                      'Halstead Length': 0,
                      'Halstead Time': 0.0,
                      'Halstead Total Operands': 0,
                      'Halstead Total Operators': 0,
                      'Halstead Vocabulary': 0,
                      'Halstead Volume': 0,
                      'Labeled Blocks': 0,
                      'Local Variables': 0,
                      'Pivovarsky N(G)': 1,
                      'Program Size': 4}


def test_entropy_metrics_calls_entropy_methods(collector, monkeypatch):
    mock_entropy = mock.Mock()
    mock_entropy.cross_entropy.return_value = 1
    mock_entropy.kl_div.return_value = 2
    mock_entropy.perplexity.return_value = 3
    mock_entropy.jensen_shannon_distance.return_value = 4
    mock_entropy.conditional_entropy.return_value = 5
    collector.entropy = mock_entropy
    res = collector.entropy_metrics("a", "b")
    assert res == {"CE": 1, "KL": 2, "PPL": 3, "JSD": 4, "CondE": 5}


def test_load_lm_reads_jsons(tmp_path, collector):
    (tmp_path / "unigram.json").write_text('{"a":1}', encoding="utf-8")
    (tmp_path / "bigram.json").write_text('{"b":2}', encoding="utf-8")
    (tmp_path / "left.json").write_text('{"c":3}', encoding="utf-8")
    p_uni, p_bi, p_left = collector.load_lm(tmp_path)
    assert p_uni == {"a": 1}
    assert p_bi == {"b": 2}
    assert p_left == {"c": 3}


def test_lm_metrics_calls_entropy_methods(collector):
    collector.entropy.cross_entropy_lang = mock.Mock(return_value=1)
    collector.entropy.kl_div_lang = mock.Mock(return_value=2)
    collector.entropy.perplexity_lang = mock.Mock(return_value=3)
    collector.entropy.jensen_shannon_distance_lang = mock.Mock(return_value=4)
    collector.entropy.conditional_entropy_lang = mock.Mock(return_value=5)
    res = collector.lm_metrics({"u": 1}, {"b": 2}, {"l": 3}, "code")
    assert res == {
        "LM_CE": 1,
        "LM_KL": 2,
        "LM_PPL": 3,
        "LM_JSD": 4,
        "LM_CondE": 5
    }


def test_collect_tests_finds_and_reads(tmp_path, monkeypatch, collector):
    test1 = tmp_path / "test1"
    test1.mkdir()
    (test1 / "Bytecode").mkdir()
    (test1 / "Bytecode" / "FileChatGPT.kt").write_text("converted")
    (test1 / "Original.kt").write_text("original code")
    monkeypatch.setattr(collector, "read_kt_flat", lambda p: (p / "Original.kt").read_text())
    monkeypatch.setattr(collector, "read_kt", lambda p: p.read_text())
    out = collector.collect_tests(tmp_path)
    assert "test1" in out
    assert out["test1"]["orig"] == "original code"
    assert "BytecodeChatGPT" in out["test1"]["decs"]
    assert out["test1"]["decs"]["BytecodeChatGPT"] == "converted"


def test_build_pairs_works():
    # test that build_pairs returns the correct structure
    tests = {
        "T": {
            "orig": "SRC",
            "decs": {"Cat1": "DEC1", "Cat2": "DEC2"}
        }
    }
    pairs = MetricsCollector.build_pairs(tests)
    assert ("T", "Original", "SRC", "SRC") in pairs
    assert ("T", "Cat1", "DEC1", "SRC") in pairs
    assert ("T", "Cat2", "DEC2", "SRC") in pairs
    # test that error is raised on empty input
    with pytest.raises(ValueError):
        MetricsCollector.build_pairs({})
