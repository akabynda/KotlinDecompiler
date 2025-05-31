import json
import shutil
from unittest import mock

import pytest

from src.main.collect.build_language_model import LanguageModelBuilder


@pytest.fixture
def dummy_entropy(monkeypatch):
    # Patch Entropy to return simple tokenization
    mock_entropy = mock.Mock()
    mock_entropy.tokens.side_effect = lambda x: x.split() if isinstance(x, str) else []
    monkeypatch.setattr(
        "main.collect.build_language_model.Entropy",
        mock.Mock(return_value=mock_entropy),
    )
    return mock_entropy


@pytest.fixture
def dummy_dataset(monkeypatch):
    # Patch load_dataset to yield fake rows
    rows1 = [{"content": "a b c", "solution": "foo"} for _ in range(3)]
    rows2 = [{"content": "x y", "solution": "bar baz"} for _ in range(2)]

    # Each dataset should yield its own data, streaming yields an iterator
    def fake_load_dataset(name, split=None, streaming=None):
        if "KStack" in name:
            return iter(rows1)
        elif "KExercises" in name:
            return iter(rows2)
        else:
            return iter([])

    monkeypatch.setattr(
        "main.collect.build_language_model.load_dataset", fake_load_dataset
    )
    return (rows1, rows2)


@pytest.fixture
def builder(tmp_path, dummy_entropy, dummy_dataset):
    datasets = [
        ("KStack-clean", "content", lambda row: bool(row["content"])),
        ("KExercises", "solution", lambda row: bool(row["solution"])),
    ]
    return LanguageModelBuilder("kotlin", datasets, tmp_path)


def test_setup_dirs_creates_dirs(tmp_path, dummy_entropy, dummy_dataset):
    datasets = [
        ("KStack-clean", "content", lambda row: bool(row["content"])),
    ]
    builder = LanguageModelBuilder("kotlin", datasets, tmp_path)
    assert builder.output_dir.exists()
    assert builder.tmp_dir.exists()


def test_dump_and_merge_partial(tmp_path, dummy_entropy, dummy_dataset):
    datasets = [
        ("KStack-clean", "content", lambda row: bool(row["content"])),
    ]
    builder = LanguageModelBuilder("kotlin", datasets, tmp_path)
    builder.unigram_counter.update({"a": 2, "b": 1})
    builder.bigram_counter.update({("a", "b"): 1, ("b", "a"): 2})
    builder._dump_partial(0)
    assert any(f.name.startswith("partial-") for f in builder.tmp_dir.iterdir())

    # Simulate a second partial
    builder.unigram_counter.update({"c": 3})
    builder.bigram_counter.update({("c", "a"): 1})
    builder._dump_partial(1)

    final_uni, final_bi = builder._merge_partials()
    assert final_uni["a"] == 2
    assert final_uni["b"] == 1
    assert final_uni["c"] == 3
    assert final_bi[("a", "b")] == 1
    assert final_bi[("b", "a")] == 2
    assert final_bi[("c", "a")] == 1
    assert not builder.tmp_dir.exists()  # Should be cleaned up


def test_process_dataset_and_token_count(tmp_path, dummy_entropy, dummy_dataset):
    datasets = [
        ("KStack-clean", "content", lambda row: bool(row["content"])),
    ]
    builder = LanguageModelBuilder("kotlin", datasets, tmp_path)
    builder.TOKENS_PER_DUMP = 5
    builder._process_dataset("KStack-clean", "content", lambda row: True)
    partials = list(builder.tmp_dir.glob("partial-*.pkl"))
    assert len(partials) == 1
    assert sum(builder.unigram_counter.values()) == 3
    assert builder.total_unigrams == 9  # 3 rows * 3 tokens each


def test_save_json_and_nested_bigram(tmp_path, builder):
    data = {"a": 1, "b": 2}
    json_path = tmp_path / "out.json"
    builder._save_json(json_path, data, indent=2)
    with json_path.open() as f:
        assert json.load(f) == data

    bi_prob = {("a", "b"): 0.5, ("b", "c"): 0.25}
    builder._save_nested_bigram(bi_prob)
    with (builder.output_dir / "bigram.json").open() as f:
        out = json.load(f)
    assert out["a"]["b"] == 0.5
    assert out["b"]["c"] == 0.25


def test_build_writes_all_json(tmp_path, dummy_entropy, dummy_dataset):
    builder = LanguageModelBuilder(
        "kotlin",
        [
            ("KStack-clean", "content", lambda row: True),
            ("KExercises", "solution", lambda row: True),
        ],
        tmp_path,
    )
    builder.TOKENS_PER_DUMP = 100  # big enough to avoid many partials
    builder.build()
    # Check output files
    out_dir = builder.output_dir
    for name in ["unigram.json", "bigram.json", "left.json", "metadata.json"]:
        assert (out_dir / name).exists()
    # Check metadata content
    with (out_dir / "metadata.json").open() as f:
        meta = json.load(f)
    assert "model_name" in meta
    assert meta["partials"] >= 1
    assert meta["tokens_total"] == builder.total_unigrams


def test_merge_partials_handles_no_partials(tmp_path, dummy_entropy, dummy_dataset):
    builder = LanguageModelBuilder(
        "kotlin", [("KStack-clean", "content", lambda row: True)], tmp_path
    )
    shutil.rmtree(builder.tmp_dir)
    import os

    orig_rmtree = shutil.rmtree

    def safe_rmtree(path, *args, **kwargs):
        if os.path.exists(path):
            orig_rmtree(path, *args, **kwargs)

    shutil.rmtree = safe_rmtree
    try:
        final_uni, final_bi = builder._merge_partials()
        assert final_uni == {}
        assert final_bi == {}
    finally:
        shutil.rmtree = orig_rmtree


def test_process_dataset_respects_filter(tmp_path, dummy_entropy, dummy_dataset):
    # Only keep rows with "x" in content
    datasets = [
        ("KStack-clean", "content", lambda row: "x" in row["content"]),
    ]
    builder = LanguageModelBuilder("kotlin", datasets, tmp_path)
    builder._process_dataset(
        "KStack-clean", "content", lambda row: "x" in row["content"]
    )
    # Only dummy rows from dummy_dataset with "x y" (so, two rows)
    # Each "x y" yields two tokens, so total_unigrams should be 0 (since dummy_dataset only gives "a b c" for KStack)
    assert builder.total_unigrams == 0
