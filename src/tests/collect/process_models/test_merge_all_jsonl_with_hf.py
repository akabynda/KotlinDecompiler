import json

import pytest

from src.main.collect.process_models.merge_all_jsonl_with_hf import DatasetMerger


@pytest.fixture
def tmp_jsonl_files(tmp_path):
    # Create two fake JSONL files with overlapping 'kt_path's
    d1 = {"kt_path": "f1.kt", "local1": "foo"}
    d2 = {"kt_path": "f2.kt", "local1": "bar"}
    d3 = {"kt_path": "f1.kt", "local2": 123}
    d4 = {"kt_path": "f3.kt", "local2": 456}
    file1 = tmp_path / "a.jsonl"
    file2 = tmp_path / "b.jsonl"
    file1.write_text("\n".join(json.dumps(x) for x in [d1, d2]), encoding="utf-8")
    file2.write_text("\n".join(json.dumps(x) for x in [d3, d4]), encoding="utf-8")
    return tmp_path, [file1, file2]

@pytest.fixture
def mock_hf(monkeypatch):
    # Base HuggingFace dataset returns two rows
    data = [
        {"kt_path": "f1.kt", "base": "yes"},
        {"kt_path": "f2.kt", "base": "no"},
        {"kt_path": "f3.kt", "base": "zzz"}
    ]
    monkeypatch.setattr(
        "main.collect.process_models.merge_all_jsonl_with_hf.load_dataset",
        lambda name, split: data
    )
    return data

def test_merge_with_hf_dataset(tmp_jsonl_files, mock_hf, tmp_path):
    in_dir, files = tmp_jsonl_files
    out_path = tmp_path / "out.jsonl"
    merger = DatasetMerger()
    merger.config.dataset_name = "dummy"
    merger.config.split = "train"

    merger.merge_with_hf_dataset(in_dir, out_path)
    # Check output: Should contain f1, f2, f3 merged, and not drop base rows with all-NA locals
    with out_path.open() as f:
        lines = [json.loads(l) for l in f if l.strip()]
    paths = {row["kt_path"] for row in lines}
    assert paths == {"f1.kt", "f2.kt", "f3.kt"}
    # Should contain merged columns
    assert any("local1" in row or "local2" in row for row in lines)
    # Should drop rows with no local columns at all
    all_local_empty = [row for row in lines if not any(k.startswith("local") for k in row)]
    assert not all_local_empty

def test_duplicate_columns_are_removed(tmp_jsonl_files, mock_hf, tmp_path):
    in_dir, files = tmp_jsonl_files
    # Add a duplicate column in local file
    file1 = in_dir / "dup.jsonl"
    file1.write_text(json.dumps({"kt_path": "f2.kt", "base": "SHOULD_BE_DROPPED", "localx": "hi"}) + "\n", encoding="utf-8")
    out_path = tmp_path / "out.jsonl"
    merger = DatasetMerger()
    merger.config.dataset_name = "dummy"
    merger.config.split = "train"
    merger.merge_with_hf_dataset(in_dir, out_path)
    # Output row for f2.kt should have 'base' from HF, not the local file (should be dropped)
    with out_path.open() as f:
        rows = [json.loads(l) for l in f if l.strip()]
    row = next(r for r in rows if r["kt_path"] == "f2.kt")
    assert row["base"] == "no"



