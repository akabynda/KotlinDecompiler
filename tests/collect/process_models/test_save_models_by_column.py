import json

import pytest

from main.collect.process_models.save_models_by_column import JSONLProcessor


@pytest.fixture
def sample_jsonl(tmp_path):
    # Sample JSONL data: 2 records, 2 models
    records = [
        {"kt_path": "a.kt", "modelA": "fun a() {}", "modelB": {"field": 1}, "classes": None},
        {"kt_path": "b.kt", "modelA": "", "modelB": "fun b() {}", "extra": "will_be_ignored"}
    ]
    jsonl = tmp_path / "data.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return jsonl, records


def test_load_jsonl_reads_lines(sample_jsonl):
    jsonl, records = sample_jsonl
    processor = JSONLProcessor()
    loaded = list(processor.load_jsonl(jsonl))
    assert loaded == records


def test_save_models_by_column_creates_files(tmp_path, sample_jsonl):
    jsonl, records = sample_jsonl
    out_root = tmp_path / "output"
    processor = JSONLProcessor()
    processor.save_models_by_column(jsonl, out_root)

    # Should create modelA/originals/a.kt (with string content)
    fileA = out_root / "modelA" / "originals" / "a.kt"
    assert fileA.exists()
    assert fileA.read_text(encoding="utf-8") == "fun a() {}"
    # Should not create modelA/originals/b.kt (content is "")
    fileA_b = out_root / "modelA" / "originals" / "b.kt"
    assert not fileA_b.exists()

    # Should create modelB/originals/a.kt (with json content)
    fileB = out_root / "modelB" / "originals" / "a.kt"
    assert fileB.exists()
    assert json.loads(fileB.read_text(encoding="utf-8")) == {"field": 1}
    # Should create modelB/originals/b.kt (string)
    fileB_b = out_root / "modelB" / "originals" / "b.kt"
    assert fileB_b.exists()
    assert fileB_b.read_text(encoding="utf-8") == "fun b() {}"


def test_save_models_by_column_handles_no_records(tmp_path):
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("")  # no records
    out_root = tmp_path / "output"
    processor = JSONLProcessor()
    processor.save_models_by_column(jsonl, out_root)
    # Output dir should NOT exist since nothing is written
    assert not out_root.exists()


def test_save_models_by_column_skips_no_kt_path(tmp_path):
    records = [
        {"modelA": "codeA"},  # missing kt_path
        {"kt_path": None, "modelA": "codeB"},  # None kt_path
    ]
    jsonl = tmp_path / "missing.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    out_root = tmp_path / "out"
    processor = JSONLProcessor()
    processor.save_models_by_column(jsonl, out_root)
    model_dir = out_root / "modelA"
    assert model_dir.exists()
    # Should be empty—no 'originals' subdir, no files
    assert not any(model_dir.rglob("*"))


def test_save_models_by_column_nonstring_content(tmp_path):
    records = [
        {"kt_path": "x.kt", "modelA": ["foo", "bar"]},
        {"kt_path": "y.kt", "modelA": {"x": 1}},
    ]
    jsonl = tmp_path / "nonstr.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    out_root = tmp_path / "out"
    processor = JSONLProcessor()
    processor.save_models_by_column(jsonl, out_root)

    file_x = out_root / "modelA" / "originals" / "x.kt"
    file_y = out_root / "modelA" / "originals" / "y.kt"
    assert file_x.exists()
    assert json.loads(file_x.read_text(encoding="utf-8")) == ["foo", "bar"]
    assert file_y.exists()
    assert json.loads(file_y.read_text(encoding="utf-8")) == {"x": 1}
