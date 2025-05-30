from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from main.collect.bytecode.bytecode_pair_collector import BytecodePairCollector


@pytest.fixture
def temp_dataset(tmp_path):
    """
    Creates a temporary dataset structure with originals/ and bytecode/.
    """
    originals = tmp_path / "originals" / "repo1"
    bytecode = tmp_path / "bytecode" / "repo1"
    originals.mkdir(parents=True)
    bytecode.mkdir(parents=True)
    # Write a Kotlin file with a package
    kt1 = originals / "foo.kt"
    kt1.write_text("package test.pkg\nfun x() = 1\n", encoding="utf-8")
    # Write a Kotlin file with no package
    kt2 = originals / "Bar.kt"
    kt2.write_text("// comment only\nfun y() = 2\n", encoding="utf-8")
    # Create bytecode files
    cls1 = bytecode / "test" / "pkg" / "FooKt.class"
    cls1.parent.mkdir(parents=True, exist_ok=True)
    cls1.write_text("dummybytecode", encoding="utf-8")
    cls2 = bytecode / "Bar.class"
    cls2.write_text("dummybytecode", encoding="utf-8")
    return tmp_path


def test_read_package_detects_package(temp_dataset):
    collector = BytecodePairCollector(temp_dataset)
    # With package
    f = temp_dataset / "originals" / "repo1" / "foo.kt"
    assert collector.read_package(f) == "test.pkg"
    # Without package
    f2 = temp_dataset / "originals" / "repo1" / "Bar.kt"
    assert collector.read_package(f2) == ""


def test_get_root_name_and_guess_kt_filename():
    root = "FooKt"
    assert BytecodePairCollector.get_root_name(Path("FooKt.class")) == "FooKt"
    assert BytecodePairCollector.guess_kt_filename("FooKt") == "foo.kt"
    assert BytecodePairCollector.guess_kt_filename("Bar") == "bar.kt"


def test_index_kt_files_indexes_by_dir_and_pkg(temp_dataset, monkeypatch):
    collector = BytecodePairCollector(temp_dataset)
    # Patch out pool usage to call _index_one sequentially
    monkeypatch.setattr("main.collect.bytecode.bytecode_pair_collector.ProcessPoolExecutor",
                        lambda *a, **kw: MagicMock(__enter__=lambda s: s, __exit__=lambda s, a, b, c: None))
    monkeypatch.setattr("main.collect.bytecode.bytecode_pair_collector.multiprocessing", MagicMock(cpu_count=lambda: 1))
    files = list((temp_dataset / "originals").rglob("*.kt"))

    # Patch pool.map to just map function locally
    def fake_map(fn, tasks):
        return map(fn, tasks)

    with patch("main.collect.bytecode.bytecode_pair_collector.ProcessPoolExecutor") as mock_pool:
        mock_pool.return_value.__enter__.return_value = mock_pool
        mock_pool.map.side_effect = fake_map
        idx_dir, idx_pkg = collector.index_kt_files()
    # Two files, indexed by DirKey and PkgKey
    assert len(idx_dir) == 2
    assert len(idx_pkg) == 2
    assert any("foo.kt" in k for k in idx_dir)
    assert any("bar.kt" in k for k in idx_dir)


def test_build_pairs_finds_match(temp_dataset, monkeypatch):
    collector = BytecodePairCollector(temp_dataset)
    # Patch index_kt_files to controlled values
    fake_idx_dir = {("repo1", ("test", "pkg"), "foo.kt"): temp_dataset / "originals/repo1/foo.kt"}
    fake_idx_pkg = {("repo1", ("test", "pkg"), "foo.kt"): [temp_dataset / "originals/repo1/foo.kt"]}
    monkeypatch.setattr(collector, "index_kt_files", lambda: (fake_idx_dir, fake_idx_pkg))
    # Create matching class in bytecode/repo1/test/pkg/FooKt.class
    result = collector.build_pairs()
    assert (temp_dataset / "originals/repo1/foo.kt") in result
    # The list of class files for that kt must contain the correct .class path
    paths = result[temp_dataset / "originals/repo1/foo.kt"]
    assert any("FooKt.class" in str(p) for p in paths)


def test_build_record_runs_command_and_yields_json(tmp_path, monkeypatch):
    collector = BytecodePairCollector(tmp_path)
    kt = tmp_path / "k.kt"
    cls = tmp_path / "C.class"
    kt.write_text("fun main() = 1\n", encoding="utf-8")
    # Patch run_command to controlled return
    monkeypatch.setattr(collector, "run_command", lambda cmd: ("BCODE", "", 0))
    collector.originals_root = tmp_path  # patch so relative_to works
    collector.bytecode_root = tmp_path
    res = collector._build_record((kt, [cls]))
    assert res is not None
    dct = __import__("json").loads(res)
    assert dct["kt_path"].endswith("k.kt")
    assert "classes" in dct and dct["classes"][0]["javap"] == "BCODE"


def test_build_record_skips_all_empty(monkeypatch, tmp_path):
    collector = BytecodePairCollector(tmp_path)
    kt = tmp_path / "k.kt"
    cls = tmp_path / "C.class"
    kt.write_text("fun main() = 1\n", encoding="utf-8")
    monkeypatch.setattr(collector, "run_command", lambda cmd: ("", "", 0))
    collector.originals_root = tmp_path
    collector.bytecode_root = tmp_path
    res = collector._build_record((kt, [cls]))
    assert res is None


def test_write_jsonl_writes_lines(monkeypatch, tmp_path):
    collector = BytecodePairCollector(tmp_path)
    kt = tmp_path / "k.kt"
    cls = tmp_path / "C.class"
    kt.write_text("fun main() = 1\n", encoding="utf-8")
    monkeypatch.setattr(collector, "_build_record", lambda task: '{"dummy":"yes"}')
    out_path = tmp_path / "out.jsonl"
    pairs = {kt: [cls]}
    # Patch ProcessPoolExecutor to just map sequentially for test
    with patch("main.collect.bytecode.bytecode_pair_collector.ProcessPoolExecutor") as mock_pool:
        mock_pool.return_value.__enter__.return_value = mock_pool
        mock_pool.map.side_effect = lambda f, tasks, chunksize=None: map(f, tasks)
        collector.write_jsonl(pairs, out_path)
    # Check written file
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert any('"dummy":"yes"' in l for l in lines)
