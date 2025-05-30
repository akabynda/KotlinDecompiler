from pathlib import Path
from unittest import mock

import pytest

from main.collect.bytecode.download_kstack_clean import KStackCleanDownloader


@pytest.fixture
def mock_examples():
    """
    Provides a mock dataset with some .kt and non-.kt files.
    """
    return [
        {
            "owner": "user1",
            "name": "repo1",
            "commit_sha": "1234567890abcdef",
            "path": "src/main.kt",
            "content": "fun main() = println(\"Hello\")",
        },
        {
            "owner": "user1",
            "name": "repo1",
            "commit_sha": "1234567890abcdef",
            "path": "README.md",
            "content": "Not a Kotlin file",
        },
        {
            "owner": "user2",
            "name": "repo2",
            "commit_sha": "abcdef1234567890",
            "path": "app/App.kt",
            "content": "// App code",
        },
    ]


def test_init_defaults():
    """
    Tests initialization with default arguments.
    """
    downloader = KStackCleanDownloader()
    assert downloader.output_root == Path("kstack-clean")
    assert downloader.originals_root == Path("kstack-clean/originals")
    assert downloader.split == "train"
    assert downloader.streaming is True


def test_init_custom(tmp_path):
    """
    Tests initialization with custom arguments.
    """
    out_dir = tmp_path / "custom"
    downloader = KStackCleanDownloader(output_root=out_dir, split="test", streaming=False)
    assert downloader.output_root == out_dir
    assert downloader.originals_root == out_dir / "originals"
    assert downloader.split == "test"
    assert downloader.streaming is False


def test_load_dataset_yields_kt_only(monkeypatch, mock_examples):
    """
    Tests that load_dataset only yields .kt examples (mocked).
    """
    downloader = KStackCleanDownloader()
    # Patch load_dataset from datasets to yield our mock examples
    mock_loader = mock.Mock(return_value=iter(mock_examples))
    monkeypatch.setattr("main.collect.bytecode.download_kstack_clean.load_dataset", mock_loader)

    # Should only yield examples with .kt paths
    filtered = list(downloader.load_dataset())
    assert all(ex["path"].endswith(".kt") for ex in filtered)
    assert len(filtered) == 2
    assert filtered[0]["path"] == "src/main.kt"
    assert filtered[1]["path"] == "app/App.kt"

    mock_loader.assert_called_once_with(
        "JetBrains/KStack-clean", split=downloader.split, streaming=downloader.streaming
    )


def test_save_kotlin_sources_creates_files(tmp_path, mock_examples):
    """
    Tests that save_kotlin_sources creates correct file structure for .kt files.
    """
    downloader = KStackCleanDownloader(output_root=tmp_path)
    # Only pass .kt files
    kt_examples = [ex for ex in mock_examples if ex["path"].endswith(".kt")]
    downloader.save_kotlin_sources(kt_examples)

    for ex in kt_examples:
        sha = ex["commit_sha"][:7]
        subdir = f"{ex['owner']}__{ex['name']}__{sha}"
        target_file = tmp_path / "originals" / subdir / ex["path"]
        assert target_file.exists()
        assert target_file.read_text(encoding="utf-8") == ex["content"]


def test_save_kotlin_sources_creates_parent_dirs(tmp_path, mock_examples):
    """
    Tests that save_kotlin_sources creates nested directories as needed.
    """
    downloader = KStackCleanDownloader(output_root=tmp_path)
    kt_examples = [ex for ex in mock_examples if ex["path"].endswith(".kt")]
    downloader.save_kotlin_sources(kt_examples)

    for ex in kt_examples:
        sha = ex["commit_sha"][:7]
        subdir = f"{ex['owner']}__{ex['name']}__{sha}"
        parent_dir = tmp_path / "originals" / subdir / Path(ex["path"]).parent
        assert parent_dir.exists()
        assert parent_dir.is_dir()


def test_process_full_pipeline(monkeypatch, tmp_path, mock_examples):
    """
    Tests the process method end-to-end with mocks.
    """
    downloader = KStackCleanDownloader(output_root=tmp_path)
    kt_examples = [ex for ex in mock_examples if ex["path"].endswith(".kt")]

    monkeypatch.setattr(downloader, "load_dataset", mock.Mock(return_value=kt_examples))
    monkeypatch.setattr(downloader, "save_kotlin_sources", mock.Mock())

    downloader.process()
    downloader.load_dataset.assert_called_once()
    downloader.save_kotlin_sources.assert_called_once_with(kt_examples)
