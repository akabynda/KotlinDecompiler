from pathlib import Path
from unittest import mock

import pytest

from src.main.collect.bytecode.download_kexercises import KExercisesDownloader


@pytest.fixture
def mock_examples():
    """
    Provides a small iterable of mock examples as would be yielded by the dataset.
    """
    return [
        {"problem": 'fun main() { println("Hello") }', "solution": "// solution code"},
        {"problem": "fun sum(a:Int,b:Int)=a+b", "solution": "// add implementation"},
    ]


def test_init_defaults():
    """
    Tests initialization with default arguments.
    """
    downloader = KExercisesDownloader()
    assert downloader.split == "train"
    assert downloader.streaming is True
    assert downloader.output_dir == Path("kexercises/originals")


def test_init_custom(tmp_path):
    """
    Tests initialization with custom arguments.
    """
    out_dir = tmp_path / "custom"
    downloader = KExercisesDownloader(split="test", streaming=False, output_dir=out_dir)
    assert downloader.split == "test"
    assert downloader.streaming is False
    assert downloader.output_dir == out_dir


def test_load_dataset_yields_examples(monkeypatch, mock_examples):
    """
    Tests that load_dataset yields items as expected (mocked).
    """
    downloader = KExercisesDownloader()
    # Mock datasets.load_dataset to yield our examples
    mock_loader = mock.Mock(return_value=iter(mock_examples))
    monkeypatch.setattr(
        "src.main.collect.bytecode.download_kexercises.load_dataset", mock_loader
    )
    result = list(downloader.load_dataset())
    assert result == mock_examples
    mock_loader.assert_called_once_with(
        "JetBrains/KExercises", split=downloader.split, streaming=downloader.streaming
    )


def test_save_exercises_creates_files(tmp_path, mock_examples):
    """
    Tests that save_exercises writes the correct files and content.
    """
    downloader = KExercisesDownloader(output_dir=tmp_path)
    downloader.save_exercises(mock_examples)

    for idx, ex in enumerate(mock_examples):
        folder = tmp_path / f"{idx}"
        file = folder / f"solution_{idx}.kt"
        assert file.exists()
        expected = f"{ex['problem'].strip()}\n{ex['solution'].strip()}"
        assert file.read_text(encoding="utf-8") == expected


def test_save_exercises_empty_problem_solution(tmp_path):
    """
    Tests that save_exercises handles missing 'problem' and 'solution' fields gracefully.
    """
    downloader = KExercisesDownloader(output_dir=tmp_path)
    data = [{}]  # No fields
    downloader.save_exercises(data)
    file = tmp_path / "0" / "solution_0.kt"
    assert file.exists()
    assert file.read_text(encoding="utf-8") == "\n"


def test_process_full_pipeline(monkeypatch, tmp_path, mock_examples):
    """
    Tests the end-to-end process method with mocks (no real dataset download).
    """
    downloader = KExercisesDownloader(output_dir=tmp_path)

    # Patch load_dataset and save_exercises
    monkeypatch.setattr(
        downloader, "load_dataset", mock.Mock(return_value=mock_examples)
    )
    monkeypatch.setattr(downloader, "save_exercises", mock.Mock())

    downloader.process()
    downloader.load_dataset.assert_called_once()
    downloader.save_exercises.assert_called_once_with(mock_examples)


def test_save_exercises_creates_parent_dirs(tmp_path, mock_examples):
    """
    Tests that save_exercises creates parent directories as needed.
    """
    deep_dir = tmp_path / "some" / "deep" / "folder"
    downloader = KExercisesDownloader(output_dir=deep_dir)
    downloader.save_exercises(mock_examples)
    for idx in range(len(mock_examples)):
        assert (deep_dir / str(idx)).is_dir()
