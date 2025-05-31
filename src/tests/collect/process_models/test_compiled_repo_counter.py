import json
import csv
from unittest import mock

import pytest

from src.main.collect.process_models.compiled_repo_counter import CompiledRepoCounter


@pytest.fixture
def sample_dataset(tmp_path):
    """
    Creates a fake dataset structure like:
    tmp_path/
      modelA/originals/repo1/File.kt
      modelA/bytecode/repo1/
      modelB/originals/
      modelB/bytecode/
      modelC/ (no originals/bytecode)
    """
    (tmp_path / "modelA" / "originals" / "repo1").mkdir(parents=True)
    (tmp_path / "modelA" / "originals" / "repo1" / "File.kt").write_text("kotlin code")
    (tmp_path / "modelA" / "bytecode" / "repo1").mkdir(parents=True)

    (tmp_path / "modelB" / "originals").mkdir(parents=True)
    (tmp_path / "modelB" / "bytecode").mkdir(parents=True)
    (tmp_path / "modelC").mkdir()
    return tmp_path


def test_count_compiles_correct_results(sample_dataset):
    crc = CompiledRepoCounter(sample_dataset)
    crc.count()
    # Only modelA and modelB should be present (they have originals)
    fields = [x[0] for x in crc.results]
    assert "modelA" in fields
    assert "modelB" in fields
    # modelA has 1 compiled repo (repo1)
    for field, count in crc.results:
        if field == "modelA":
            assert count == 1
        elif field == "modelB":
            assert count == 0


def test_count_compiled_info_paths(sample_dataset):
    crc = CompiledRepoCounter(sample_dataset)
    crc.count()
    # Should include File.kt under modelA
    files = crc.compiled_info["modelA"]
    assert any(f == "repo1/File.kt" for f in files)
    # modelB should have no files
    assert crc.compiled_info["modelB"] == []


def test_save_results_creates_csv_and_json(sample_dataset, monkeypatch):
    crc = CompiledRepoCounter(sample_dataset)
    crc.count()
    # Patch out _save_plot so we don't actually save a plot file
    monkeypatch.setattr(crc, "_save_plot", lambda: None)
    crc.save_results()
    # Check CSV
    csv_path = sample_dataset / "bytecode_repo_counts.csv"
    with csv_path.open() as f:
        lines = list(csv.reader(f))
    assert lines[0] == ["field", "repo_count"]
    assert any("modelA" in row and "1" in row for row in lines)
    # Check JSON
    json_path = sample_dataset / "compiled_repos.json"
    with json_path.open() as f:
        data = json.load(f)
    assert "modelA" in data
    assert isinstance(data["modelA"], list)


def test_save_plot_creates_png(sample_dataset, monkeypatch):
    crc = CompiledRepoCounter(sample_dataset)
    crc.count()
    # Patch plt.savefig to a mock
    import matplotlib.pyplot as plt

    mock_savefig = mock.Mock()
    monkeypatch.setattr(plt, "savefig", mock_savefig)
    crc._save_plot()
    plot_path = sample_dataset / "bytecode_repo_counts.png"
    mock_savefig.assert_called_once_with(plot_path)
