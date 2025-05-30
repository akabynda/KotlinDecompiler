from pathlib import Path
from unittest import mock
import pytest
import json

from src.main.collect.bytecode.merge_datasets import DatasetMerger


@pytest.fixture
def mock_hf_dataset():
    """Create a minimal fake HuggingFace Dataset-like object."""
    class FakeDataset:
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def shuffle(self, seed=None):
            # Return new shuffled object for test (not actually shuffled)
            return FakeDataset(self.items[::-1])
        def train_test_split(self, test_size):
            n = int(len(self) * (1 - test_size))
            return {"train": FakeDataset(self.items[:n]), "test": FakeDataset(self.items[n:])}
        def to_list(self):
            return self.items
    return FakeDataset

def test_init_sets_attributes(tmp_path):
    merger = DatasetMerger(
        "ds1", "ds2", tmp_path, test_size=0.3, seed=42
    )
    assert merger.dataset1_name == "ds1"
    assert merger.dataset2_name == "ds2"
    assert merger.output_dir == tmp_path
    assert merger.test_size == 0.3
    assert merger.seed == 42

def test_load_and_merge(monkeypatch, mock_hf_dataset):
    FakeDataset = mock_hf_dataset
    fake1 = FakeDataset([{"x": 1}])
    fake2 = FakeDataset([{"y": 2}])

    # Patch load_dataset and concatenate_datasets
    monkeypatch.setattr("main.collect.bytecode.merge_datasets.load_dataset", lambda name, split="train": fake1 if name == "ds1" else fake2)
    monkeypatch.setattr("main.collect.bytecode.merge_datasets.concatenate_datasets", lambda datasets: FakeDataset(fake1.items + fake2.items))

    merger = DatasetMerger("ds1", "ds2", Path(""), test_size=0.2, seed=123)
    merged = merger.load_and_merge()
    assert len(merged) == 2
    assert merged.items == [{"x": 1}, {"y": 2}]

def test_shuffle_and_split(mock_hf_dataset):
    FakeDataset = mock_hf_dataset
    dset = FakeDataset([{"a": i} for i in range(10)])
    merger = DatasetMerger("ds1", "ds2", Path(""), test_size=0.2, seed=1)
    splits = merger.shuffle_and_split(dset)
    assert set(splits.keys()) == {"train", "test"}
    total = len(splits["train"]) + len(splits["test"])
    assert total == len(dset)
    # Should roughly follow test_size split
    assert len(splits["test"]) == 2

def test_save_to_json(tmp_path, mock_hf_dataset):
    FakeDataset = mock_hf_dataset
    ds = FakeDataset([{"foo": "bar"}, {"baz": 1}])
    merger = DatasetMerger("a", "b", tmp_path)
    file = tmp_path / "out.json"
    merger.save_to_json(ds, file)
    with file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == [{"foo": "bar"}, {"baz": 1}]

def test_process_full_pipeline(monkeypatch, tmp_path, mock_hf_dataset):
    FakeDataset = mock_hf_dataset
    ds1 = FakeDataset([{"i": 1}, {"i": 2}])
    ds2 = FakeDataset([{"i": 3}, {"i": 4}])
    merged = FakeDataset(ds1.items + ds2.items)
    split = {"train": FakeDataset([{"i": 1}, {"i": 2}, {"i": 3}]), "test": FakeDataset([{"i": 4}])}

    merger = DatasetMerger("ds1", "ds2", tmp_path, test_size=0.25, seed=123)

    monkeypatch.setattr("main.collect.bytecode.merge_datasets.load_dataset", lambda name, split="train": ds1 if name == "ds1" else ds2)
    monkeypatch.setattr("main.collect.bytecode.merge_datasets.concatenate_datasets", lambda datasets: merged)
    monkeypatch.setattr(merger, "shuffle_and_split", lambda d: split)
    monkeypatch.setattr(merger, "save_to_json", mock.Mock())

    merger.process()
    # Should call save_to_json for train and test splits
    assert merger.save_to_json.call_count == 2
    # Check output file paths are correct
    called_paths = [call.args[1].name for call in merger.save_to_json.call_args_list]
    assert "train.json" in called_paths and "test.json" in called_paths

