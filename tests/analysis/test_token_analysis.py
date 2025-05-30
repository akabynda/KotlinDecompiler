from __future__ import annotations

from typing import Any
from unittest import mock

import pandas as pd
import pytest
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from main.analysis.token_analysis import TokenAnalysis


class MockTokenizer(PreTrainedTokenizerBase):
    """
    Mock tokenizer to isolate token counting logic from actual tokenizers.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, text: str, **kwargs: Any) -> mock.Mock:
        """
        Mocks the tokenizer's behavior by splitting text into words.

        Args:
            text (str): Input text.

        Returns:
            mock.Mock: Mock object with 'input_ids' representing tokens.
        """
        return mock.Mock(input_ids=text.split())


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    """
    Provides a mock tokenizer for testing.
    """
    return MockTokenizer()


@pytest.fixture
def mock_dataset() -> Dataset:
    """
    Provides a mock dataset for testing.
    """
    data = {
        "kt_source": ["fun main() = println(\"Hi\")", "val x = 42"],
        "classes": [
            [{"javap": "public void main() {...}"}],
            [{"javap": "public final class Main {...}"}]
        ]
    }
    return Dataset.from_dict(data)


def test_count_tokens(monkeypatch: pytest.MonkeyPatch, mock_tokenizer: MockTokenizer) -> None:
    """
    Tests token counting logic, isolating from real dataset loading.
    """
    monkeypatch.setattr("main.analysis.token_analysis.TokenAnalysis.__init__",
                        lambda self, dataset_name, tokenizer_name: None)

    analysis = TokenAnalysis(dataset_name="irrelevant", tokenizer_name="irrelevant")
    analysis.tokenizer = mock_tokenizer

    text = "this is a test"
    token_count = analysis.count_tokens(text)

    assert token_count == 4


def test_analyze(monkeypatch: pytest.MonkeyPatch, mock_tokenizer: MockTokenizer, mock_dataset: Dataset) -> None:
    """
    Tests the analysis workflow, ensuring the resulting DataFrame has correct structure.
    """
    monkeypatch.setattr("main.analysis.token_analysis.TokenAnalysis.__init__",
                        lambda self, dataset_name, tokenizer_name: None)

    analysis = TokenAnalysis(dataset_name="irrelevant", tokenizer_name="irrelevant")
    analysis.tokenizer = mock_tokenizer
    analysis.dataset = mock_dataset

    df: pd.DataFrame = analysis.analyze()

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"kt_tokens", "bc_tokens", "ratio_bc_to_kt"}
    assert len(df) == len(mock_dataset)


def test_compute_statistics(capsys: pytest.CaptureFixture[str]) -> None:
    """
    Tests statistics computation logic and its output.
    """
    data = {
        "kt_tokens": [1, 2, 3],
        "bc_tokens": [2, 4, 6],
        "ratio_bc_to_kt": [2.0, 2.0, 2.0]
    }
    df = pd.DataFrame(data)

    TokenAnalysis.compute_statistics(df)

    captured = capsys.readouterr()
    assert "Pearson r" in captured.out
    assert "Spearman" in captured.out
    assert "Mean ratio" in captured.out
    assert "Median ratio" in captured.out


def test_plot_data(monkeypatch: pytest.MonkeyPatch, mock_tokenizer: MockTokenizer, mock_dataset: Dataset) -> None:
    """
    Tests plotting functionality to ensure no errors during plotting.
    """
    monkeypatch.setattr("main.analysis.token_analysis.TokenAnalysis.__init__",
                        lambda self, dataset_name, tokenizer_name: None)

    analysis = TokenAnalysis(dataset_name="irrelevant", tokenizer_name="irrelevant")
    analysis.tokenizer = mock_tokenizer
    analysis.dataset = mock_dataset

    df: pd.DataFrame = analysis.analyze()

    with mock.patch("matplotlib.pyplot.show"):
        analysis.plot_data(df)
