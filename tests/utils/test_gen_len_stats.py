import numpy as np
import pytest

from main.utils.gen_len_stats import gen_len_stats, get_max_new


# Minimal mock of the Row object
class Row:
    def __init__(self, kt_source, bytecode=None):
        self.kt_source = kt_source
        self.bytecode = bytecode


# Minimal tokenizer mock that returns a dummy object with an input_ids attribute
class MockTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping  # dict: str -> list[int]

    def __call__(self, text):
        return type("FakeTokenized", (), {"input_ids": self.mapping.get(text, [])})


@pytest.mark.parametrize(
    "rows, token_map, expected",
    [
        pytest.param(
            # Mix of short and long lengths
            [Row("a" * 4), Row("b" * 8), Row("c" * 2)],
            {"a" * 4: [1, 2, 3, 4], "b" * 8: list(range(8)), "c" * 2: [1, 2]},
            int(np.percentile([4, 8, 2], 95) * 1.5),
            id="short-and-long"
        ),
        pytest.param(
            # All rows with the same length
            [Row("a"), Row("a"), Row("a")],
            {"a": [1, 2, 3]},
            int(np.percentile([3, 3, 3], 95) * 1.5),
            id="all-same"
        ),
        pytest.param(
            # Only one row
            [Row("a")],
            {"a": [1]},
            int(np.percentile([1], 95) * 1.5),
            id="single-row"
        ),
        pytest.param(
            # Two rows with different lengths
            [Row("foo"), Row("bar")],
            {"foo": [1, 2, 3], "bar": [1]},
            int(np.percentile([3, 1], 95) * 1.5),
            id="two-different"
        ),
    ]
)
def test_get_max_new(rows, token_map, expected):
    tokenizer = MockTokenizer(token_map)
    assert get_max_new(rows, tokenizer) == expected


@pytest.mark.parametrize(
    "rows, token_map, expected_max, expected_median",
    [
        pytest.param(
            # Three examples: lengths [4, 8, 2]; ratios: [0.4, 0.5, 0] -> median = 0.4, max = 8 (95th percentile) * 1.5 = 12
            [Row("a", "x" * 10), Row("b", "y" * 16), Row("c", "")],
            {
                "a": [1, 2, 3, 4], "b": list(range(8)), "c": [1, 2],
                "x" * 10: list(range(10)), "y" * 16: list(range(16)), "": [],
            },
            int(np.percentile([4, 8, 2], 95) * 1.5),
            0.4,  # min(0.5, median([4/10,8/16,0]))
            id="with-empty-bytecode"
        ),
        pytest.param(
            # Lengths and ratios: (10,10)=1, (2,4)=0.5, (6,6)=1, median=1, capped at 0.5
            [Row("a", "A"), Row("b", "BB"), Row("c", "CCC")],
            {
                "a": list(range(10)), "A": list(range(10)),
                "b": [1, 2], "BB": [1, 2, 3, 4],
                "c": [1, 2, 3, 4, 5, 6], "CCC": [1, 2, 3, 4, 5, 6],
            },
            int(np.percentile([10, 2, 6], 95) * 1.5),
            0.5,  # min(0.5, median([1.0,0.5,1.0]))
            id="median-capped"
        ),
        pytest.param(
            # Only one row, kt=5, bc=2, ratio=2.5, median=2.5 (capped at 0.5)
            [Row("k", "b")],
            {"k": [1, 2, 3, 4, 5], "b": [1, 2]},
            int(np.percentile([5], 95) * 1.5),
            0.5,
            id="single-row-capped"
        ),
        pytest.param(
            # Only one, empty bytecode, ratio=0
            [Row("x", "")],
            {"x": [1, 2, 3], "": []},
            int(np.percentile([3], 95) * 1.5),
            0,
            id="single-row-empty-bc"
        ),
    ]
)
def test_gen_len_stats(rows, token_map, expected_max, expected_median):
    tokenizer = MockTokenizer(token_map)
    result_max, result_median = gen_len_stats(rows, tokenizer)
    assert result_max == expected_max
    assert result_median == expected_median
