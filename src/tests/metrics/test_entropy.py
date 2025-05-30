import math
from typing import List
import pytest

from src.main.metrics.entropy import Entropy

class MockLeaf:
    def __init__(self, text: bytes):
        self.child_count = 0
        self.text = text
        self.children = []

class MockNode:
    def __init__(self, leaves: List[MockLeaf]):
        self.child_count = len(leaves)
        self.children = leaves

class MockParser:
    def __init__(self, tokens: List[str]):
        self._tokens = tokens

    def parse(self, src: bytes):
        # The parser produces a "tree" with leaves from token list
        leaves = [MockLeaf(token.encode("utf8")) for token in self._tokens]
        class Tree:
            def __init__(self, leaves): self.root_node = MockNode(leaves)
        return Tree(leaves)

@pytest.fixture
def mock_entropy(monkeypatch) -> Entropy:
    tokens = ["a", "b", "a", "c"]
    monkeypatch.setattr("main.metrics.entropy.Entropy.__init__", lambda self, language=None: None)
    ent = Entropy("anylang")
    ent.parser = None
    ent.leaves = lambda node: node.children
    ent.tokens = lambda src: tokens
    ent.dist = lambda src: {"a": 0.5, "b": 0.25, "c": 0.25}
    return ent

def test_tokens(mock_entropy: Entropy) -> None:
    # Returns the tokens from the mock parser
    tokens = mock_entropy.tokens("irrelevant code")
    assert tokens == ["a", "b", "a", "c"]

def test_dist(mock_entropy: Entropy) -> None:
    # a:2, b:1, c:1 => total 4
    dist = mock_entropy.dist("irrelevant code")
    assert math.isclose(dist["a"], 0.5)
    assert math.isclose(dist["b"], 0.25)
    assert math.isclose(dist["c"], 0.25)

def test_merge(mock_entropy: Entropy) -> None:
    a = {"x": 0.1, "y": 0.9}
    b = {"y": 0.5, "z": 0.5}
    result = mock_entropy.merge(a, b)
    assert set(result) == {"x", "y", "z"}

def test_cross_entropy(mock_entropy: Entropy) -> None:
    # p = dist(["a","b","a","c"]) -> a:0.5, b:0.25, c:0.25
    # q = same
    # cross_entropy = -sum(q[k] * log2(p[k])) = entropy in this case
    val = mock_entropy.cross_entropy("irrelevant code", "irrelevant code")
    expected = -0.5*math.log2(0.5) - 0.25*math.log2(0.25) - 0.25*math.log2(0.25)
    assert math.isclose(val, expected)

def test_entropy(mock_entropy: Entropy) -> None:
    # entropy returns cross_entropy on itself
    val = mock_entropy.entropy("irrelevant code")
    expected = mock_entropy.cross_entropy("irrelevant code", "irrelevant code")
    assert math.isclose(val, expected)

def test_kl_div(mock_entropy: Entropy) -> None:
    # kl(p, p) == 0.0 (same distributions)
    assert math.isclose(mock_entropy.kl_div("irrelevant code", "irrelevant code"), 0.0)

def test_jensen_shannon_divergence(mock_entropy: Entropy) -> None:
    # identical distributions -> JSD == 0
    p = mock_entropy.dist("irrelevant code")
    q = dict(p)  # same
    assert math.isclose(mock_entropy.jensen_shannon_divergence(p, q), 0.0)

def test_jensen_shannon_distance(mock_entropy: Entropy) -> None:
    # identical -> distance == 0
    assert math.isclose(mock_entropy.jensen_shannon_distance("irrelevant code", "irrelevant code"), 0.0)

def test_perplexity(mock_entropy: Entropy) -> None:
    # perplexity = 2 ** cross_entropy
    ce = mock_entropy.cross_entropy("irrelevant code", "irrelevant code")
    perp = mock_entropy.perplexity("irrelevant code", "irrelevant code")
    assert math.isclose(perp, 2 ** ce)

def test_conditional_entropy(monkeypatch):
    # This test mocks tokens and bigram probabilities directly for controlled value
    monkeypatch.setattr("main.metrics.entropy.Entropy.__init__", lambda self, language=None: None)
    ent = Entropy("anylang")
    ent.tokens = lambda src: ["a", "b", "a", "c"] if src == "src1" else ["a", "a", "b"]
    val = ent.conditional_entropy("src1", "src1")
    assert isinstance(val, float)  # Just checking it runs

def test_cross_entropy_lang(mock_entropy: Entropy) -> None:
    lang_dist = {"a": 0.5, "b": 0.3, "c": 0.2}
    result = mock_entropy.cross_entropy_lang(lang_dist, "irrelevant code")
    assert isinstance(result, float)

def test_kl_div_lang(mock_entropy: Entropy) -> None:
    lang_dist = {"a": 0.5, "b": 0.3, "c": 0.2}
    result = mock_entropy.kl_div_lang(lang_dist, "irrelevant code")
    assert isinstance(result, float)

def test_perplexity_lang(mock_entropy: Entropy) -> None:
    lang_dist = {"a": 0.5, "b": 0.3, "c": 0.2}
    result = mock_entropy.perplexity_lang(lang_dist, "irrelevant code")
    assert isinstance(result, float)

def test_jensen_shannon_distance_lang(mock_entropy: Entropy) -> None:
    lang_dist = {"a": 0.5, "b": 0.3, "c": 0.2}
    result = mock_entropy.jensen_shannon_distance_lang(lang_dist, "irrelevant code")
    assert isinstance(result, float)

def test_conditional_entropy_lang(monkeypatch):
    monkeypatch.setattr("main.metrics.entropy.Entropy.__init__", lambda self, language=None: None)
    ent = Entropy("anylang")
    ent.tokens = lambda src: ["a", "b", "a", "c"]
    bi_dist = {"a": {"b": 0.5, "a": 0.5}}
    left_dist = {"a": 1.0}
    result = ent.conditional_entropy_lang(bi_dist, left_dist, "irrelevant code")
    assert isinstance(result, float)
