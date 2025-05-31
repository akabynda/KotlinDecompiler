import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.main.metrics import registry
from src.main.metrics.entropy import Entropy
from src.main.utils.kotlin_parser import parse


class MetricsCollector:
    """
    Collects metrics and processes Kotlin decompiled code for analysis.
    """

    def __init__(self, language: str = "kotlin") -> None:
        """
        Initialize the metrics collector.

        Args:
            language (str): Programming language to analyze. Default is "kotlin".
        """
        self.language: str = language
        self.entropy: Entropy = Entropy(language)
        self.decompilers: Set[str] = {"Bytecode", "CFR", "JDGUI", "Fernflower"}
        self.converters: Set[str] = {"ChatGPT", "J2K"}
        self._issue_re = re.compile(
            r"^(?P<issue>\w+)\s+-.*\s+at\s+(?P<path>/.*?):\d+:\d+"
        )

    @staticmethod
    def read_kt(path: Path) -> str:
        """
        Read Kotlin source code from a file or recursively from a directory.

        Args:
            path (Path): Path to the file or directory.

        Returns:
            str: Concatenated Kotlin source code.
        """
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return "\n".join(p.read_text(encoding="utf-8") for p in path.rglob("*.kt"))

    @staticmethod
    def read_kt_flat(path: Path) -> str:
        """
        Read Kotlin source code from a file or flat directory.

        Args:
            path (Path): Path to the file or directory.

        Returns:
            str: Concatenated Kotlin source code.
        """
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return "\n".join(p.read_text(encoding="utf-8") for p in path.glob("*.kt"))

    @staticmethod
    def structural(src: str) -> Dict[str, float]:
        """
        Compute structural metrics using the registry.

        Args:
            src (str): Source code.

        Returns:
            dict: Dictionary with structural metric names and values.
        """
        tree = parse(src)
        return {name: fn(tree) for name, fn in registry.items()}

    def entropy_metrics(self, orig: str, dec: str) -> Dict[str, float]:
        """
        Compute entropy-based metrics between original and decompiled code.

        Args:
            orig (str): Original code.
            dec (str): Decompiled code.

        Returns:
            dict: Dictionary with entropy metric names and values.
        """
        return {
            "CE": self.entropy.cross_entropy(orig, dec),
            "KL": self.entropy.kl_div(orig, dec),
            "PPL": self.entropy.perplexity(orig, dec),
            "JSD": self.entropy.jensen_shannon_distance(orig, dec),
            "CondE": self.entropy.conditional_entropy(orig, dec),
        }

    def load_lm(
        self, lm_dir: Optional[Path] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Load language model probabilities from JSON files.

        Args:
            lm_dir (Optional[Path]): Directory containing the language model. If None, asks the user.

        Returns:
            tuple: unigram, bigram, and left probabilities.
        """
        if lm_dir is None:
            lm_dir = Path(input("Path to language model: "))
        with open(lm_dir / "unigram.json", encoding="utf-8") as f:
            p_uni = json.load(f)
        with open(lm_dir / "bigram.json", encoding="utf-8") as f:
            p_bi = json.load(f)
        with open(lm_dir / "left.json", encoding="utf-8") as f:
            p_left = json.load(f)
        return p_uni, p_bi, p_left

    def lm_metrics(
        self,
        p_uni: Dict[str, Any],
        p_bi: Dict[str, Any],
        p_left: Dict[str, Any],
        src: str,
    ) -> Dict[str, float]:
        """
        Compute language model-based metrics.

        Args:
            p_uni (dict): Unigram probabilities.
            p_bi (dict): Bigram probabilities.
            p_left (dict): Left-context probabilities.
            src (str): Source code.

        Returns:
            dict: Dictionary with LM metric names and values.
        """
        return {
            "LM_CE": self.entropy.cross_entropy_lang(p_uni, src),
            "LM_KL": self.entropy.kl_div_lang(p_uni, src),
            "LM_PPL": self.entropy.perplexity_lang(p_uni, src),
            "LM_JSD": self.entropy.jensen_shannon_distance_lang(p_uni, src),
            "LM_CondE": self.entropy.conditional_entropy_lang(p_bi, p_left, src),
        }

    def collect_tests(self, test_root: Path) -> Dict[str, Dict[str, Any]]:
        """
        Collect original and decompiled code for each test directory.

        Args:
            test_root (Path): Root directory with tests.

        Returns:
            dict: Dictionary with test data.
        """
        tests: Dict[str, Dict[str, Any]] = {}
        for td in test_root.iterdir():
            if not td.is_dir():
                continue
            orig_code = self.read_kt_flat(td)
            if not orig_code:
                continue
            tests[td.name] = {"orig": orig_code, "decs": {}}

            for dec in self.decompilers:
                root = td / dec
                if not root.is_dir():
                    continue

                buckets: Dict[str, Set[Path]] = defaultdict(set)
                for f in root.glob("*.kt"):
                    if "CodeConvert" in f.stem:
                        continue
                    cv = next((c for c in self.converters if c in f.stem), None)
                    if cv:
                        buckets[f"{dec}{cv}"].add(f)

                for sub in root.iterdir():
                    if not sub.is_dir() or "CodeConvert" in sub.name:
                        continue
                    cv = next((c for c in self.converters if c in sub.name), None)
                    if cv:
                        buckets[f"{dec}{cv}"].add(sub)

                for cat, paths in buckets.items():
                    code = "\n".join(self.read_kt(p) for p in sorted(paths))
                    tests[td.name]["decs"][cat] = code

        return tests

    @staticmethod
    def build_pairs(
        tests: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, str, str, str]]:
        """
        Build pairs of (test, category, decompiled/original code, original code).

        Args:
            tests (dict): Dictionary with test data.

        Returns:
            list: List of pairs.
        """
        pairs: List[Tuple[str, str, str, str]] = []
        for test, data in tests.items():
            pairs.append((test, "Original", data["orig"], data["orig"]))
            for cat, code in data["decs"].items():
                pairs.append((test, cat, code, data["orig"]))

        if not pairs:
            raise ValueError("No original/decompiled pairs found.")

        return pairs
