import json
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

from datasets import load_dataset
from tqdm import tqdm

from src.main.metrics.entropy import Entropy


class LanguageModelBuilder:
    """
    Builds a language model from datasets using unigram and bigram statistics.
    """

    TOKENS_PER_DUMP: int = 10_000_000

    def __init__(
            self,
            language: str,
            datasets: list[tuple[str, str, Callable[[dict], bool]]],
            output_dir: Path
    ) -> None:
        """
        Initialize the LanguageModelBuilder.

        Args:
            language: Programming language (e.g., 'kotlin').
            datasets: List of dataset configurations: (dataset name, column name, filter function).
            output_dir: Directory to store the generated model files.
        """
        self.language: str = language
        self.datasets: list[tuple[str, str, Callable[[dict], bool]]] = datasets
        self.model_name: str = "+".join(d[0] for d in datasets)
        self.output_dir: Path = output_dir / language.lower() / self.model_name.lower()
        self.tmp_dir: Path = self.output_dir / "_partials"
        self.entropy: Entropy = Entropy(language)

        self.unigram_counter: Counter[str] = Counter()
        self.bigram_counter: Counter[tuple[str, str]] = Counter()

        self.total_unigrams: int = 0
        self.total_bigrams: int = 0
        self.token_buffer: int = 0
        self.partial_idx: int = 0

        self._setup_dirs()

    def _setup_dirs(self) -> None:
        """Create necessary directories for output and partial files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(exist_ok=True)

    def _dump_partial(self, idx: int) -> None:
        """Dump current unigram and bigram counts to a partial file."""
        fname: Path = self.tmp_dir / f"partial-{idx}.pkl"
        with fname.open("wb") as f:
            pickle.dump(
                (self.unigram_counter.copy(), self.bigram_counter.copy()),
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        self.unigram_counter.clear()
        self.bigram_counter.clear()

    def _process_dataset(self, dname: str, col: str, keep: Callable[[dict], bool]) -> None:
        """Process a single dataset and update unigram and bigram counters."""
        ds: Iterable[dict] = load_dataset(f"JetBrains/{dname}", split="train", streaming=True)
        for row in tqdm(ds, desc=dname, leave=False):
            if not keep(row):
                continue
            tokens: list[str] = self.entropy.tokens(row[col])
            self.unigram_counter.update(tokens)
            self.bigram_counter.update(zip(tokens, tokens[1:]))

            self.total_unigrams += len(tokens)
            self.total_bigrams += max(0, len(tokens) - 1)
            self.token_buffer += len(tokens)

            if self.token_buffer >= self.TOKENS_PER_DUMP:
                self._dump_partial(self.partial_idx)
                self.partial_idx += 1
                self.token_buffer = 0

    def _merge_partials(self) -> tuple[Counter[str], Counter[tuple[str, str]]]:
        """Merge all partial files into final unigram and bigram counters."""
        final_uni: Counter[str] = Counter()
        final_bi: Counter[tuple[str, str]] = Counter()

        for pkl in self.tmp_dir.glob("partial-*.pkl"):
            with pkl.open("rb") as f:
                u, b = pickle.load(f)
            final_uni.update(u)
            final_bi.update(b)

        shutil.rmtree(self.tmp_dir)
        return final_uni, final_bi

    def build(self) -> None:
        """
        Build the language model: process datasets, compute probabilities, and store results.
        """
        for dname, col, keep in self.datasets:
            self._process_dataset(dname, col, keep)

        if self.token_buffer > 0:
            self._dump_partial(self.partial_idx)

        final_uni, final_bi = self._merge_partials()

        # Calculate probabilities
        uni_prob: dict[str, float] = {
            t: v / self.total_unigrams for t, v in final_uni.items()
        }
        bi_prob: dict[tuple[str, str], float] = {
            k: v / self.total_bigrams for k, v in final_bi.items()
        }
        left_prob: dict[str, float] = {
            a: v / self.total_bigrams for a, v in Counter(a for a, _ in final_bi).items()
        }

        # Save results
        self._save_json(self.output_dir / "unigram.json", uni_prob)
        self._save_nested_bigram(bi_prob)
        self._save_json(self.output_dir / "left.json", left_prob)

        # Save metadata
        meta: dict[str, int | str] = {
            "model_name": self.model_name,
            "num_unigrams": len(uni_prob),
            "num_bigrams": len(bi_prob),
            "tokens_total": self.total_unigrams,
            "bigrams_total": self.total_bigrams,
            "partials": self.partial_idx + 1,
            "tokens_per_dump": self.TOKENS_PER_DUMP,
        }
        self._save_json(self.output_dir / "metadata.json", meta, indent=2)

        print(
            f"{self.model_name}: {meta['num_unigrams']:,} tokens / "
            f"{meta['num_bigrams']:,} bigrams, {meta['partials']} partial files merged"
        )

    def _save_json(
            self, path: Path, data: dict, indent: int | None = None
    ) -> None:
        """Save a dictionary as a JSON file."""
        with path.open("w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

    def _save_nested_bigram(self, bi_prob: dict[tuple[str, str], float]) -> None:
        """Save nested bigram probabilities to a JSON file."""
        nested: dict[str, dict[str, float]] = defaultdict(dict)
        for (a, b), p in bi_prob.items():
            nested[a][b] = p
        self._save_json(self.output_dir / "bigram.json", nested)


if __name__ == "__main__":
    LANGUAGE: str = "kotlin"
    DATASETS: list[tuple[str, str, Callable[[dict], bool]]] = [
        ("KStack-clean", "content", lambda row: bool(row["content"])),
        ("KExercises", "solution", lambda row: bool(row["solution"])),
    ]
    OUTPUT_DIR: Path = Path("lang_models")

    builder = LanguageModelBuilder(LANGUAGE, DATASETS, OUTPUT_DIR)
    builder.build()
