from pathlib import Path
from typing import Any, Dict, Iterable

from datasets import load_dataset


class KStackCleanDownloader:
    """
    Processes the JetBrains/KStack-clean dataset and saves .kt source files.
    """

    def __init__(
            self,
            output_root: Path = Path("./kstack-clean"),
            split: str = "train",
            streaming: bool = True,
    ) -> None:
        """
        Initialize the processor.

        Args:
            output_root (Path): Root directory to save extracted sources.
            split (str): Dataset split to use (default: "train").
            streaming (bool): Whether to stream the dataset (default: True).
        """
        self.output_root: Path = output_root
        self.originals_root: Path = output_root / "originals"
        self.split: str = split
        self.streaming: bool = streaming

    def load_dataset(self) -> Iterable[Dict[str, Any]]:
        """
        Load and filter the Kotlin dataset to yield .kt files only.

        Yields:
            dict: Example containing a Kotlin file.
        """
        dataset = load_dataset(
            "JetBrains/KStack-clean",
            split=self.split,
            streaming=self.streaming
        )
        for example in dataset:
            if example.get("path", "").endswith(".kt"):
                yield example

    def save_kotlin_sources(self, dataset: Iterable[Dict[str, Any]]) -> None:
        """
        Save each .kt file to the output directory with structured paths.

        Args:
            dataset (Iterable[dict]): Dataset containing Kotlin source files.
        """
        for example in dataset:
            owner: str = example["owner"]
            repo: str = example["name"]
            sha: str = example["commit_sha"][:7]
            rel_dir: Path = self.originals_root / f"{owner}__{repo}__{sha}"
            dst_file: Path = rel_dir / example["path"]

            dst_file.parent.mkdir(parents=True, exist_ok=True)
            dst_file.write_text(example["content"], encoding="utf-8")

        print(f"Saved all .kt files to '{self.originals_root}'.")

    def process(self) -> None:
        """
        Full pipeline: load dataset and save Kotlin source files.
        """
        self.originals_root.mkdir(parents=True, exist_ok=True)
        kotlin_dataset = self.load_dataset()
        self.save_kotlin_sources(kotlin_dataset)
