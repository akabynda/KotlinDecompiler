from pathlib import Path
from typing import Any, Dict, Iterable

from datasets import load_dataset


class KExercisesDownloader:
    """
    Processes the JetBrains/KExercises dataset and saves Kotlin solutions to individual files.
    """

    def __init__(
            self,
            split: str = "train",
            streaming: bool = True,
            output_dir: Path = Path("./kexercises/originals"),
    ) -> None:
        """
        Initialize the processor.

        Args:
            split (str): Dataset split to use (default: "train").
            streaming (bool): Whether to stream the dataset (default: True).
            output_dir (Path): Directory to save extracted solutions (default: "./kexercises/originals").
        """
        self.split: str = split
        self.streaming: bool = streaming
        self.output_dir: Path = output_dir

    def load_dataset(self) -> Iterable[Dict[str, Any]]:
        """
        Load the dataset.

        Yields:
            dict: Each example from the dataset.
        """
        dataset = load_dataset(
            "JetBrains/KExercises",
            split=self.split,
            streaming=self.streaming
        )
        for example in dataset:
            yield example

    def save_exercises(self, dataset: Iterable[Dict[str, Any]]) -> None:
        """
        Save each example as a separate Kotlin file.

        Args:
            dataset (Iterable[dict]): Dataset to process.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for idx, example in enumerate(dataset):
            folder = self.output_dir / f"{idx}"
            folder.mkdir(parents=True, exist_ok=True)

            file_path = folder / f"solution_{idx}.kt"
            problem = example.get("problem", "").strip()
            solution = example.get("solution", "").strip()
            combined_code = f"{problem}\n{solution}"

            file_path.write_text(combined_code, encoding="utf-8")

    def process(self) -> None:
        """
        Full pipeline: load the dataset and save the exercises.
        """
        dataset = self.load_dataset()
        self.save_exercises(dataset)
