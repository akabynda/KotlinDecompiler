import json
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from global_config import GLOBAL_SEED


class DatasetMerger:
    """
    Merges and shuffles two datasets, then splits them into train/test
    and saves them as JSON files.
    """

    def __init__(
            self,
            dataset1_name: str,
            dataset2_name: str,
            output_dir: Path,
            test_size: float = 0.1,
            seed: int = GLOBAL_SEED,
    ) -> None:
        """
        Initialize the DatasetMerger.

        Args:
            dataset1_name (str): HuggingFace dataset name for the first dataset.
            dataset2_name (str): HuggingFace dataset name for the second dataset.
            output_dir (Path): Directory to save the output JSON files.
            test_size (float): Fraction of data to use as test set. Default is 0.1.
            seed (int): Seed for shuffling. Default is GLOBAL_SEED.
        """
        self.dataset1_name: str = dataset1_name
        self.dataset2_name: str = dataset2_name
        self.output_dir: Path = output_dir
        self.test_size: float = test_size
        self.seed: int = seed

    def load_and_merge(self) -> Dataset:
        """
        Load and merge the two datasets.

        Returns:
            Dataset: Merged dataset.
        """
        print("Loading datasets...")
        dataset1 = load_dataset(self.dataset1_name, split="train")
        dataset2 = load_dataset(self.dataset2_name, split="train")
        merged_dataset = concatenate_datasets([dataset1, dataset2])
        print(f"Merged dataset with {len(merged_dataset)} examples.")
        return merged_dataset

    def shuffle_and_split(self, dataset: Dataset) -> DatasetDict:
        """
        Shuffle and split the dataset into train/test.

        Args:
            dataset (Dataset): Merged dataset.

        Returns:
            DatasetDict: Train/test split dataset.
        """
        print("Shuffling and splitting dataset...")
        shuffled = dataset.shuffle(seed=self.seed)
        split_dataset = shuffled.train_test_split(test_size=self.test_size)
        print("Dataset split complete.")
        return split_dataset

    def save_to_json(self, dataset: Dataset, file_path: Path) -> None:
        """
        Save a dataset to a JSON file.

        Args:
            dataset (Dataset): Dataset to save.
            file_path (Path): Path to output JSON file.
        """
        print(f"Saving {file_path.name} with {len(dataset)} examples...")
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(dataset.to_list(), f, ensure_ascii=False, indent=2)
        print(f"{file_path.name} saved.")

    def process(self) -> None:
        """
        Full pipeline: load, merge, shuffle, split, and save datasets.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        merged_dataset = self.load_and_merge()
        split_dataset = self.shuffle_and_split(merged_dataset)

        train_path = self.output_dir / "train.json"
        test_path = self.output_dir / "test.json"

        self.save_to_json(split_dataset["train"], train_path)
        self.save_to_json(split_dataset["test"], test_path)

        print(f"Combined dataset saved to '{self.output_dir}' with train/test splits.")


def main() -> None:
    """
    Entry point for dataset merging and splitting.
    """
    output_dir = Path("KExercises-KStack-clean-bytecode")
    merger = DatasetMerger(
        dataset1_name="akabynda/KExercises-bytecode",
        dataset2_name="akabynda/KStack-clean-bytecode",
        output_dir=output_dir,
    )
    merger.process()


if __name__ == "__main__":
    main()
