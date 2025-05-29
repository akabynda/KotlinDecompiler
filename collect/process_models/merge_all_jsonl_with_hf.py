import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset

from collect.process_models.shared import Config


class DatasetMerger:
    """
    Merges multiple JSONL files with a base HuggingFace dataset and saves the result.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """
        Initialize the DatasetMerger with a configuration.

        Args:
            config (Optional[Config]): Configuration for dataset details. Defaults to global Config.
        """
        self.config: Config = config if config is not None else Config()

    def merge_with_hf_dataset(self, input_dir: Path, output_file: Path) -> None:
        """
        Merge all JSONL files in the input directory with the HuggingFace dataset.

        Args:
            input_dir (Path): Path to the directory containing JSONL files.
            output_file (Path): Path to save the merged JSONL file.
        """
        print(f"Loading base dataset: {self.config.dataset_name} (split: {self.config.split})")
        base_dataset = load_dataset(self.config.dataset_name, split=self.config.split)
        hf_df = pd.DataFrame(base_dataset)
        merged_df = hf_df.copy()

        for jsonl_file in input_dir.glob("*.jsonl"):
            print(f"Merging {jsonl_file.name}")
            local_df = pd.read_json(jsonl_file, lines=True)

            # Remove duplicate columns except 'kt_path'
            local_df = local_df.drop(
                columns=[
                    col for col in local_df.columns
                    if col in merged_df.columns and col != "kt_path"
                ],
                errors="ignore"
            )

            # Merge local dataframe
            merged_df = pd.merge(
                merged_df,
                local_df,
                on="kt_path",
                how="left",
                suffixes=("", f"_{jsonl_file.stem}")
            )

        # Remove rows with no local information
        local_columns = [
            col for col in merged_df.columns
            if col != "kt_path" and col not in hf_df.columns
        ]
        merged_df = merged_df.dropna(subset=local_columns, how="all")

        # Save to JSONL
        merged_df.to_json(
            output_file,
            orient="records",
            lines=True,
            force_ascii=False
        )
        print(f"Merged dataset saved to: {output_file.resolve()}")


def main() -> None:
    """
    Entry point for merging JSONL files with the base HuggingFace dataset.
    """
    input_dir = Path(input("Enter path to directory with .jsonl files: ").strip())
    output_file = input_dir.with_suffix(".jsonl")

    merger = DatasetMerger()
    merger.merge_with_hf_dataset(input_dir, output_file)


if __name__ == "__main__":
    sys.exit(main())
