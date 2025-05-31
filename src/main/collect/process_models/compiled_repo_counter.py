import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


class CompiledRepoCounter:
    """
    Analyzes and summarizes successful Kotlin bytecode compilation across dataset fields.
    """

    def __init__(self, dataset_root: Path) -> None:
        """
        Initialize the analyzer.

        Args:
            dataset_root (Path): Path to the dataset containing fields with 'originals' directories.
        """
        self.dataset_root: Path = dataset_root
        self.compiled_info: Dict[str, List[str]] = {}
        self.results: List[Tuple[str, int]] = []

    def count(self) -> None:
        """
        Analyze all fields in the dataset and save results.
        """
        fields: List[str] = [
            p.name
            for p in self.dataset_root.iterdir()
            if p.is_dir() and (p / "originals").is_dir()
        ]

        for field in fields:
            orig_root = self.dataset_root / field / "originals"
            bytecode_root = self.dataset_root / field / "bytecode"

            compiled_paths: List[str] = []
            if bytecode_root.exists():
                repo_dirs = [d for d in bytecode_root.iterdir() if d.is_dir()]
                for repo in repo_dirs:
                    orig_repo_dir = orig_root / repo.name
                    if orig_repo_dir.exists():
                        for kt_file in orig_repo_dir.rglob("*.kt"):
                            rel = kt_file.relative_to(orig_root)
                            compiled_paths.append(rel.as_posix())
                count = len(repo_dirs)
            else:
                count = 0

            self.results.append((field, count))
            self.compiled_info[field] = compiled_paths

        self.results.sort(key=lambda x: x[1], reverse=True)

    def save_results(self) -> None:
        """
        Save analysis results as CSV, JSON, and a bar plot.
        """
        counts_csv = self.dataset_root / "bytecode_repo_counts.csv"
        with counts_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["field", "repo_count"])
            writer.writerows(self.results)
        print(f"Repository counts saved to {counts_csv}")

        compiled_json = self.dataset_root / "compiled_repos.json"
        with compiled_json.open("w", encoding="utf-8") as f:
            json.dump(self.compiled_info, f, ensure_ascii=False, indent=2)
        print(f"Compiled .kt file lists saved to {compiled_json}")

        self._save_plot()

    def _save_plot(self) -> None:
        """
        Generate and save a bar plot of repository counts.
        """
        fields_sorted: List[str] = [
            f.replace("KExercises-KStack-clean-bytecode-4bit-lora", "Finetuned")
            for f, _ in self.results
        ]
        counts_sorted: List[int] = [count for _, count in self.results]

        plt.figure(figsize=(12, 6))
        plt.bar(fields_sorted, counts_sorted)
        plt.xticks(rotation=45, ha="right")
        plt.title("Compiled Repositories per Model")
        plt.xlabel("Model")
        plt.ylabel("Count")
        plt.tight_layout()

        plot_path = self.dataset_root / "bytecode_repo_counts.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")


def main() -> None:
    """
    Entry point for the bytecode compilation analysis.
    """
    dataset_path = Path(input("Path to dataset: ").strip())
    counter = CompiledRepoCounter(dataset_path)
    counter.count()
    counter.save_results()


if __name__ == "__main__":
    main()
