from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.global_config import FEATURES


class MetricVisualizer:
    """
    Generates category-based metric summaries and visualizations.
    """

    def __init__(self, metrics_path: Path, out_dir: Path) -> None:
        """
        Initialize the visualizer with paths and basic settings.

        Args:
            metrics_path (Path): Path to the CSV file with metrics data.
            out_dir (Path): Directory to save charts and summary files.
        """
        self.metrics_path: Path = metrics_path
        self.out_dir: Path = out_dir
        self.skip_columns: List[str] = ["kt_path", "model"]
        self.reference_model: str = "kt_source"
        self.dataframe: pd.DataFrame = self._load_data()
        self.metrics: List[str] = self._list_metrics()

    def _load_data(self) -> pd.DataFrame:
        """
        Load the metrics CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        print(f"Loading data from {self.metrics_path} ...")
        return pd.read_csv(self.metrics_path)

    def _list_metrics(self) -> List[str]:
        """
        Get a list of metric columns excluding skip columns.

        Returns:
            List[str]: List of metric column names.
        """
        return [c for c in self.dataframe.columns if c not in self.skip_columns]

    def build_category_summary(self) -> pd.DataFrame:
        """
        Build a summary DataFrame with average metric values for each category.

        Returns:
            pd.DataFrame: Summary DataFrame.
        """
        categories = self.dataframe[self.skip_columns[1]].unique()
        data = {
            self.skip_columns[1]: [],
            **{m: [] for m in self.metrics},
        }

        for cat in categories:
            sub_df = self.dataframe.loc[
                self.dataframe[self.skip_columns[1]] == cat, self.metrics
            ]
            data[self.skip_columns[1]].append(cat)
            for metric in self.metrics:
                data[metric].append(sub_df[metric].mean())

        summary_df = pd.DataFrame(data).set_index(self.skip_columns[1])
        return summary_df

    def save_charts(self, summary: pd.DataFrame) -> None:
        """
        Save bar charts of metrics by category.

        Args:
            summary (pd.DataFrame): Summary DataFrame.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        for metric in summary.columns:
            if self.reference_model not in summary.index:
                print(
                    f"Reference '{self.reference_model}' not found for metric '{metric}', skipping..."
                )
                continue

            ref_value = summary.loc[self.reference_model, metric]
            sorted_summary = summary.copy()
            sorted_summary["distance"] = (sorted_summary[metric] - ref_value).abs()
            sorted_summary = sorted_summary.sort_values("distance").drop(
                columns="distance"
            )

            plt.figure(figsize=(10, 4))
            sorted_summary[metric].plot(kind="bar")
            plt.title(metric)
            plt.ylabel("Average value")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(
                self.out_dir / f"{metric.replace(' ', '_')}_by_category.png", dpi=150
            )
            plt.close()

    def save_heatmap(self, summary: pd.DataFrame) -> None:
        """
        Save a heatmap of selected metrics.

        Args:
            summary (pd.DataFrame): Summary DataFrame.
        """
        selected_metrics = [m for m in FEATURES if m in summary.columns]
        if not selected_metrics:
            print("No selected metrics found in summary, skipping heatmap.")
            return

        subset = summary[selected_metrics]

        rename_dict = {
            "Conditional Complexity": "CondComp",
            "Halstead Distinct Operators": "HalstDO",
        }
        subset = subset.rename(columns=rename_dict)

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            subset, annot=True, cmap="YlGnBu", cbar_kws={"label": "Average value"}
        )
        plt.title("Heatmap of Selected Metrics by Model")
        plt.ylabel("Model")
        plt.xlabel("Metric")
        plt.tight_layout()

        heatmap_path = self.out_dir / "heatmap_selected_metrics.png"
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f"Heatmap saved to {heatmap_path.absolute()}")

    def run(self) -> None:
        """
        Full pipeline: build summary, save charts and heatmap.
        """
        summary = self.build_category_summary()
        summary.index = [
            i.replace("KExercises-KStack-clean-bytecode-4bit-lora", "Finetuned")
            for i in summary.index
        ]

        (self.out_dir / "summary.csv").write_text(summary.to_csv())
        self.save_charts(summary)
        self.save_heatmap(summary)
        print(f"Charts and summary saved to {self.out_dir.absolute()}")


def main() -> None:
    """
    Entry point for generating metric charts and heatmaps.
    """
    metrics_path = Path(input("Path to metrics.csv: "))
    out_dir = Path(input("Path to charts dir: "))
    visualizer = MetricVisualizer(metrics_path, out_dir)
    visualizer.run()


if __name__ == "__main__":
    main()
