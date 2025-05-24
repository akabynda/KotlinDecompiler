from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from charts.build_heatmap import save_heatmap

skip = ["kt_path", "model"]
reference = "kt_source"


# skip = ["Test", "Category"]
# reference = "Original"


def list_metrics(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c not in skip]


def build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    cats = df[skip[1]].unique()
    metrics = list_metrics(df.columns)
    data = {
        skip[1]: [],
        **{m: [] for m in metrics},
    }
    for cat in cats:
        sub = df.loc[df[skip[1]] == cat, metrics]
        data[skip[1]].append(cat)
        for m in metrics:
            data[m].append(sub[m].mean())
    return pd.DataFrame(data).set_index(skip[1])


def save_charts(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in summary.columns:
        if reference not in summary.index:
            print(f"Reference '{reference}' not found for metric '{metric}', skipping...")
            continue
        ref_value = summary.loc[reference, metric]
        sorted_summary = summary.copy()
        sorted_summary["distance"] = (sorted_summary[metric] - ref_value).abs()
        sorted_summary = sorted_summary.sort_values("distance").drop(columns="distance")

        plt.figure(figsize=(10, 4))
        sorted_summary[metric].plot(kind="bar")
        plt.title(metric)
        plt.ylabel("Average value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric.replace(" ", "_")}_by_category.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    df = pd.read_csv(input("Path to metrics.csv: "))
    summary = build_category_summary(df)
    summary.index = [i.replace("KExercises-KStack-clean-bytecode-4bit-lora", "Finetuned") for i in summary.index]
    charts_dir = Path(input("Path to charts dir: "))
    charts_dir.mkdir(parents=True, exist_ok=True)
    (charts_dir / "summary.csv").write_text(summary.to_csv())
    save_charts(summary, charts_dir)
    save_heatmap(summary, charts_dir)
    print(f"Charts and summary saved to {charts_dir.absolute()}")
