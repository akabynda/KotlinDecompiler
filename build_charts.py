from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def list_metrics(columns: Iterable[str]) -> list[str]:
    skip = {"Test", "Category"}
    return [c for c in columns if c not in skip]


def build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    cats = df["Category"].unique()
    metrics = list_metrics(df.columns)
    data = {
        "Category": [],
        **{m: [] for m in metrics},
    }
    for cat in cats:
        sub = df.loc[df["Category"] == cat, metrics]
        data["Category"].append(cat)
        for m in metrics:
            data[m].append(sub[m].mean())
    return pd.DataFrame(data).set_index("Category")


def save_charts(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in summary.columns:
        plt.figure(figsize=(10, 4))
        summary[metric].plot(kind="bar")
        plt.title(metric)
        plt.ylabel("Average value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric.replace(" ", "_")}_by_category.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    df = pd.read_csv(input("Path to full_metrics.csv:"))
    summary = build_category_summary(df)
    charts_dir = Path("charts")
    charts_dir.mkdir(parents=True, exist_ok=True)
    (charts_dir / "summary.csv").write_text(summary.to_csv())
    save_charts(summary, charts_dir)
    print(f"Charts and summary saved to {charts_dir.absolute()}")
