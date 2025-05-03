from pathlib import Path
from statistics import mean
from typing import Dict, List

from matplotlib import pyplot as plt


def build_charts(summary: Dict[str, Dict[str, List[float]]],
                 out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[str] = []
    for m_dict in summary.values():
        for m in m_dict:
            if m not in all_metrics:
                all_metrics.append(m)

    for metric in all_metrics:
        cats, vals = [], []
        for cat, m_dict in summary.items():
            if metric in m_dict and m_dict[metric]:
                cats.append(cat)
                vals.append(mean(m_dict[metric]))

        if not cats:
            continue

        plt.figure(figsize=(8, 4))
        plt.bar(cats, vals)
        plt.xticks(rotation=40, ha='right')
        plt.title(metric)
        plt.ylabel("Average value")
        plt.tight_layout()

        png_path = out_dir / f"{metric.replace(' ', '')}_by_category.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Chart saved: {png_path}")
