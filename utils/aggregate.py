from collections import defaultdict
from pathlib import Path

from utils.build_charts import build_charts
from utils.csv_utils import write_category_summary


def aggregate(base_dir: Path, csv_name: str, collect):
    data = defaultdict(lambda: defaultdict(list))

    for test_dir in base_dir.iterdir():
        if not test_dir.is_dir():
            continue

        test_files = list(test_dir.rglob("*.kt"))
        if not test_files:
            continue

        test_metrics = collect(test_files, base_dir)

        for category, m_dict in test_metrics.items():
            for name, value in m_dict.items():
                data[category][name].append(value)

            print(f"{category:>16} | {test_dir.name}/...  → {m_dict}")

    summary_csv = base_dir / f"{csv_name}.csv"
    write_category_summary(summary_csv, data)
    print(f"\nCSV‑сводка: {summary_csv}")

    build_charts(data, base_dir / "charts")
