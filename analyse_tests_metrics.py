from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List

from metrics import registry
from utils.aggregate import aggregate
from utils.detect_category import detect_category
from utils.kotlin_parser import parse


def analyse_file(file_path: Path) -> Dict[str, Any]:
    tree = parse(file_path.read_text(encoding="utf8"))
    return {name: fn(tree) for name, fn in registry.items()}


def collect_test_metrics(files_in_test, base_dir):
    tmp: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for file_path in files_in_test:
        category = detect_category(base_dir, file_path)
        if category is None:
            continue

        metrics = analyse_file(file_path)
        for name, val in metrics.items():
            tmp[category][name].append(float(val))

    return {
        cat: {m: mean(vals) for m, vals in m_dict.items()}
        for cat, m_dict in tmp.items()
    }


if __name__ == "__main__":
    folder = Path(input("Введите путь к папке с тестами: ").strip()).expanduser()
    aggregate(folder, "metrics_category_summary", collect_test_metrics)
