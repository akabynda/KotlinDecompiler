from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Callable

import matplotlib.pyplot as plt

from entropy.common import (cross_entropy, kl_div, perplexity,
                            nid, conditional_entropy, cross_entropy_ngram)

_ENT_FUNCS: dict[str, Callable[[str, str], float]] = {
    "CE": cross_entropy,
    "KL": kl_div,
    "PPL": perplexity,
    "NID": nid,
    "Hcond": conditional_entropy,
    **{f"CE_{n}gram": (lambda p, q, n=n: cross_entropy_ngram(p, q, n))
       for n in range(1, 7)}
}


_DECOMPILERS = {"Bytecode", "CFR", "Fernflower", "JDGUI"}
_CONVERTERS = {"ChatGPT", "J2K"}


def detect_category(base_test_dir: Path, file_path: Path) -> str | None:
    rel_parts = file_path.relative_to(base_test_dir).parts
    if len(rel_parts) == 2:
        return "Original"

    decompiler = rel_parts[1]
    if decompiler not in _DECOMPILERS:
        return None

    if decompiler == "Bytecode":
        return "BytecodeChatGPT"

    converter = None
    for part in rel_parts[2:]:
        for conv in _CONVERTERS:
            if conv.lower() in part.lower():
                converter = conv
                break
        if converter:
            break

    return f"{decompiler}{converter}" if converter else None


def collect_entropy_metrics(files_in_test: list[Path],
                            base_dir: Path) -> dict[str, dict[str, float]]:
    originals: dict[str, str] = {}
    for p in files_in_test:
        if detect_category(base_dir, p) == "Original":
            originals[p.name] = p.read_text(encoding="utf8")

    tmp: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for p in files_in_test:
        cat = detect_category(base_dir, p)
        if cat is None or cat == "Original":
            continue

        orig_src = originals.get(p.name)
        if orig_src is None:
            continue

        dec_src = p.read_text(encoding="utf8")

        for name, fn in _ENT_FUNCS.items():
            val = fn(orig_src, dec_src)
            tmp[cat][name].append(float(val))

    return {cat: {m: mean(vals) for m, vals in m_dict.items()} for cat, m_dict in tmp.items()}


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
        print(f"📊Chart saved: {png_path}")


def aggregate(base_dir: Path):
    from utils.csv_utils import write_category_summary

    data = defaultdict(lambda: defaultdict(list))

    for test_dir in base_dir.iterdir():
        if not test_dir.is_dir():
            continue

        test_files = list(test_dir.rglob("*.kt"))
        if not test_files:
            continue

        test_metrics = collect_entropy_metrics(test_files, base_dir)

        for category, m_dict in test_metrics.items():
            for name, value in m_dict.items():
                data[category][name].append(value)

            print(f"{category:>16} | {test_dir.name}/...  → {m_dict}")

    summary_csv = base_dir / "entropy_metrics_category_summary.csv"
    write_category_summary(summary_csv, data)
    print(f"\nCSV-сводка: {summary_csv}")

    build_charts(data, base_dir / "entropy_charts")


if __name__ == "__main__":
    folder = Path(input("Введите путь к папке с тестами: ").strip()).expanduser()
    aggregate(folder)
