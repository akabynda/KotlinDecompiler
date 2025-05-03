from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Callable

from entropy import Entropy
from utils.aggregate import aggregate
from utils.detect_category import detect_category

language = "kotlin"
entr = Entropy(language)

_ENT_FUNCS: dict[str, Callable[[str, str], float]] = {
    "CE": entr.cross_entropy,
    "KL": entr.kl_div,
    "PPL": entr.perplexity,
    "NID": entr.nid,
    "Hcond": entr.conditional_entropy
}


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


if __name__ == "__main__":
    folder = Path(input("Введите путь к папке с тестами: ").strip()).expanduser()
    aggregate(folder, "entropy_metrics_category_summary", collect_entropy_metrics)
