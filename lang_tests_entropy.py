import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List

from entropy import Entropy
from utils.aggregate import aggregate
from utils.detect_category import detect_category

language = "kotlin"
entr = Entropy(language)
language_model_name = "kstack-clean+kexercises"

LANG_MODEL_DIR = Path(f"lang_models/{language}/{language_model_name}")
with open(LANG_MODEL_DIR / "unigram.json", encoding="utf8") as f:
    P_uni = json.load(f)
with open(LANG_MODEL_DIR / "bigram.json", encoding="utf8") as f:
    P_bi = json.load(f)
with open(LANG_MODEL_DIR / "left.json", encoding="utf8") as f:
    P_left = json.load(f)

_ENT_FUNCS: dict[str, Callable[[str], float]] = {
    "CE": lambda s: entr.cross_entropy_lang(P_uni, s),
    "KL": lambda s: entr.kl_div_lang(P_uni, s),
    "PPL": lambda s: entr.perplexity_lang(P_uni, s),
    "NID": lambda s: entr.nid_lang(P_uni, s),
    "Hcond": lambda s: entr.conditional_entropy_lang(P_bi, P_left, s)
}


def collect_entropy_metrics(files_in_test: list[Path],
                            base_dir: Path) -> dict[str, dict[str, float]]:
    tmp: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for p in files_in_test:
        cat = detect_category(base_dir, p)
        if cat is None:
            continue

        src = p.read_text(encoding="utf8")

        for name, fn in _ENT_FUNCS.items():
            tmp[cat][name].append(float(fn(src)))

    return {cat: {m: mean(vals) for m, vals in m_dict.items()}
            for cat, m_dict in tmp.items()}


if __name__ == "__main__":
    folder = Path(input("Введите путь к папке с тестами: ").strip()).expanduser()
    aggregate(folder, f"{language_model_name}_entropy_metrics_category_summary", collect_entropy_metrics)
