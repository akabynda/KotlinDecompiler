from statistics import median
from typing import Iterable

from numpy import percentile

from collect.process_models.shared import Row


def gen_len_stats(rows: Iterable[Row], tokenizer) -> tuple[int, float]:
    kt_lens, ratios = [], []
    for r in rows:
        kt = len(tokenizer(r.kt_source).input_ids)
        bc = len(tokenizer(r.bytecode).input_ids)
        kt_lens.append(kt)
        ratios.append(kt / bc if bc else 0)

    return min(2048, int(percentile(kt_lens, 90) * 1.5)), round(min(0.5, median(ratios)), 3)