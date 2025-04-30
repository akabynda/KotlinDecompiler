from __future__ import annotations

import itertools
import multiprocessing as mp
from pathlib import Path

import pandas as pd

from mcorr_common import prepared_data, multiple_r, flush, mask_from_cols, comb_cnt, TARGETS

OUT_DIR = Path("mcorr_vif")
OUT_DIR.mkdir(exist_ok=True)

MAX_PREDICTORS = 10
FLUSH_EVERY = 100000

df, struct, X_std = prepared_data()
n_struct = len(struct)


def worker(target: str) -> None:
    out = OUT_DIR / f"{target}.csv"

    if out.exists() and out.stat().st_size:
        old = pd.read_csv(out, usecols=[0])["Predictors"]
        already: set[int] = {
            mask_from_cols(tuple(struct.index(s.strip())
                                 for s in pred.split(", ")))
            for pred in old
        }
        first_write = False
    else:
        already = set()
        first_write = True

    y = (df[target] - df[target].mean()) / df[target].std(ddof=0)

    total = comb_cnt(n_struct, MAX_PREDICTORS)
    processed = len(already)
    buf: list[tuple[str, float]] = []

    for k in range(1, MAX_PREDICTORS + 1):
        for cols in itertools.combinations(range(n_struct), k):
            m = mask_from_cols(cols)
            if m in already:
                continue

            X = X_std[:, cols]
            r = multiple_r(y, X)

            predictors = ", ".join(struct[i] for i in cols)
            buf.append((predictors, r))
            processed += 1

            if len(buf) >= FLUSH_EVERY:
                first_write = flush(out, buf, first_write)
                pct = processed / total * 100
                print(f"{target:<6s}: {processed:,}/{total:,}  {pct:5.2f}%")

    flush(out, buf, first_write)
    print(f"{target:<6s}: done ({processed:,} combos)")


if __name__ == "__main__":
    with mp.Pool() as p:
        p.map(worker, TARGETS)
