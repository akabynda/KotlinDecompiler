from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from mcorr_common import prepared_data, multiple_r, flush, TARGETS

OUTDIR = Path("mcorr_stepwise")
OUTDIR.mkdir(exist_ok=True)
FLUSH_EVERY = 100000

df, struct, X_all = prepared_data(corr_drop=1.00)
n_struct = len(struct)


def bidirectional_history(y: np.ndarray,
                          pool: list[int]) -> list[tuple[list[int], float]]:
    sel: list[int] = []
    cur = 0.0
    hist: list[tuple[list[int], float]] = []
    while pool:
        # forward
        best, best_r = None, cur
        for v in pool:
            r = multiple_r(y, X_all[:, sel + [v]])
            if r > best_r + 1e-4:
                best_r, best = r, v
        if best is None:
            break
        sel.append(best)
        pool.remove(best)
        cur = best_r
        # backward
        changed = True
        while changed and len(sel) > 1:
            changed = False
            for v in sel[:-1]:
                trial = sel.copy()
                trial.remove(v)
                r = multiple_r(y, X_all[:, trial])
                if r > cur + 1e-4:
                    sel, cur, changed = trial, r, True
                    pool.append(v)
                    break
        hist.append((sel.copy(), cur))
    return hist


def worker(target: str) -> None:
    out = OUTDIR / f"{target}.csv"
    done = set(pd.read_csv(out)["Predictors"]) if out.exists() else set()
    first = not out.exists()

    y = np.squeeze(df[[target]].values)
    y = (y - y.mean()) / y.std(ddof=0)

    r_uni = np.abs(np.corrcoef(X_all, y, rowvar=False)[-1, :-1])
    pool_idx = list(np.argsort(r_uni)[::-1])

    buf: list[tuple[str, float]] = []
    for sel, r in bidirectional_history(y, pool_idx.copy()):
        preds = ", ".join(struct[i] for i in sel)
        if preds in done:
            continue
        buf.append((preds, r))
        done.add(preds)
        if len(buf) >= FLUSH_EVERY:
            first = flush(out, buf, first)
    flush(out, buf, first)


if __name__ == "__main__":
    with mp.Pool() as p:
        p.map(worker, TARGETS)
