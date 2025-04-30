from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

from mcorr_common import prepared_data, multiple_r, flush, TARGETS

OUTDIR = Path("mcorr_lasso")
OUTDIR.mkdir(exist_ok=True)

df, struct, X_all = prepared_data(corr_drop=1.00)


def worker(target: str) -> None:
    out = OUTDIR / f"{target}.csv"
    first = not out.exists()
    done = set(pd.read_csv(out)["Predictors"]) if out.exists() else set()

    y = np.squeeze(df[[target]].values)
    y = (y - y.mean()) / y.std(ddof=0)

    lass = LassoCV(
        cv=5,
        n_jobs=1,
        max_iter=20000,
        tol=1e-3
    ).fit(X_all, y)
    mask = lass.coef_ != 0
    if not mask.any():
        return
    sel_idx = [i for i, m in enumerate(mask) if m]
    preds = ", ".join(struct[i] for i in sel_idx)
    if preds in done:
        return
    r = multiple_r(y, X_all[:, sel_idx])
    flush(out, [(preds, r)], first)


if __name__ == "__main__":
    with mp.Pool() as p:
        p.map(worker, TARGETS)
