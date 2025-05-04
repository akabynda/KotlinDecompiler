from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FULL = Path("../full_metrics.csv")
TARGETS = ["CE", "KL", "PPL", "NID", "Hcond"]
CORR_DROP = .95


def prepared_data(corr_drop: float = CORR_DROP) -> tuple[pd.DataFrame,
list[str],
np.ndarray]:
    df = pd.read_csv(FULL)
    df = df[df["Category"] != "Original"]

    struct0 = [c for c in df.columns
               if c not in ("Test", "Category")
               and c not in TARGETS]

    corr = df[struct0].corr().abs()
    keep, dropped = [], set()
    for col in corr.columns:
        if col in dropped:
            continue
        keep.append(col)
        dropped.update(corr.index[corr[col] > corr_drop])

    Xstd = StandardScaler().fit_transform(df[keep])
    return df, keep, Xstd


def multiple_r(y: np.ndarray, X: np.ndarray) -> float:
    β, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(abs(np.corrcoef(y, X @ β)[0, 1]))


def flush(path: Path,
          buffer: list[tuple[str, float]],
          header: bool) -> bool:
    if buffer:
        pd.DataFrame(buffer, columns=["Predictors", "R"]).to_csv(
            path, mode="a" if not header else "w",
            header=header, index=False
        )
        buffer.clear()
    return False


def comb_cnt(n: int, k_max: int) -> int:
    return sum(math.comb(n, k) for k in range(1, k_max + 1))


def mask_from_cols(cols: tuple[int, ...]) -> int:
    """bit-mask combo → int (до 64 признаков ok)"""
    m = 0
    for c in cols:
        m |= 1 << c
    return m
