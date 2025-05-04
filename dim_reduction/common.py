from typing import List

import numpy as np
import pandas as pd


def numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number").copy()
    num.drop(columns=[c for c in ("Test", "Category") if c in num.columns],
             inplace=True, errors="ignore")
    const_cols = num.columns[num.var(ddof=0) == 0].tolist()
    if const_cols:
        print("Удалено константных столбцов:", const_cols)
        num.drop(columns=const_cols, inplace=True)
    return num


def drop_high_corr(df: pd.DataFrame, thresh: float = 0.995,
                   removed: List[str] | None = None) -> pd.DataFrame:
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones_like(corr), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > thresh)]
    if to_drop:
        # print(f"Удалено {len(to_drop)} высоко коррелирующих столбцов:", to_drop)
        if removed is not None:
            removed.extend(to_drop)
        df = df.drop(columns=to_drop)
    return df


def safe_corr_df(df: pd.DataFrame, removed: List[str]) -> pd.DataFrame:
    thresh = 0.99
    while thresh >= 0.80:
        tmp = drop_high_corr(df, thresh, removed)
        if tmp.shape[1] < 3:
            break
        if np.linalg.matrix_rank(np.corrcoef(tmp.T)) == tmp.shape[1]:
            return tmp
        thresh -= 0.02
        print(f"пробуем thresh={thresh:.2f}")
    raise RuntimeError("Не удалось получить невырожденную корреляционную матрицу")
