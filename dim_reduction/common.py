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


def drop_high_corr(df: pd.DataFrame,
                   thresh: float = 0.995,
                   removed: list[str] | None = None) -> pd.DataFrame:
    df_copy = df.copy()
    corr = df_copy.corr().abs()
    upper = corr.where(np.triu(np.ones_like(corr), k=1).astype(bool))
    to_drop: set[str] = set()

    for col in upper.columns:
        for row in upper.index:
            if row in to_drop or col in to_drop:
                continue
            if upper.at[row, col] > thresh:
                if "detekt" in row:
                    to_drop.add(row)
                else:
                    to_drop.add(col)
    if to_drop:
        if removed is not None:
            removed.extend(to_drop)
        df_copy = df_copy.drop(columns=list(to_drop))
    return df_copy


def drop_high_corr_interactive(df: pd.DataFrame,
                               thresh: float = 0.995,
                               removed: list[str] | None = None) -> pd.DataFrame:
    df_copy = df.copy()
    corr = df_copy.corr().abs()
    upper = corr.where(np.triu(np.ones_like(corr), k=1).astype(bool))
    to_drop: set[str] = set()

    for col in upper.columns:
        for row in upper.index:
            if row in to_drop or col in to_drop:
                continue
            if upper.at[row, col] > thresh:
                print(f"\nСильная корреляция: {row} vs {col} (коэф = {upper.at[row, col]:.3f})")
                print("Какой признак вы хотите удалить?")
                print(f"1 — удалить {row}")
                print(f"2 — удалить {col}")
                while True:
                    choice = input("Ваш выбор (1/2): ")
                    if choice == "1":
                        to_drop.add(row)
                        break
                    elif choice == "2":
                        to_drop.add(col)
                        break
                    else:
                        print("Некорректный ввод, попробуйте снова.")

    if to_drop:
        if removed is not None:
            removed.extend(to_drop)
        df_copy = df_copy.drop(columns=list(to_drop))
    return df_copy


def safe_corr_df_interactive(df: pd.DataFrame, removed: List[str]) -> pd.DataFrame:
    thresh = 0.99
    tmp_df = df.copy()
    while thresh >= 0.80:
        print(f"\n=== Порог корреляции: {thresh:.2f} ===")
        tmp_df = drop_high_corr_interactive(tmp_df, thresh, removed)
        if tmp_df.shape[1] < 3:
            print("Осталось менее 3 признаков — останавливаемся.")
            break
        if np.linalg.matrix_rank(np.corrcoef(tmp_df.T)) == tmp_df.shape[1]:
            print("Корреляционная матрица невырождена, готово.")
            return tmp_df
        thresh -= 0.02
    raise RuntimeError("Не удалось получить невырожденную корреляционную матрицу")


def safe_corr_df(df: pd.DataFrame, removed: List[str]) -> pd.DataFrame:
    thresh = 0.99
    tmp_df = df.copy()
    while thresh >= 0.80:
        print(f"\n=== Порог корреляции: {thresh:.2f} ===")
        tmp_df = drop_high_corr(tmp_df, thresh, removed)
        if tmp_df.shape[1] < 3:
            print("Осталось менее 3 признаков — останавливаемся.")
            break
        if np.linalg.matrix_rank(np.corrcoef(tmp_df.T)) == tmp_df.shape[1]:
            print("Корреляционная матрица невырождена, готово.")
            return tmp_df
        thresh -= 0.02
    raise RuntimeError("Не удалось получить невырожденную корреляционную матрицу")
