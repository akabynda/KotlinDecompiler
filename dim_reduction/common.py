from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler


def numeric_df(df_in: pd.DataFrame, removed: List[str]) -> pd.DataFrame:
    df_out = df_in.select_dtypes(include="number").copy()
    drop_cols = [col for col in ("Test", "Category") if col in df_out.columns]
    df_out.drop(columns=drop_cols, inplace=True, errors="ignore")
    removed.extend(drop_cols)

    const_cols = df_out.columns[df_out.var(ddof=0) == 0].tolist()
    if const_cols:
        removed.extend(const_cols)
        df_out.drop(columns=const_cols, inplace=True)

    print("Удалены константные признаки", const_cols)
    return df_out


def drop_low_variance(df_in: pd.DataFrame,
                      removed: List[str],
                      threshold: float = 0.01) -> pd.DataFrame:
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df_in)
    scaled_df = pd.DataFrame(scaled_array, columns=df_in.columns, index=df_in.index)

    sel = VarianceThreshold(threshold)
    sel.fit_transform(scaled_df)
    mask = sel.get_support()
    selected_cols = df_in.columns[mask]
    dropped_cols = df_in.columns[np.logical_not(mask)].tolist()

    if dropped_cols:
        removed.extend(dropped_cols)

    df_out = df_in[selected_cols]

    print("Удалены признаки с низкой дисперсией:", dropped_cols)
    return df_out


def drop_high_corr(df_in: pd.DataFrame, removed: List[str], thresh: float = 0.995) -> pd.DataFrame:
    df_out = df_in.copy()
    corr = df_out.corr().abs()
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
        removed.extend(to_drop)
        df_out.drop(columns=list(to_drop), inplace=True)
    return df_out


def drop_high_corr_interactive(df_in: pd.DataFrame, removed: List[str], thresh: float = 0.995) -> pd.DataFrame:
    df_out = df_in.copy()
    corr = df_out.corr().abs()
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
        removed.extend(to_drop)
        df_out.drop(columns=list(to_drop), inplace=True)
    return df_out


def safe_corr_df(df_in: pd.DataFrame, removed: List[str]) -> pd.DataFrame:
    df_out = df_in.copy()
    thresh = 0.99
    while thresh >= 0.80:
        print(f"\n=== Порог корреляции: {thresh:.2f} ===")
        df_out = drop_high_corr(df_out, removed, thresh)
        if df_out.shape[1] < 3:
            print("Осталось менее 3 признаков — останавливаемся.")
            break
        if np.linalg.matrix_rank(np.corrcoef(df_out.T)) == df_out.shape[1]:
            print("Корреляционная матрица невырождена, готово.")
            return df_out
        thresh -= 0.02
    raise RuntimeError("Не удалось получить невырожденную корреляционную матрицу")


def safe_corr_df_interactive(df_in: pd.DataFrame, removed: List[str]) -> pd.DataFrame:
    df_out = df_in.copy()
    thresh = 0.99
    while thresh >= 0.80:
        print(f"\n=== Порог корреляции: {thresh:.2f} ===")
        df_out = drop_high_corr_interactive(df_out, removed, thresh)
        if df_out.shape[1] < 3:
            print("Осталось менее 3 признаков — останавливаемся.")
            break
        if np.linalg.matrix_rank(np.corrcoef(df_out.T)) == df_out.shape[1]:
            print("Корреляционная матрица невырождена, готово.")
            return df_out
        thresh -= 0.02
    raise RuntimeError("Не удалось получить невырожденную корреляционную матрицу")
