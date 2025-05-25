from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler


def numeric_df(df_in: pd.DataFrame, removed: List[str]) -> pd.DataFrame:
    df_out = df_in.select_dtypes(include="number").copy()

    drop_cols = [col for col in ("Test", "Category", "kt_path", "model") if col in df_out.columns]
    df_out.drop(columns=drop_cols, inplace=True, errors="ignore")
    removed.extend(drop_cols)

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


def drop_high_corr_recursive(df_in: pd.DataFrame, upper: pd.DataFrame, thresh: float,
                             to_drop: set[str] = None, path: List[str] = None,
                             best_result: dict = None) -> dict:
    if to_drop is None:
        to_drop = set()
    if path is None:
        path = []
    if best_result is None:
        best_result = {'max_dropped': 0, 'to_drop': set()}

    for row in upper.index:
        for col in upper.columns:
            if upper.at[row, col] > thresh and row not in to_drop and col not in to_drop:
                new_to_drop_1 = to_drop.copy()
                new_to_drop_1.add(row)
                drop_high_corr_recursive(df_in, upper, thresh, new_to_drop_1, path + [f"drop {row}"], best_result)

                new_to_drop_2 = to_drop.copy()
                new_to_drop_2.add(col)
                drop_high_corr_recursive(df_in, upper, thresh, new_to_drop_2, path + [f"drop {col}"], best_result)

                return best_result

    if len(to_drop) > best_result['max_dropped']:
        best_result['max_dropped'] = len(to_drop)
        best_result['to_drop'] = to_drop.copy()

    return best_result


def drop_high_corr(df_in: pd.DataFrame, removed: List[str], thresh: float = 0.995) -> pd.DataFrame:
    df_out = df_in.copy()
    corr = df_out.corr().abs()
    upper = corr.where(np.triu(np.ones_like(corr), k=1).astype(bool))

    best_result = drop_high_corr_recursive(df_out, upper, thresh)
    final_to_drop = best_result['to_drop']

    if final_to_drop:
        removed.extend(final_to_drop)
        df_out.drop(columns=list(final_to_drop), inplace=True)

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
