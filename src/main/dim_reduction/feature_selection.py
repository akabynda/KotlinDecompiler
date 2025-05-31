from pathlib import Path
from typing import List, Set, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler


class FeatureSelector:
    """
    Provides methods to clean numeric data by removing low variance and highly correlated features.
    """

    @staticmethod
    def numeric_df(df_in: pd.DataFrame, removed: List[str]) -> pd.DataFrame:
        """
        Retain only numeric columns and remove unwanted ones.
        """
        df_out = df_in.select_dtypes(include="number").copy()

        drop_cols = [
            col
            for col in ("Test", "Category", "kt_path", "model")
            if col in df_out.columns
        ]
        df_out.drop(columns=drop_cols, inplace=True, errors="ignore")
        removed.extend(drop_cols)

        return df_out

    @staticmethod
    def drop_low_variance(
        df_in: pd.DataFrame, removed: List[str], threshold: float = 0.01
    ) -> pd.DataFrame:
        """
        Drop features with low variance.
        """
        scaler = MinMaxScaler()
        scaled_df = pd.DataFrame(
            scaler.fit_transform(df_in), columns=df_in.columns, index=df_in.index
        )

        sel = VarianceThreshold(threshold)
        mask = sel.fit(scaled_df).get_support()
        dropped_cols = df_in.columns[np.logical_not(mask)].tolist()

        if dropped_cols:
            removed.extend(dropped_cols)

        print("Removed low variance features:", dropped_cols)
        return df_in.loc[:, mask]

    @staticmethod
    def drop_high_corr_recursive(
        df_in: pd.DataFrame,
        upper: pd.DataFrame,
        thresh: float,
        to_drop: Optional[Set[str]] = None,
        path: Optional[List[str]] = None,
        best_result: Optional[Dict] = None,
    ) -> Dict:
        """
        Recursively explore and drop highly correlated features to maximize removal.
        """
        if to_drop is None:
            to_drop = set()
        if path is None:
            path = []
        if best_result is None:
            best_result = {"max_dropped": 0, "to_drop": set()}

        for row in upper.index:
            for col in upper.columns:
                if (
                    upper.at[row, col] > thresh
                    and row not in to_drop
                    and col not in to_drop
                ):
                    new_to_drop_1 = to_drop.copy()
                    new_to_drop_1.add(row)
                    FeatureSelector.drop_high_corr_recursive(
                        df_in,
                        upper,
                        thresh,
                        new_to_drop_1,
                        path + [f"drop {row}"],
                        best_result,
                    )

                    new_to_drop_2 = to_drop.copy()
                    new_to_drop_2.add(col)
                    FeatureSelector.drop_high_corr_recursive(
                        df_in,
                        upper,
                        thresh,
                        new_to_drop_2,
                        path + [f"drop {col}"],
                        best_result,
                    )

                    return best_result

        if len(to_drop) > best_result["max_dropped"]:
            best_result["max_dropped"] = len(to_drop)
            best_result["to_drop"] = to_drop.copy()

        return best_result

    @staticmethod
    def drop_high_corr(
        df_in: pd.DataFrame, removed: List[str], thresh: float = 0.995
    ) -> pd.DataFrame:
        """
        Drop highly correlated features using a recursive search.
        """
        corr = df_in.corr().abs()
        upper = corr.where(np.triu(np.ones_like(corr), k=1).astype(bool))

        best_result = FeatureSelector.drop_high_corr_recursive(df_in, upper, thresh)
        final_to_drop = best_result["to_drop"]

        if final_to_drop:
            removed.extend(final_to_drop)
            df_in.drop(columns=list(final_to_drop), inplace=True)

        return df_in

    @staticmethod
    def safe_corr_df(df_in: pd.DataFrame, removed: List[str]) -> pd.DataFrame:
        """
        Iteratively drop highly correlated features while ensuring matrix invertibility.
        """
        df_out = df_in.copy()
        thresh = 0.99
        while thresh >= 0.80:
            print(f"\n=== Correlation threshold: {thresh:.2f} ===")
            df_out = FeatureSelector.drop_high_corr(df_out, removed, thresh)
            if df_out.shape[1] < 3:
                print("Fewer than 3 features left — stopping.")
                break
            if np.linalg.matrix_rank(np.corrcoef(df_out.T)) == df_out.shape[1]:
                print("Correlation matrix is non-singular, done.")
                return df_out
            thresh -= 0.02

        return df_out


if __name__ == "__main__":
    df = pd.read_csv(Path(input("Path to metrics.csv: ")))

    removed: List[str] = []

    num_df = FeatureSelector.numeric_df(df, removed)
    low_var_df = FeatureSelector.drop_low_variance(num_df, removed)
    reduced_df = FeatureSelector.safe_corr_df(low_var_df, removed)

    if removed:
        print("Final removed features:", len(removed), removed)

    print("Remaining features:", len(reduced_df.columns), reduced_df.columns.tolist())
