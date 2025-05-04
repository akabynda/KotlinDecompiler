from pathlib import Path

import pandas as pd

from dim_reduction.efa import efa
from dim_reduction.pca import pca


def combine_features(
        df: pd.DataFrame,
        *,
        pca_kwargs: dict = {},
        efa_kwargs: dict = {},
) -> list[str]:
    _, _, _, recommended_pca = pca(df, **pca_kwargs)
    _, _, _, recommended_efa = efa(df, **efa_kwargs)

    return sorted(list(set(recommended_pca + recommended_efa)))


if __name__ == "__main__":
    df = pd.read_csv(Path(input("Path to metrics.csv:")))

    pca_args = {
        "scale": True,
        "n_components": 0.8,
        "top_k": 3,
        "recommend": "per_pc",
    }

    efa_args = {
        "scale": True,
        "n_factors": "auto",
        "rotation": "promax",
        "method": "ml",
        "top_k": 3,
        "recommend": "per_factor",
        "kmo_warn": 0.6
    }

    combined = combine_features(df, pca_kwargs=pca_args, efa_kwargs=efa_args)
    print("Объединённый список признаков:")
    for feat in combined:
        print(" -", feat)
