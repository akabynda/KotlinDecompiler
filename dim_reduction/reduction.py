from pathlib import Path

import pandas as pd

from dim_reduction.efa import efa
from dim_reduction.pca import pca


def combine_features(
        df: pd.DataFrame,
        *,
        pca_kwargs: dict = {},
        minres_efa_kwargs: dict = {},
        ml_efa_kwargs: dict = {},
) -> list[str]:
    _, _, _, recommended_pca = pca(df, **pca_kwargs)
    _, _, _, recommended_minres_efa = efa(df, **minres_efa_kwargs)
    _, _, _, recommended_ml_efa = efa(df, **ml_efa_kwargs)

    return sorted(list(set(recommended_pca + recommended_minres_efa + recommended_ml_efa)))


if __name__ == "__main__":
    df = pd.read_csv(Path(input("Path to full_metrics.csv:")))

    pca_args = {
        "scale": True,
        "n_components": 0.8,
        "top_k": 3,
        "recommend": "per_pc",
    }

    ml_efa_args = {
        "scale": True,
        "n_factors": "auto",
        "rotation": "promax",
        "method": "ml",
        "top_k": 3,
        "recommend": "per_factor",
    }

    minres_efa_args = {
        "scale": True,
        "n_factors": "auto",
        "rotation": "varimax",
        "method": "minres",
        "top_k": 3,
        "recommend": "per_factor",
    }

    combined = combine_features(df, pca_kwargs=pca_args, minres_efa_kwargs=minres_efa_args, ml_efa_kwargs=ml_efa_args)
    print("Объединённый список признаков:")
    for feat in combined:
        print(" -", feat)
