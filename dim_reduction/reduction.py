from pathlib import Path

import pandas as pd

from dim_reduction.efa import efa
from dim_reduction.pca import pca


def reduction(
        df: pd.DataFrame,
        *,
        pca_kwargs: dict = {},
        efa_kwargs: dict = {},
):
    _, _, = pca(df, **pca_kwargs)
    _, _, = efa(df, **efa_kwargs)


if __name__ == "__main__":
    df = pd.read_csv(Path(input("Path to metrics.csv:")))

    pca_args = {
        "scale": True,
        "n_components": 0.8,
    }

    efa_args = {
        "scale": True,
        "n_factors": "auto",
        "rotation": "promax",
        "method": "ml",
        "kmo_warn": 0.6
    }

    reduction(df, pca_kwargs=pca_args, efa_kwargs=efa_args)
