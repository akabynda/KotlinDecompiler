from typing import List, Tuple

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dim_reduction.common import numeric_df, safe_corr_df


def pca(df: pd.DataFrame,
        scale: bool = True,
        n_components: float | int = 0.95
        ) -> Tuple[PCA, pd.DataFrame]:
    num = numeric_df(df)

    removed: List[str] = []
    num = safe_corr_df(num, removed)
    if removed:
        print("Окончательно удалены как избыточные:", removed)

    X = StandardScaler().fit_transform(num) if scale else num.values

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=num.columns,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)]
    )

    print("\n=== Сводка PCA ===")
    expl = pca.explained_variance_ratio_
    cum = expl.cumsum()
    for i, (v, c) in enumerate(zip(expl, cum), start=1):
        pc = f"PC{i}"
        print(f"\n{pc}: {v:5.1%} (накопл. {c:5.1%})")
        sorted_feats = loadings[pc].abs().sort_values(ascending=False)
        for feat in sorted_feats.index:
            raw_value = loadings.loc[feat, pc]
            print(f"  {feat:<40} {raw_value: .4f}")

    return pca, loadings
