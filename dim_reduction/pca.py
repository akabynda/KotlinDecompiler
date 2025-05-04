from typing import List, Tuple, Dict

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dim_reduction.common import numeric_df, safe_corr_df


def pca(df: pd.DataFrame,
        scale: bool = True,
        n_components: float | int = 0.95,
        top_k: int = 3,
        recommend: str | None = "union"
        ) -> Tuple[PCA, pd.DataFrame, Dict[str, List[str]], List[str]]:
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

    top_features: Dict[str, List[str]] = {}
    for pc in loadings.columns:
        abs_load = loadings[pc].abs()
        top_features[pc] = abs_load.nlargest(top_k).index.tolist()

    print("\n=== Сводка PCA ===")
    expl = pca.explained_variance_ratio_
    cum = expl.cumsum()
    for i, (v, c) in enumerate(zip(expl, cum), start=1):
        pcs = ", ".join(top_features[f"PC{i}"])
        print(f"PC{i:>2}: {v:5.1%} (накопл. {c:5.1%})  →  {pcs}")

    if recommend == "union":
        recommended = sorted({f for lst in top_features.values() for f in lst})
        title = "Рекомендуемые признаки (объединённый топ)"
    elif recommend == "per_pc":
        recommended = []
        used = set()
        for pc in loadings.columns:
            candidates = loadings[pc].abs().sort_values(ascending=False).index.tolist()
            for feat in candidates:
                if feat not in used:
                    recommended.append(feat)
                    used.add(feat)
                    break
        title = "Рекомендуемые признаки (по одному на каждую компоненту)"
    else:
        recommended = []
        title = None

    if title:
        print(f"\n{title}:")
        print(", ".join(recommended))

    return pca, loadings, top_features, recommended
