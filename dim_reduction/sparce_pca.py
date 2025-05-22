from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler

from dim_reduction.common import numeric_df, safe_corr_df


def run_sparse_pca(
        df: pd.DataFrame,
        *,
        scale: bool = True,
        n_components: int | None = 7,
        alpha: float = 1.0,
        print_zero_stats: bool = True,
) -> tuple[object, DataFrame, DataFrame, List[str]]:
    num = safe_corr_df(numeric_df(df), [])
    X = StandardScaler().fit_transform(num) if scale else num.values

    spca = SparsePCA(n_components=n_components,
                     alpha=alpha,
                     ridge_alpha=0.01,
                     ).fit(X)

    comps = pd.DataFrame(
        spca.transform(X),
        index=df.index,
        columns=[f"SPC{i + 1}" for i in range(n_components)],
    )

    loadings = pd.DataFrame(
        spca.components_.T,
        index=num.columns,
        columns=comps.columns,
    )

    if print_zero_stats:
        nz_per_pc = (loadings == 0).sum()
        print("\nSparsePCA: нулевые нагрузки")
        for pc in loadings.columns:
            zeros = int(nz_per_pc[pc])
            sparsity = zeros / loadings.shape[0] * 100
            print(f"{pc:<6}: {zeros:>3} нулей  ({sparsity:4.1f}%)")

    print("\n=== Сводка SparsePCA ===")
    main_feats = set()
    for pc in loadings.columns:
        nonzero = loadings[pc][loadings[pc] != 0]
        if nonzero.empty:
            continue
        print(f"\n{pc}:")
        for feat, w in nonzero.abs().sort_values(ascending=False).items():
            sign = loadings.loc[feat, pc]
            main_feats.add(feat)
            print(f"  {feat:<40} {sign: .4f}")
    main_feats = list(main_feats)

    return spca, comps, loadings, main_feats


def grid_search_spca(
        df: pd.DataFrame,
        *,
        n_components_list: List[int],
        alphas: List[float],
        min_explained_ratio: float = 0.50,
) -> Tuple[int, float]:
    num = safe_corr_df(numeric_df(df), [])
    X = StandardScaler().fit_transform(num)

    best_score = -np.inf
    best = (None, None)

    for n in n_components_list:
        for a in alphas:
            print("test:", n, a)
            spca = SparsePCA(n_components=n,
                             alpha=a,
                             ridge_alpha=0.01).fit(X)

            X_trans = spca.transform(X)
            var_comp = np.var(X_trans, axis=0)
            total_var = np.var(X, axis=0).sum()
            ratio = var_comp.sum() / total_var

            sparsity = np.mean(spca.components_ == 0)

            if ratio < min_explained_ratio:
                continue

            score = sparsity * ratio

            if score > best_score:
                best_score = score
                best = (n, a)

            print(f"n={n:2d}, α={a:4.2f} → var={ratio:5.1%}, sparsity={sparsity:5.1%}, score={score:5.3f}")

    if best[0] is None:
        raise ValueError("Нет сочетаний, удовлетворяющих порогу explained_ratio")
    print(f"\nЛучшее: n_components={best[0]}, alpha={best[1]:.2f} (score={best_score:5.3f})")
    return best


if __name__ == "__main__":
    df = pd.read_csv(Path(input("Path to metrics.csv:")))

    n_list = list(range(14, 24))
    alpha_list = list()
    for i in range(1, 201):
        alpha_list.append(i * 0.1)

    best_n, best_alpha = grid_search_spca(
        df,
        n_components_list=n_list,
        alphas=alpha_list,
        min_explained_ratio=0.80
    )

    """best_n = 14
    best_alpha = 5.7"""

    spca, comps, loadings, main_feats = run_sparse_pca(
        df,
        n_components=best_n,
        alpha=best_alpha,
    )

    print("Все признаки:", len(main_feats), main_feats)

    dropped = loadings.index[(loadings == 0).all(axis=1)].tolist()
    print("\nПризнаки со всеми нулевыми нагрузками (можно исключить):")
    print(", ".join(dropped) if dropped else "‑")
