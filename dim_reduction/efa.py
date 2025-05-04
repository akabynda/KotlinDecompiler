from typing import List, Tuple, Dict, Literal, Optional

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_kmo,
)
from sklearn.preprocessing import StandardScaler

from dim_reduction.common import numeric_df, safe_corr_df


def efa(
        df: pd.DataFrame,
        *,
        scale: bool = True,
        n_factors: int | Literal["auto"] = "auto",
        rotation: Optional[str] = "varimax",
        method: Optional[str] = "minres",
        top_k: int = 3,
        recommend: Literal["union", "per_factor", None] = "union",
        kmo_warn: float = 0.60,
) -> Tuple[FactorAnalyzer, pd.DataFrame, Dict[str, List[str]], List[str]]:
    num = numeric_df(df)
    removed: List[str] = []
    num = safe_corr_df(num, removed)
    if removed:
        print("Окончательно удалены как избыточные:", removed)

    X = StandardScaler().fit_transform(num) if scale else num.values

    kmo_per_item, kmo_total = calculate_kmo(X)

    min_vars = max(3, X.shape[1] // 2)

    while kmo_total < kmo_warn and num.shape[1] > min_vars:
        worst_idx = int(np.argmin(kmo_per_item))
        worst_feat = num.columns[worst_idx]
        print(f"Удаляем {worst_feat!r} (KMO={kmo_per_item[worst_idx]:.3f}) для улучшения KMO")
        num = num.drop(columns=[worst_feat])
        X = StandardScaler().fit_transform(num) if scale else num.values

        kmo_per_item, kmo_total = calculate_kmo(X)
        print(f"Новый общий KMO = {kmo_total:5.3f}")

    if kmo_total < kmo_warn:
        print(f"После всех удалений общий KMO всё ещё < {kmo_warn:.2f}")

    print("\n=== Диагностика факторизации ===")
    print(f"KMO (общий)          : {kmo_total:5.3f}")

    if n_factors == "auto":
        fa_tmp = FactorAnalyzer(rotation=None)
        fa_tmp.fit(X)
        ev, _ = fa_tmp.get_eigenvalues()
        n_factors = int((ev > 1.0).sum()) or 1
        print(f"Автовыбор числа факторов: {n_factors}")

    fa = FactorAnalyzer(
        n_factors=n_factors,
        rotation=rotation,
        method=method,
        is_corr_matrix=False,
        rotation_kwargs={"tol": 1e-5},
    )
    fa.fit(X)

    loadings = pd.DataFrame(
        fa.loadings_,
        index=num.columns,
        columns=[f"F{i + 1}" for i in range(n_factors)],
    )

    top_features: Dict[str, List[str]] = {}
    for f in loadings.columns:
        abs_load = loadings[f].abs()
        top_features[f] = abs_load.nlargest(top_k).index.tolist()

    print("\n=== Сводка EFA ===")
    var_exp, prop_var, cum_var = fa.get_factor_variance()
    for i in range(n_factors):
        feats = ", ".join(top_features[f"F{i + 1}"])
        print(
            f"F{i + 1:>2}: {prop_var[i]:5.1%} "
            f"(накопл. {cum_var[i]:5.1%})  →  {feats}"
        )

    if recommend == "union":
        recommended = sorted({f for lst in top_features.values() for f in lst})
        title = "Рекомендуемые признаки (объединённый топ)"
    elif recommend == "per_factor":
        recommended, used = [], set()
        for f in loadings.columns:
            for feat in loadings[f].abs().sort_values(ascending=False).index:
                if feat not in used:
                    recommended.append(feat)
                    used.add(feat)
                    break
        title = "Рекомендуемые признаки (по одному на фактор, без дубликатов)"
    else:
        recommended, title = [], None

    if title:
        print(f"\n{title}:")
        print(", ".join(recommended))

    return fa, loadings, top_features, recommended
