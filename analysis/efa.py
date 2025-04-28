from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from factor_analyzer import (
    FactorAnalyzer,
    calculate_kmo,
    calculate_bartlett_sphericity,
)
from sklearn.preprocessing import StandardScaler


def drop_high_corr(df: pd.DataFrame, th: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    corr = df.corr().abs()
    mask = np.triu(np.ones_like(corr, bool), k=1)
    to_drop = {corr.columns[j]
               for i, j in zip(*np.where((corr.values > th) & mask))}
    return df.drop(columns=to_drop), sorted(to_drop)


def cumulative_var(x_std: np.ndarray, k: int) -> float:
    return FactorAnalyzer(n_factors=k, rotation=None).fit(x_std) \
        .get_factor_variance()[1][k - 1]


DATA_CSV = Path("full_metrics.csv")
ROTATION = "varimax"
DELTA_TH = 0.03
PA_ITER = 100
CORR_TH = 0.99

df = pd.read_csv(DATA_CSV)
df = df[df["Category"] != "Original"]

X = df.drop(columns=["Test", "Category"]).copy()

const_cols = [c for c in X.columns if X[c].nunique() <= 1]
if const_cols:
    print("Константы удалены:", ", ".join(const_cols))
    X.drop(columns=const_cols, inplace=True)

X, dropped_corr = drop_high_corr(X, CORR_TH)
if dropped_corr:
    print("Cильно коррелирующие удалены:", ", ".join(dropped_corr))

Xs = StandardScaler().fit_transform(X)
n_obs, n_vars = Xs.shape

kmo_item, kmo_overall = calculate_kmo(Xs)

try:
    chi2, p_val = calculate_bartlett_sphericity(Xs)
except np.linalg.LinAlgError:
    chi2 = p_val = float("nan")

corr = np.corrcoef(Xs, rowvar=False)
eigvals = np.sort(np.linalg.eigvals(corr).real)[::-1]

rand_eigs = np.zeros((PA_ITER, n_vars))
for i in range(PA_ITER):
    R = StandardScaler().fit_transform(np.random.randn(n_obs, n_vars))
    rand_eigs[i] = np.sort(np.linalg.eigvals(np.corrcoef(R, rowvar=False)).real)[::-1]
mean_rand = rand_eigs.mean(axis=0)

k_horn = max(1, int((eigvals > mean_rand).sum()))

prev_cum = cumulative_var(Xs, k_horn)
k = k_horn
while k < n_vars:
    new_cum = cumulative_var(Xs, k + 1)
    if new_cum - prev_cum < DELTA_TH:
        break
    k += 1
    prev_cum = new_cum

fa = FactorAnalyzer(n_factors=k, rotation=ROTATION).fit(Xs)

loadings = pd.DataFrame(
    fa.loadings_, index=X.columns, columns=[f"F{i}" for i in range(1, k + 1)]
)
scores = pd.DataFrame(
    fa.transform(Xs), columns=[f"F{i}" for i in range(1, k + 1)]
)

loadings.to_csv("factor_loadings.csv")
scores.to_csv("factor_scores.csv", index=False)

plt.figure()
plt.plot(range(1, len(eigvals) + 1), eigvals, "o-")
plt.axhline(mean_rand[0], ls="--", c="gray")
plt.title("Scree plot")
plt.xlabel("Factor #")
plt.ylabel("Eigenvalue")
plt.tight_layout()
plt.savefig("scree_plot.png")

plt.figure()
plt.plot(range(1, n_vars + 1), eigvals, label="Data")
plt.plot(range(1, n_vars + 1), mean_rand, "--", label="Random mean")
plt.title("Parallel analysis")
plt.xlabel("Factor #")
plt.ylabel("Eigenvalue")
plt.legend()
plt.tight_layout()
plt.savefig("parallel_analysis.png")

cum = fa.get_factor_variance()[1]
with open("efa_info.txt", "w", encoding="utf8") as f:
    f.write(f"KMO overall = {kmo_overall:.3f}\n")
    f.write(f"Bartlett χ² = {chi2:.1f}, p = {p_val:.3g}\n\n")
    f.write(f"Horn → {k_horn} фактора(ов)\n")
    f.write(f"δ-порог {DELTA_TH:.0%} → итог k = {k}\n\n")
    f.write("Кумулятивная объяснённая дисперсия:\n")
    for i, v in enumerate(cum[:k], 1):
        f.write(f"  F{i}: {v:.2%}\n")

print(f"EFA готово — факторов: {k}")
