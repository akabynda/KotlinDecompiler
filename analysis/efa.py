from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from factor_analyzer import (FactorAnalyzer, calculate_kmo,
                             calculate_bartlett_sphericity)
from sklearn.preprocessing import StandardScaler

DATA_CSV = Path("./full_metrics.csv")
ROTATION = "varimax"
DELTA_THRESHOLD = 0.03  # мин. прирост, % дисперсии в долях (0.03 = 3 %)
PA_ITER = 100  # итерации для parallel analysis

df = pd.read_csv(DATA_CSV)
X = df.drop(columns=["Test", "Category"]).fillna(0)
Xs = StandardScaler().fit_transform(X)
n_obs, n_vars = Xs.shape

kmo_item, kmo_overall = calculate_kmo(Xs)
chi2, p_val = calculate_bartlett_sphericity(Xs)

corr = np.corrcoef(Xs, rowvar=False)
eigvals = np.sort(np.linalg.eigvals(corr).real)[::-1]

# --- Parallel analysis ---
rand_eigs = np.zeros((PA_ITER, n_vars))
for i in range(PA_ITER):
    R = np.random.normal(size=(n_obs, n_vars))
    R = StandardScaler().fit_transform(R)
    rand_eigs[i] = np.sort(np.linalg.eigvals(np.corrcoef(R, rowvar=False)).real)[::-1]
mean_rand = rand_eigs.mean(axis=0)

k_horn = int((eigvals > mean_rand).sum())  # ≥ 1 фактор, гарантируем
k = max(1, k_horn)

# --- приростная остановка ---
fa_tmp = FactorAnalyzer(n_factors=k, rotation=None).fit(Xs)
cumvar = fa_tmp.get_factor_variance()[1]  # кумулятивная объяснённая

while k < n_vars:
    marginal = cumvar[k - 1] - cumvar[k - 2] if k > 1 else cumvar[0]
    if marginal < DELTA_THRESHOLD:
        break
    k += 1
    fa_tmp = FactorAnalyzer(n_factors=k, rotation=None).fit(Xs)
    cumvar = fa_tmp.get_factor_variance()[1]

# --- финальный FA ---
fa = FactorAnalyzer(n_factors=k, rotation=ROTATION).fit(Xs)
loadings = pd.DataFrame(fa.loadings_,
                        index=X.columns,
                        columns=[f"F{i}" for i in range(1, k + 1)])
scores = pd.DataFrame(fa.transform(Xs),
                      columns=[f"F{i}" for i in range(1, k + 1)])

loadings.to_csv("factor_loadings.csv")
scores.to_csv("factor_scores.csv", index=False)

# --- графики Scree / Parallel ---
plt.figure()
plt.plot(range(1, len(eigvals) + 1), eigvals, 'o-')
plt.axhline(mean_rand[0], linestyle='--', color='gray')
plt.xlabel("Factor #");
plt.ylabel("Eigenvalue");
plt.title("Scree Plot")
plt.tight_layout();
plt.savefig("scree_plot.png")

plt.figure()
plt.plot(range(1, n_vars + 1), eigvals, label="Data")
plt.plot(range(1, n_vars + 1), mean_rand, '--', label="Random")
plt.xlabel("Factor #");
plt.ylabel("Eigenvalue");
plt.title("Parallel Analysis")
plt.legend();
plt.tight_layout();
plt.savefig("parallel_analysis.png")

# --- отчёт ---
with open("efa_info.txt", "w", encoding="utf8") as f:
    f.write(f"KMO overall: {kmo_overall:.3f}\n")
    f.write(f"Bartlett χ²={chi2:.1f} p={p_val:.3g}\n")
    f.write(f"Horn factors: {k_horn}\n")
    f.write(f"Chosen factors (Horn + Δ≥{DELTA_THRESHOLD:.0%}): {k}\n")
    f.write("Cumulative variance:\n")
    for i, v in enumerate(cumvar[:k], 1):
        f.write(f"  F1..F{i}: {v:.2%}\n")

print(f"✓ Factor analysis готов: k = {k}.")
