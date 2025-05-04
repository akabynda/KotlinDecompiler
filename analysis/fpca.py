from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA

BEST_N_TXT = Path("./ngram_best.txt")
FULL_METRICS = Path("../full_metrics.csv")
PC_OUT = Path("./fpca_scores.csv")
INFO_OUT = Path("./fpca_info.txt")

best_n = int(BEST_N_TXT.read_text().strip())
df = pd.read_csv(FULL_METRICS)

ce_cols = [f"CE_{n}gram" for n in range(1, best_n + 1)]

mask_dec = df["Category"] != "Original"
fd_train = FDataGrid(df.loc[mask_dec, ce_cols].to_numpy(),
                     grid_points=list(range(1, best_n + 1)))

fpca_tmp = FPCA(n_components=best_n).fit(fd_train)

cumvar = np.cumsum(fpca_tmp.explained_variance_ratio_)
delta = np.diff(np.hstack(([0], cumvar)))
k = next((i for i, d in enumerate(delta[1:], 1) if d < 0.03), best_n)
k = max(1, k)

fpca = FPCA(n_components=k).fit(fd_train)

fd_all = FDataGrid(df[ce_cols].to_numpy(),
                   grid_points=list(range(1, best_n + 1)))
scores = fpca.transform(fd_all)

pd.DataFrame(scores,
             columns=[f"PC{i}" for i in range(1, k + 1)]
             ).to_csv(PC_OUT, index=False)

with INFO_OUT.open("w") as f:
    f.write(f"best_n = {best_n}\n")
    f.write(f"выбрано PC = {k}\n")
    f.write("накопленная дисперсия:\n")
    for i, v in enumerate(cumvar[:k], 1):
        f.write(f"  PC1..PC{i}: {v:.3%}\n")

print(f"PC-координаты → {PC_OUT}"
      f"сводка → {INFO_OUT}")
