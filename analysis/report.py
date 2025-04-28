from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import pandas as pd

FULL = Path("./full_metrics.csv")
LOAD = Path("./factor_loadings.csv")
SCORES = Path("./factor_scores.csv")
FPCA = Path("./fpca_scores.csv")

LOADING_CUT = 0.50  # |λ|
CORR_TH = 0.70  # |r|


def _label(col: pd.Series) -> str:
    hi = col[abs(col) >= LOADING_CUT].abs().sort_values(ascending=False)
    if hi.empty:
        return "Misc"
    top = " ".join(hi.index[:3]).lower()
    if any(k in top for k in ("cyclomatic", "conditional", "chapin", "pivovarsky")):
        return "Struct"
    if any(m.startswith("halstead") for m in hi.index[:3]):
        return "Halstead"
    if any(m.startswith("detekt") for m in hi.index[:3]):
        return "Detekt"
    if any(k in top for k in ("ce_", "kl", "ppl", "nid", "hcond", "ce ")):
        return "Entropy"
    return " / ".join(hi.index[:3])


def _save(path: Path, df: pd.DataFrame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


df_all = pd.read_csv(FULL)
df_all = df_all[df_all["Category"] != "Original"]

loadings = pd.read_csv(LOAD, index_col=0)
scores = pd.read_csv(SCORES).join(df_all[["Test", "Category"]])

factor_names = {f: _label(loadings[f]) for f in loadings.columns}

rows_fs: List[Tuple[str, str, float]] = [
    (f, m, l)
    for f in loadings.columns
    for m, l in loadings[f].items()
    if abs(l) >= LOADING_CUT
]
_save(Path("./factor_structure.csv"),
      pd.DataFrame(rows_fs, columns=["Factor", "Metric", "Loading"]))

cat_stats = (
    scores.groupby("Category")[loadings.columns]
    .agg(["mean", "std"])
    .stack(level=1, future_stack=True)
    .reset_index()
    .rename(columns={"level_1": "Factor",
                     "mean": "Mean",
                     "std": "Std"})
)
_save(Path("./category_factor_scores.csv"), cat_stats)

struct_cols = [c for c in df_all.columns
               if c not in ("Test", "Category") and not c.startswith("CE_")]
entropy_cols = [c for c in df_all.columns if c.startswith("CE_")] + \
               ["CE", "KL", "PPL", "NID", "Hcond"]

corr = df_all[struct_cols + entropy_cols].corr().loc[struct_cols, entropy_cols]
strong = (corr.where(corr.abs() >= CORR_TH)
          .stack()
          .reset_index()
          .rename(columns={0: "r",
                           "level_0": "Metric1",
                           "level_1": "Metric2"}))
strong = strong[strong["Metric1"] != strong["Metric2"]]
_save(Path("./strong_correlations.csv"), strong)

if FPCA.exists():
    pc = pd.read_csv(FPCA)
    mat = pd.concat([scores[loadings.columns], pc], axis=1).corr()
    _save(Path("./fa_vs_fpca.csv"),
          mat.loc[loadings.columns, pc.columns]
          .reset_index().rename(columns={"index": "FA_Factor"}))
print("done")
