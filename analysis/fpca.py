from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA

from entropy.common import cross_entropy_ngram, _tokens

TEST_ROOT = Path(input("Путь к папке с тестами: ").strip()).expanduser()
DECOMPILERS = {"Bytecode", "CFR", "JDGUI", "Fernflower"}
CONVERTERS = {"ChatGPT", "J2K"}


def read_kt(p: Path) -> str:
    return p.read_text("utf8") if p.is_file() else "\n".join(f.read_text("utf8")
                                                             for f in p.rglob("*.kt"))


pairs = []
for td in TEST_ROOT.iterdir():
    if not td.is_dir(): continue
    orig = read_kt(td)
    if not orig: continue
    for dec in DECOMPILERS:
        for kt in (td / dec).rglob("*.kt"):
            if "CodeConvert" in kt.stem: continue
            cv = next((c for c in CONVERTERS if c in kt.stem), None)
            if cv:
                code = read_kt(kt.parent if kt.parent != td / dec else kt)
                pairs.append((td.name, f"{dec}{cv}", code, orig))

if not pairs:
    raise SystemExit("Нет комбинаций")

# найдём динамический hard_cap
avg_len = int(np.mean([len(_tokens(o)) for *_, o in pairs]))
hard_cap = max(3, int(avg_len * 0.25))  # правило ¼, но не меньше 3


# кривые CE_n-gram
def ce_curve(o, d): return [cross_entropy_ngram(o, d, n) for n in range(1, hard_cap + 1)]


df_ce = pd.DataFrame([ce_curve(o, c) for _, _, c, o in pairs],
                     index=[f"{t}/{cat}" for t, cat, *_ in pairs],
                     columns=[f"CE_{n}" for n in range(1, hard_cap + 1)])

rel = df_ce.pct_change(axis=1).abs().iloc[:, 1:]

# адаптивный REL_THRESH: берём ½-медианы всех приращений
REL_THRESH = rel.stack().median() * 0.5
best_n = next((i + 2 for i, v in enumerate(rel.mean()) if v < REL_THRESH), hard_cap)
Path("ngram_best.txt").write_text(str(best_n))
print(f"★ best_n = {best_n}, REL_THRESH = {REL_THRESH:.4f}")

#  FPCA: выбираем k по “мarginal gain < 3 %”
full = pd.read_csv("./full_metrics.csv")

# урезаем best_n по доступным столбцам
avail_n = sorted(int(c.split("_")[1][:-4]) for c in full.columns
                 if c.startswith("CE_") and c.endswith("gram"))
max_avail = max(avail_n)
if best_n > max_avail:
    print(f"only CE_1..CE_{max_avail}gram in CSV; "
          f"truncate best_n {best_n} → {max_avail}")
    best_n = max_avail
    Path("ngram_best.txt").write_text(str(best_n))

cols = [f"CE_{n}gram" for n in range(1, best_n + 1)]
fd = FDataGrid(full[cols].to_numpy(),
               grid_points=list(range(1, best_n + 1)))

fpca_all = FPCA(n_components=best_n).fit(fd)
cumvar = np.cumsum(fpca_all.explained_variance_ratio_)
delta = np.diff(np.hstack(([0], cumvar)))

idx = np.argmax(delta < 0.03)
k = max(1, idx) if delta[idx] < 0.03 else best_n  # страхуемся

fpca = FPCA(n_components=k).fit(fd)
pd.DataFrame(fpca.transform(fd),
             columns=[f"PC{i}" for i in range(1, k + 1)]
             ).to_csv("fpca_scores.csv", index=False)

with open("fpca_info.txt", "w") as f:
    f.write(f"best_n={best_n}, k={k}\n")
    f.write(f"Δ={delta[:k + 2]}\n")
    f.write(f"cum={cumvar[:k]}\n")

print("✓ fpca_scores.csv / fpca_info.txt")
