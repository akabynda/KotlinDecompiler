from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from collect.common import structural, load_lm

out_csv = Path("kstack-clean_metrics.csv")

ds = load_dataset("JetBrains/KStack-clean", split="train", streaming=True)

p_uni, _, _ = load_lm("kstack-clean+kexercises")

rows: list[dict[str, float]] = []
count = 0
for example in ds:
    count += 1
    dec_code = example["content"]
    row: dict[str, Any] = {"Test": f"{count}", "Category": "Original"}
    row.update(structural(dec_code))
    # row.update(lm_metrics_wo_cond_entr(src=dec_code, p_uni=p_uni))
    rows.append(row)

df = pd.DataFrame(rows).reset_index(drop=True)
df.to_csv(out_csv, index=False)
print(f"{out_csv} (строк: {len(df)})")
