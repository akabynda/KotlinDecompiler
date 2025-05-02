from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from entropy import Entropy

language = "kotlin"
dataset_name = "KStack"

entr = Entropy(language)
ds_train = load_dataset(
    f"JetBrains/{dataset_name}",
    split="train",
    streaming=True,
)

uni = Counter()
bi = Counter()
tot_uni = 0
tot_bi = 0

for row in tqdm(ds_train, desc=dataset_name.lower()):
    toks = entr.tokens(row["content"])
    uni.update(toks)
    bi.update(zip(toks, toks[1:]))
    tot_uni += len(toks)
    tot_bi += max(0, len(toks) - 1)

uni_prob = {t: v / tot_uni for t, v in uni.items()}
bi_prob = {k: v / tot_bi for k, v in bi.items()}
left_prob = {a: v / tot_bi for a, v in Counter(a for a, _ in bi).items()}

out = Path(f"lang_models/{language.lower()}/{dataset_name.lower()}")
out.mkdir(parents=True, exist_ok=True)
with open(out / "unigram.json", "w", encoding="utf8") as f:
    json.dump(uni_prob, f, ensure_ascii=False)
bi_nested: dict[str, dict[str, float]] = defaultdict(dict)
for (a, b), p in bi_prob.items():
    bi_nested[a][b] = p
with open(out / "bigram.json", "w", encoding="utf8") as f:
    json.dump(bi_nested, f, ensure_ascii=False)
with open(out / "left.json", "w", encoding="utf8") as f:
    json.dump(left_prob, f, ensure_ascii=False)

print(f"сохранено: {len(uni_prob):,} токенов / {len(bi_prob):,} биграмм")
