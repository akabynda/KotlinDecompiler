from __future__ import annotations

import json
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from metrics.entropy import Entropy

language = "kotlin"

datasets = [
    ("KStack-clean", "content", lambda row: bool(row["content"])),
    ("KExercises", "solution", lambda row: bool(row["solution"])),
]

"""
datasets = [
    ("KStack", "content",
     lambda row: bool(row["content"]) and (int(row.get("stars", 0)) + int(row.get("forks", 0)) >= 6))
]"""

model_name = "+".join(d[0] for d in datasets)
out = Path(f"lang_models/{language.lower()}/{model_name.lower()}")
tmp_dir = out / "_partials"
out.mkdir(parents=True, exist_ok=True)
tmp_dir.mkdir(exist_ok=True)

uni = Counter()
bi = Counter()
tot_uni = 0
tot_bi = 0
TOKENS_PER_DUMP = 10_000_000
token_buf = 0

entr = Entropy(language)


def dump_partial(idx: int):
    fname = tmp_dir / f"partial-{idx}.pkl"
    with fname.open("wb") as f:
        pickle.dump((uni.copy(), bi.copy()), f, protocol=pickle.HIGHEST_PROTOCOL)
    uni.clear()
    bi.clear()


partial_idx = 0
for dname, col, keep in datasets:
    ds = load_dataset(f"JetBrains/{dname}", split="train", streaming=True)
    for row in tqdm(ds, desc=dname, leave=False):
        if not keep(row): continue
        toks = entr.tokens(row[col])
        uni.update(toks)
        bi.update(zip(toks, toks[1:]))
        tot_uni += len(toks)
        tot_bi += max(0, len(toks) - 1)
        token_buf += len(toks)

        if token_buf >= TOKENS_PER_DUMP:
            dump_partial(partial_idx)
            partial_idx += 1
            token_buf = 0

if token_buf > 0:
    dump_partial(partial_idx)

final_uni = Counter()
final_bi = Counter()

for pkl in tmp_dir.glob("partial-*.pkl"):
    with pkl.open("rb") as f:
        u, b = pickle.load(f)
    final_uni.update(u)
    final_bi.update(b)

shutil.rmtree(tmp_dir)

uni_prob = {t: v / tot_uni for t, v in final_uni.items()}
bi_prob = {k: v / tot_bi for k, v in final_bi.items()}
left_prob = {a: v / tot_bi for a, v in Counter(a for a, _ in final_bi).items()}

with open(out / "unigram.json", "w", encoding="utf8") as f:
    json.dump(uni_prob, f, ensure_ascii=False)

nested: dict[str, dict[str, float]] = defaultdict(dict)
for (a, b), p in bi_prob.items():
    nested[a][b] = p
with open(out / "bigram.json", "w", encoding="utf8") as f:
    json.dump(nested, f, ensure_ascii=False)

with open(out / "left.json", "w", encoding="utf8") as f:
    json.dump(left_prob, f, ensure_ascii=False)

meta = {
    "model_name": model_name,
    "num_unigrams": len(uni_prob),
    "num_bigrams": len(bi_prob),
    "tokens_total": tot_uni,
    "bigrams_total": tot_bi,
    "partials": partial_idx + 1,
    "tokens_per_dump": TOKENS_PER_DUMP,
}
with open(out / "metadata.json", "w", encoding="utf8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"{model_name}: {meta['num_unigrams']:,} токенов / "
      f"{meta['num_bigrams']:,} биграмм, {meta['partials']} partial‑файлов слиты")
