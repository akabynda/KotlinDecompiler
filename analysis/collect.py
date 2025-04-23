from __future__ import annotations

from pathlib import Path

import pandas as pd

from entropy.common import cross_entropy_ngram
from entropy.conditional_entropy import compute as Hc
from entropy.cross_entropy import compute as CE
from entropy.kl_divergence import compute as KL
from entropy.nid import compute as NID
from entropy.perplexity import compute as PPL
from metrics import registry
from utils.kotlin_parser import parse

TEST_ROOT = Path(input("Путь к папке с тестами: ").strip()).expanduser()
DECOMPILERS = {"Bytecode", "CFR", "JDGUI", "Fernflower"}
CONVERTERS = {"ChatGPT", "J2K"}
NGRAM_MAX = 11


def read_kt(p: Path) -> str:
    if p.is_file(): return p.read_text("utf8")
    return "\n".join(f.read_text("utf8") for f in p.rglob("*.kt"))


def struct(src: str):
    t = parse(src)
    return {n: fn(t) for n, fn in registry.items()}


def ent(orig: str, dec: str):
    m = dict(CE=CE(orig, dec), KL=KL(orig, dec),
             PPL=PPL(orig, dec), NID=NID(orig, dec), Hcond=Hc(orig, dec))
    for n in range(1, NGRAM_MAX + 1):
        m[f"CE_{n}gram"] = cross_entropy_ngram(orig, dec, n)
    return m


rows = []
for test in TEST_ROOT.iterdir():
    if not test.is_dir(): continue
    orig = read_kt(test)
    for dec in DECOMPILERS:
        root = test / dec
        if not root.is_dir(): continue
        for kt in root.rglob("*.kt"):
            if "CodeConvert" in kt.stem: continue
            cv = next((c for c in CONVERTERS if c in kt.stem), None)
            if cv is None: continue
            cat = f"{dec}{cv}"
            code = read_kt(kt.parent if kt.parent != root else kt)
            row = {"Test": test.name, "Category": cat}
            row |= struct(code)
            row |= ent(orig, code)
            rows.append(row)

pd.DataFrame(rows).to_csv("full_metrics.csv", index=False)
print("✓ full_metrics.csv")
