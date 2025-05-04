import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from metrics import registry
from metrics.entropy import Entropy
from utils.kotlin_parser import parse

language = "kotlin"
entr = Entropy(language)

decompilers = {"Bytecode", "CFR", "JDGUI", "Fernflower"}
converters = {"ChatGPT", "J2K"}


def read_kt(path: Path) -> str:
    if path.is_file():
        return path.read_text("utf8")
    return "\n".join(p.read_text("utf8") for p in path.rglob("*.kt"))


def read_kt_flat(path: Path) -> str:
    if path.is_file():
        return path.read_text("utf8")
    return "\n".join(p.read_text("utf8") for p in path.glob("*.kt"))


def structural(src: str) -> dict[str, float]:
    tree = parse(src)
    return {n: fn(tree) for n, fn in registry.items()}


def entropy_metrics(orig: str, dec: str) -> dict[str, float]:
    return {
        "CE": entr.cross_entropy(orig, dec),
        "KL": entr.kl_div(orig, dec),
        "PPL": entr.perplexity(orig, dec),
        "JSD": entr.jensen_shannon_distance(orig, dec),
        "CondE": entr.conditional_entropy(orig, dec),
    }


def load_lm(model_name: str) -> tuple[dict, dict, dict]:
    lm_dir = Path(f"../lang_models/{language}/{model_name}")
    with open(lm_dir / "unigram.json", encoding="utf8") as f:
        p_uni = json.load(f)
    with open(lm_dir / "bigram.json", encoding="utf8") as f:
        p_bi = json.load(f)
    with open(lm_dir / "left.json", encoding="utf8") as f:
        p_left = json.load(f)
    return p_uni, p_bi, p_left


def lm_metrics(p_uni: dict, p_bi: dict, p_left: dict, src: str) -> dict[str, float]:
    return {
        "LM_CE": entr.cross_entropy_lang(p_uni, src),
        "LM_KL": entr.kl_div_lang(p_uni, src),
        "LM_PPL": entr.perplexity_lang(p_uni, src),
        "LM_JSD": entr.jensen_shannon_distance_lang(p_uni, src),
        "LM_CondE": entr.conditional_entropy_lang(p_bi, p_left, src),
    }

def lm_metrics_wo_cond_entr(p_uni: dict, src: str) -> dict[str, float]:
    return {
        "LM_CE": entr.cross_entropy_lang(p_uni, src),
        "LM_KL": entr.kl_div_lang(p_uni, src),
        "LM_PPL": entr.perplexity_lang(p_uni, src),
        "LM_JSD": entr.jensen_shannon_distance_lang(p_uni, src),
    }

def collect_tests(test_root: Path) -> dict[str, dict]:
    tests: dict[str, dict] = {}
    for td in test_root.iterdir():
        if not td.is_dir():
            continue
        orig_code = read_kt_flat(td)
        if not orig_code:
            continue
        tests[td.name] = {"orig": orig_code, "decs": {}}
        for dec in decompilers:
            root = td / dec
            if not root.is_dir():
                continue
            buckets: dict[str, set[Path]] = defaultdict(set)
            for f in root.glob("*.kt"):
                if "CodeConvert" in f.stem:
                    continue
                cv = next((c for c in converters if c in f.stem), None)
                if cv:
                    buckets[f"{dec}{cv}"].add(f)
            for sub in root.iterdir():
                if not sub.is_dir() or "CodeConvert" in sub.name:
                    continue
                cv = next((c for c in converters if c in sub.name), None)
                if cv:
                    buckets[f"{dec}{cv}"].add(sub)
            for cat, paths in buckets.items():
                code = "\n".join(read_kt(p) for p in sorted(paths))
                tests[td.name]["decs"][cat] = code
    return tests


def build_pairs(tests: dict[str, dict]) -> list[tuple[str, str, str, str]]:
    pairs: list[tuple[str, str, str, str]] = []
    for test, data in tests.items():
        pairs.append((test, "Original", data["orig"], data["orig"]))
        for cat, code in data["decs"].items():
            pairs.append((test, cat, code, data["orig"]))
    if not pairs:
        raise ValueError("no original/decompiled pairs found")
    return pairs


_issue_re = re.compile(r"^(?P<issue>\w+)\s+-.*\s+at\s+(?P<path>/.*?):\d+:\d+")


def parse_detekt(report_path: Path, test_root: Path) -> pd.DataFrame:
    detekt: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    with report_path.open(encoding="utf8") as f:
        for line in f:
            m = _issue_re.match(line)
            if not m:
                continue
            issue = m["issue"]
            try:
                rel = Path(m["path"]).relative_to(test_root)
            except ValueError:
                continue
            parts = rel.parts
            decomp = parts[1] if len(parts) > 1 and parts[1] in decompilers else "Original"
            conv = "Original"
            if decomp != "Original" and len(parts) > 2:
                conv = next((c for c in converters if c in parts[2]), conv)
            if decomp != "Original" and conv == "Original":
                continue
            cat = "Original" if decomp == conv == "Original" else f"{decomp}{conv}"
            detekt[cat][issue] += 1
    return pd.DataFrame(detekt).T.fillna(0).astype(int)
