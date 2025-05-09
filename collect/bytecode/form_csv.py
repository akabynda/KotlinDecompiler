import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def run(cmd: List[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def get_root_name(class_file: Path) -> str:
    return class_file.stem.split('$')[0]


def guess_kt_filename(root: str) -> str:
    base = root[:-2] if root.endswith("Kt") else root
    return base[:1].lower() + base[1:] + ".kt"


def index_kt_files(originals_root: Path) -> Dict[Tuple[str, str], List[Path]]:
    idx: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for kt in originals_root.rglob("*.kt"):
        repo = kt.relative_to(originals_root).parts[0]
        idx[(repo, kt.name.lower())].append(kt)
    return idx


def build_pairs(orig_root: Path, bc_root: Path) -> Dict[Path, List[Path]]:
    kt_index = index_kt_files(orig_root)
    pairs: Dict[Path, List[Path]] = defaultdict(list)

    skipped_stdlib = ("kotlin/", "kotlinx/", "java/", "javax/")

    for cls in bc_root.rglob("*.class"):
        p = cls.as_posix()
        if "/META-INF/" in p or any(s in p for s in skipped_stdlib):
            continue

        rel_cls = cls.relative_to(bc_root)
        repo = rel_cls.parts[0]
        cls_subpath = rel_cls.with_suffix("").parts[1:]
        root = get_root_name(cls)
        kt_filename = guess_kt_filename(root).lower()

        cands = kt_index.get((repo, kt_filename))
        if not cands:
            continue

        if len(cands) == 1:
            pairs[cands[0]].append(cls)
        else:
            for cand in cands:
                cand_rel = cand.relative_to(orig_root / repo)
                cand_parts = cand_rel.with_suffix("").parts
                if cls_subpath[-len(cand_parts):] == [p.lower() for p in cand_parts]:
                    pairs[cand].append(cls)
                    break
                else:
                    print(cand)

    return pairs


def write_jsonl(pairs: Dict[Path, List[Path]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_obj = 0
    with out_path.open("w", encoding="utf-8") as f:
        for kt_path, class_list in pairs.items():
            kt_source = kt_path.read_text(encoding="utf-8", errors="ignore")
            classes = [
                {
                    "class_path": str(cls.relative_to(kt_path.parents[2])),
                    "javap": run(["javap", "-c", "-p", str(cls)]),
                }
                for cls in class_list
            ]
            rec = {
                "kt_path": str(kt_path.relative_to(kt_path.parents[1])),
                "kt_source": kt_source,
                "classes": classes,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_obj += 1
    print(f"Записано {n_obj} объектов в {out_path}")


def main() -> None:
    ds_root = Path(
        input("Путь к dataset (должны быть originals/ и bytecode/): ").strip()
    ).resolve()
    orig = ds_root / "originals"
    bc = ds_root / "bytecode"
    if not orig.is_dir() or not bc.is_dir():
        raise SystemExit("originals/ или bytecode/ не найдены.")

    pairs = build_pairs(orig, bc)
    print(f"Найдено {len(pairs)} исходных .kt с байткодом")

    write_jsonl(pairs, ds_root / "pairs.jsonl")


if __name__ == "__main__":
    main()
