import json
import multiprocessing
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


def run(cmd: List[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True,
                          check=False).stdout


_pkg_re = re.compile(r"^\s*package\s+([\w.]+)")


def read_package(kt_path: Path) -> str:
    with kt_path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if m := _pkg_re.match(line):
                return m.group(1)
            if line.strip() and not line.lstrip().startswith("//"):
                break
    return ""


def get_root_name(class_file: Path) -> str:
    return class_file.stem.split("$")[0]


def guess_kt_filename(root: str) -> str:
    base = root[:-2] if root.endswith("Kt") else root
    return (base[:1].lower() + base[1:] + ".kt").lower()


DirKey = Tuple[str, Tuple[str, ...], str]
PkgKey = Tuple[str, Tuple[str, ...], str]


def _index_one(kt_path: Path, originals_root: Path):
    repo = kt_path.relative_to(originals_root).parts[0]
    dir_parts = kt_path.relative_to(originals_root / repo).parent.parts
    dir_key = (repo, dir_parts, kt_path.name.lower())

    pkg_str = read_package(kt_path)
    pkg_parts = tuple(pkg_str.split(".")) if pkg_str else ()
    pkg_key = (repo, pkg_parts, kt_path.name.lower())
    return dir_key, pkg_key, kt_path


def index_kt_files(originals_root: Path):
    idx_dir, idx_pkg = {}, defaultdict(list)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as pool:
        tasks = (kt for kt in originals_root.rglob("*.kt"))
        for dir_key, pkg_key, kt_path in pool.map(
                partial(_index_one, originals_root=originals_root), tasks):
            if dir_key in idx_dir:
                raise ValueError(f"Дубликат файла {kt_path}")
            idx_dir[dir_key] = kt_path
            idx_pkg[pkg_key].append(kt_path)

    return idx_dir, idx_pkg


def build_pairs(orig_root: Path, bc_root: Path) -> Dict[Path, List[Path]]:
    idx_dir, idx_pkg = index_kt_files(orig_root)
    pairs: Dict[Path, List[Path]] = defaultdict(list)

    skipped_prefixes = ("kotlin/", "kotlinx/", "java/", "javax/")

    for cls in bc_root.rglob("*.class"):
        p = cls.as_posix()
        if "/META-INF/" in p or any(p.startswith(
                bc_root.joinpath(pref).as_posix()) for pref in skipped_prefixes):
            continue

        repo, *pkg_dirs, _fname = cls.relative_to(bc_root).parts
        pkg_parts: Tuple[str, ...] = tuple(pkg_dirs)

        root = get_root_name(cls)
        kt_name = guess_kt_filename(root)
        pkg_key: PkgKey = (repo, pkg_parts, kt_name)

        candidates = idx_pkg.get(pkg_key, [])

        if len(candidates) > 1:
            raise ValueError(candidates)

        if len(candidates) != 1:
            continue

        kt_path = candidates[0]

        if cls in pairs[kt_path]:
            raise ValueError(f"Дублирование класса {cls} для {kt_path}")
        pairs[kt_path].append(cls)

    return pairs


def _build_record(task: Tuple[Path, List[Path], Path, Path]) -> str:
    kt_path, cls_list, orig_root, bc_root = task
    record = {
        "kt_path": str(kt_path.relative_to(orig_root)),
        "kt_source": kt_path.read_text(encoding="utf-8", errors="ignore"),
        "classes": [
            {
                "class_path": str(c.relative_to(bc_root)),
                "javap": run(["javap", "-c", "-p", str(c)]),
            }
            for c in cls_list
        ],
    }
    return json.dumps(record, ensure_ascii=False)


def write_jsonl(pairs: Dict[Path, List[Path]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = out_path.parent
    orig_root = dataset_root / "originals"
    bc_root = dataset_root / "bytecode"

    tasks = [
        (kt_path, cls_list, orig_root, bc_root)
        for kt_path, cls_list in pairs.items()
        if cls_list
    ]

    with out_path.open("w", encoding="utf-8") as fh, \
            ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as pool:
        for json_line in tqdm(pool.map(_build_record, tasks, chunksize=8),
                              total=len(tasks),
                              desc="javap"):
            fh.write(json_line + "\n")

    print(f"Записано {len(tasks)} объектов → {out_path}")


def main() -> None:
    ds_root = Path(
        input("Путь к dataset (должны быть originals/ и bytecode/): ").strip()
    ).resolve()
    orig = ds_root / "originals"
    bc = ds_root / "bytecode"
    if not orig.is_dir() or not bc.is_dir():
        raise SystemExit("originals/ или bytecode/ не найдены.")

    pairs = build_pairs(orig, bc)
    print(f"Найдено {len(pairs)} .kt‑файлов с байткодом")

    write_jsonl(pairs, ds_root / "pairs.jsonl")


if __name__ == "__main__":
    main()
