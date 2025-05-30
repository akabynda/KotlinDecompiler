import json
import multiprocessing
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

DirKey = Tuple[str, Tuple[str, ...], str]
PkgKey = Tuple[str, Tuple[str, ...], str]


class BytecodePairCollector:
    """
    Collects Kotlin source files and their corresponding bytecode classes, then runs javap on them
    and saves the results in a JSONL file.
    """

    def __init__(self, dataset_root: Path) -> None:
        """
        Initialize the collector with the dataset root directory.

        Args:
            dataset_root (Path): Root directory containing originals/ and bytecode/ subdirectories.
        """
        self.dataset_root: Path = dataset_root
        self.originals_root: Path = dataset_root / "originals"
        self.bytecode_root: Path = dataset_root / "bytecode"
        self._pkg_re = re.compile(r"^\s*package\s+([\w.]+)")

    @staticmethod
    def run_command(cmd: List[str]) -> Tuple[str, str, int]:
        """
        Run a shell command and capture its output.

        Args:
            cmd (List[str]): Command as a list of arguments.

        Returns:
            Tuple containing stdout, stderr, and return code.
        """
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.stdout, proc.stderr, proc.returncode

    def read_package(self, kt_path: Path) -> str:
        """
        Read the package declaration from a Kotlin source file.

        Args:
            kt_path (Path): Path to the .kt file.

        Returns:
            Package name if found, else an empty string.
        """
        with kt_path.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if m := self._pkg_re.match(line):
                    return m.group(1)
                if line.strip() and not line.lstrip().startswith("//"):
                    break
        return ""

    @staticmethod
    def get_root_name(class_file: Path) -> str:
        return class_file.stem.split("$")[0]

    @staticmethod
    def guess_kt_filename(root: str) -> str:
        base = root[:-2] if root.endswith("Kt") else root
        return (base[:1].lower() + base[1:] + ".kt").lower()

    def _index_one(self, kt_path: Path) -> Tuple[DirKey, PkgKey, Path]:
        repo = kt_path.relative_to(self.originals_root).parts[0]
        dir_parts = kt_path.relative_to(self.originals_root / repo).parent.parts
        dir_key: DirKey = (repo, dir_parts, kt_path.name.lower())

        pkg_str = self.read_package(kt_path)
        pkg_parts = tuple(pkg_str.split(".")) if pkg_str else ()
        pkg_key: PkgKey = (repo, pkg_parts, kt_path.name.lower())

        return dir_key, pkg_key, kt_path

    def index_kt_files(self) -> Tuple[Dict[DirKey, Path], Dict[PkgKey, List[Path]]]:
        idx_dir: Dict[DirKey, Path] = {}
        idx_pkg: Dict[PkgKey, List[Path]] = defaultdict(list)

        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as pool:
            tasks = (kt for kt in self.originals_root.rglob("*.kt"))
            for dir_key, pkg_key, kt_path in pool.map(
                    partial(self._index_one), tasks
            ):
                if dir_key in idx_dir:
                    raise ValueError(f"Duplicate file {kt_path}")
                idx_dir[dir_key] = kt_path
                idx_pkg[pkg_key].append(kt_path)

        return idx_dir, idx_pkg

    def build_pairs(self) -> Dict[Path, List[Path]]:
        idx_dir, idx_pkg = self.index_kt_files()
        pairs: Dict[Path, List[Path]] = defaultdict(list)
        skipped_prefixes = ("kotlin/", "kotlinx/", "java/", "javax/")

        for cls in self.bytecode_root.rglob("*.class"):
            p = cls.as_posix()
            if "/META-INF/" in p or any(
                    p.startswith(self.bytecode_root.joinpath(pref).as_posix())
                    for pref in skipped_prefixes
            ):
                continue

            repo, *pkg_dirs, _fname = cls.relative_to(self.bytecode_root).parts
            pkg_parts: Tuple[str, ...] = tuple(pkg_dirs)

            root = self.get_root_name(cls)
            kt_name = self.guess_kt_filename(root)
            pkg_key: PkgKey = (repo, pkg_parts, kt_name)

            candidates = idx_pkg.get(pkg_key, [])
            if len(candidates) > 1:
                raise ValueError(candidates)
            if len(candidates) != 1:
                continue

            kt_path = candidates[0]
            if cls in pairs[kt_path]:
                raise ValueError(f"Duplicate class {cls} for {kt_path}")
            pairs[kt_path].append(cls)

        return pairs

    def _build_record(self, task: Tuple[Path, List[Path]]) -> Optional[str]:
        kt_path, cls_list = task
        class_entries = []
        all_empty = True

        for c in cls_list:
            stdout, stderr, code = self.run_command(["javap", "-c", "-p", str(c)])
            if stdout.strip():
                all_empty = False
            class_entries.append({
                "class_path": str(c.relative_to(self.bytecode_root)),
                "javap": stdout,
                "javap_err": stderr
            })

        if all_empty:
            print(f"\nSkipped: {kt_path} — all javap outputs are empty.")
            return None

        record = {
            "kt_path": str(kt_path.relative_to(self.originals_root)),
            "kt_source": kt_path.read_text(encoding="utf-8", errors="ignore"),
            "classes": class_entries,
        }
        return json.dumps(record, ensure_ascii=False)

    def write_jsonl(self, pairs: Dict[Path, List[Path]], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tasks = [(kt_path, cls_list) for kt_path, cls_list in pairs.items() if cls_list]

        with out_path.open("w", encoding="utf-8") as fh, \
                ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() // 2) as pool:
            for json_line in tqdm(pool.map(self._build_record, tasks, chunksize=8),
                                  total=len(tasks),
                                  desc="javap"):
                if json_line is not None:
                    fh.write(json_line + "\n")

        print(f"Wrote {len(tasks)} records to {out_path}")


DirKey = Tuple[str, Tuple[str, ...], str]
PkgKey = Tuple[str, Tuple[str, ...], str]


def main() -> None:
    ds_root = Path(input("Path to dataset (must contain originals/ and bytecode/): ").strip()).resolve()
    if not (ds_root / "originals").is_dir() or not (ds_root / "bytecode").is_dir():
        raise SystemExit("originals/ or bytecode/ not found.")

    collector = BytecodePairCollector(ds_root)
    pairs = collector.build_pairs()
    print(f"Found {len(pairs)} .kt files with bytecode classes.")

    collector.write_jsonl(pairs, ds_root / "pairs.jsonl")


if __name__ == "__main__":
    main()
