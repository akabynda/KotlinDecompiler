import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple


def find_repositories(src_root: Path) -> List[Path]:
    return [d for d in src_root.iterdir() if d.is_dir()]


def compile_repository(
        repo_dir: Path,
        bytecode_root: Path
) -> tuple[str, None] | tuple[str, str]:
    kt_files = list(repo_dir.rglob("*.kt"))
    if not kt_files:
        return repo_dir.name, None

    target_dir = bytecode_root / repo_dir.name
    if target_dir.exists():
        return repo_dir.name, None

    # print("Compiling", repo_dir)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir) / "out"
        tmp_out.mkdir()
        cmd = ["kotlinc", *map(str, kt_files), "-d", str(tmp_out)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return repo_dir.name, result.stderr

        copy_bytecode(tmp_out, bytecode_root / repo_dir.name)

    return repo_dir.name, None


def copy_bytecode(source_dir: Path, target_dir: Path) -> None:
    for root, _, files in os.walk(source_dir):
        rel = Path(root).relative_to(source_dir)
        dest_dir = target_dir / rel
        dest_dir.mkdir(parents=True, exist_ok=True)
        for file_name in files:
            if file_name.endswith(".class"):
                shutil.copy2(Path(root) / file_name, dest_dir / file_name)


def log_errors(errors: List[Tuple[Path, str]], bytecode_root: Path) -> None:
    if not errors:
        return
    log_path = bytecode_root / "compile_errors.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        for repo_name, error_msg in errors:
            log_file.write(f"{repo_name}:\n{error_msg}\n\n")


def main() -> None:
    dataset_name = input("Name of dataset:").lower()
    src = Path(f"./{dataset_name}/originals")
    dst = Path(f"./{dataset_name}/bytecode")

    src.mkdir(parents=True, exist_ok=True)
    dst.mkdir(parents=True, exist_ok=True)

    repos = find_repositories(src)
    print(f"Found {len(repos)} repositories to compile.")

    num_workers = cpu_count() - 1
    with Pool(processes=num_workers) as pool:
        args = [(repo, dst) for repo in repos]
        results = pool.starmap(compile_repository, args)

    errors = [(Path(name), err) for name, err in results if err]
    log_errors(errors, dst)

    print("Compilation finished.")
    if errors:
        print(f"Errors occurred in {len(errors)} repositories. See {dst / 'compile_errors.log'}")


if __name__ == "__main__":
    main()
