import os
import random
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Tuple, Optional, Set

import requests
from tqdm.contrib.concurrent import process_map

MAVEN_SEARCH_URL = "https://search.maven.org/solrsearch/select"
KOTLIN_VERSION = "2.1.20"

INSERT_IMPORTS = [
    "import kotlin.math.*"
]

STANDARD_PREFIXES = (
    "kotlin.",
    "java.",
    "javax.",
)

from threading import Lock

log_lock = Lock()


def log_error_immediately(repo: Path, msg: str, bytecode_root: Path) -> None:
    with log_lock:
        with (bytecode_root / "compile_errors.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{repo}:\n{msg}\n\n")


def find_repositories(src_root: Path) -> List[Path]:
    print(f"Looking in: {src_root.absolute()}")
    return [d for d in src_root.iterdir() if d.is_dir()]


@lru_cache(maxsize=4096)
def resolve_artifact(package_prefix: str) -> Optional[str]:
    try:
        resp = requests.get(
            MAVEN_SEARCH_URL,
            params={"q": f"fc:{package_prefix}", "rows": 1, "wt": "json"},
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        docs = resp.json().get("response", {}).get("docs", [])
        if not docs:
            return None
        g = docs[0]["g"]
        a = docs[0]["a"]
        return f"{g}:{a}:+"
    except Exception:
        return None


def parse_kotlin_imports(file_path: Path) -> List[str]:
    imports: List[str] = []
    rx = re.compile(r"^\s*import\s+([\w.]+)")
    with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = rx.match(line)
            if m:
                imports.append(m.group(1))
    return imports


def guess_dependencies(imports: List[str]) -> Set[str]:
    deps: Set[str] = set()
    for imp in imports:
        if imp.startswith(STANDARD_PREFIXES):
            continue
        segments = imp.split(".")
        for i in range(len(segments), 1, -1):
            prefix = ".".join(segments[:i])
            coord = resolve_artifact(prefix)
            if coord:
                deps.add(coord)
                break
    return deps


def copy_with_auto_imports(src: Path, dest: Path) -> None:
    text = src.read_text(encoding="utf-8", errors="ignore")
    missing: List[str] = [imp for imp in INSERT_IMPORTS if imp not in text]
    if missing:
        lines = text.splitlines()
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("package "):
                insert_at = i + 1
            elif line.startswith("import "):
                insert_at = i
                break
        for imp in reversed(missing):
            lines.insert(insert_at, imp)
        text = "\n".join(lines)
    dest.write_text(text, encoding="utf-8")


def build_with_synthetic_gradle(repo_dir: Path, bytecode_root: Path) -> Tuple[bool, str]:
    kt_files = list(repo_dir.rglob("*.kt"))
    if not kt_files:
        return True, ""

    imports: List[str] = []
    for kt in kt_files:
        imports.extend(parse_kotlin_imports(kt))
    dependencies = guess_dependencies(imports)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src_root = tmp / "src/main/kotlin"
        src_root.mkdir(parents=True, exist_ok=True)

        for kt in kt_files:
            rel = kt.relative_to(repo_dir)
            dst_kt = src_root / rel
            dst_kt.parent.mkdir(parents=True, exist_ok=True)
            copy_with_auto_imports(kt, dst_kt)

        (tmp / "settings.gradle.kts").write_text("rootProject.name = \"synthetic\"\n", encoding="utf-8")
        deps_block = "\n".join(
            f"    implementation(\"{c}\")" for c in sorted(dependencies)) or "    // no external deps"
        (tmp / "build.gradle.kts").write_text(f"""
plugins {{
    kotlin("jvm") version "{KOTLIN_VERSION}"
}}
repositories {{
    mavenCentral()
}}
tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {{
    kotlinOptions.jvmTarget = "23"
}}
dependencies {{
{deps_block}
}}
""", encoding="utf-8")

        result = subprocess.run(["gradle", "--no-daemon", "assemble", "-x", "test"], cwd=tmp, capture_output=True,
                                text=True)
        if result.returncode != 0:
            return False, result.stderr

        target_dir = bytecode_root / repo_dir.name
        for f in (tmp / "build/classes").rglob("*.class"):
            rel = f.relative_to(tmp / "build/classes")
            out_f = target_dir / rel
            out_f.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out_f)
        return True, ""


def manual_kotlinc_compile(repo_dir: Path, target_dir: Path) -> Tuple[bool, str]:
    kt_files = list(repo_dir.rglob("*.kt"))
    if not kt_files:
        return True, ""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir) / "out"
        tmp_out.mkdir()
        cmd = ["kotlinc", "-jvm-target", "23", *map(str, kt_files), "-d", str(tmp_out)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return False, res.stderr
        for root, _, files in os.walk(tmp_out):
            for fn in files:
                if fn.endswith(".class"):
                    src_f = Path(root) / fn
                    rel = src_f.relative_to(tmp_out)
                    dst_f = target_dir / rel
                    dst_f.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_f, dst_f)
        return True, ""


def copy_bytecode(source_dir: Path, target_dir: Path) -> None:
    for root, _, files in os.walk(source_dir):
        for fn in files:
            if fn.endswith(".class"):
                src_f = Path(root) / fn
                rel = src_f.relative_to(source_dir)
                dst_f = target_dir / rel
                dst_f.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_f, dst_f)


def compile_repository(repo_dir: Path, bytecode_root: Path) -> Tuple[str, Optional[str]]:
    target_dir = bytecode_root / repo_dir.name
    if target_dir.exists():
        return repo_dir.name, None
    ok, err = manual_kotlinc_compile(repo_dir, target_dir)
    if ok:
        return repo_dir.name, None
    ok, err = build_with_synthetic_gradle(repo_dir, bytecode_root)
    if ok:
        return repo_dir.name, None

    return repo_dir.name, err


def log_errors(errors: List[Tuple[Path, str]], bytecode_root: Path) -> None:
    if not errors:
        return
    with (bytecode_root / "compile_errors.log").open("w", encoding="utf-8") as fh:
        for repo, msg in errors:
            fh.write(f"{repo}:\n{msg}\n\n")


def compile_task(args: Tuple[Path, Path]) -> None:
    repo_dir, bytecode_root = args
    name, err = compile_repository(repo_dir, bytecode_root)
    if err:
        log_error_immediately(repo_dir, err, bytecode_root)


def main() -> None:
    dataset = input("Path to dataset: ").strip()
    src = Path(f"{dataset}/originals")
    dst = Path(f"{dataset}/bytecode")

    repos = find_repositories(src)
    print(f"Found {len(repos)} repositories.")

    tasks = [
        (repo, dst)
        for repo in repos
        if not (dst / repo.name).exists()
    ]

    random.shuffle(tasks)

    print(f"Found {len(tasks)} tasks to compile.")

    process_map(
        compile_task,
        tasks,
        max_workers=cpu_count() - 1,
        chunksize=1,
        desc="Compiling"
    )

    print("Compilation finished.")
    print(f"See compile_errors.log for errors.")


if __name__ == "__main__":
    main()