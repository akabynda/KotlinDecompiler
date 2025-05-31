import os
import random
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from multiprocessing import cpu_count
from pathlib import Path
from threading import Lock
from typing import List, Optional, Set, Tuple

import requests
from tqdm.contrib.concurrent import process_map

MAVEN_SEARCH_URL = "https://search.maven.org/solrsearch/select"
KOTLIN_VERSION = "2.1.20"
INSERT_IMPORTS: List[str] = ["import kotlin.math.*"]
STANDARD_PREFIXES: Tuple[str, ...] = ("kotlin.", "java.", "javax.")
log_lock = Lock()


class KotlinBytecodeCompiler:
    """
    Compiles Kotlin repositories into bytecode using kotlinc or synthetic Gradle.
    """

    def __init__(self, dataset_path: Path) -> None:
        """
        Initialize the compiler.

        Args:
            dataset_path (Path): Root directory of the dataset.
        """
        self.dataset_path: Path = dataset_path
        self.source_root: Path = dataset_path / "originals"
        self.bytecode_root: Path = dataset_path / "bytecode"

    def find_repositories(self) -> List[Path]:
        """
        Find all Kotlin repositories.

        Returns:
            List of repository paths.
        """
        print(f"Looking in: {self.source_root.resolve()}")
        return [d for d in self.source_root.iterdir() if d.is_dir()]

    @staticmethod
    @lru_cache(maxsize=4096)
    def resolve_artifact(package_prefix: str) -> Optional[str]:
        """
        Resolve a Maven artifact for a given package prefix.

        Args:
            package_prefix (str): Package prefix.

        Returns:
            Optional Maven coordinate.
        """
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

    @staticmethod
    def parse_kotlin_imports(file_path: Path) -> List[str]:
        """
        Parse import statements from a Kotlin file.

        Args:
            file_path (Path): Path to the .kt file.

        Returns:
            List of import statements.
        """
        rx = re.compile(r"^\s*import\s+([\w.]+)")
        with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
            return [m.group(1) for line in fh if (m := rx.match(line))]

    def guess_dependencies(self, imports: List[str]) -> Set[str]:
        """
        Guess Maven dependencies from imports.

        Args:
            imports (List[str]): List of import statements.

        Returns:
            Set of Maven coordinates.
        """
        deps: Set[str] = set()
        for imp in imports:
            if imp.startswith(STANDARD_PREFIXES):
                continue
            segments = imp.split(".")
            for i in range(len(segments), 1, -1):
                prefix = ".".join(segments[:i])
                coord = self.resolve_artifact(prefix)
                if coord:
                    deps.add(coord)
                    break
        return deps

    @staticmethod
    def copy_with_auto_imports(src: Path, dest: Path) -> None:
        """
        Copy a Kotlin file, adding missing auto-imports if needed.

        Args:
            src (Path): Source file.
            dest (Path): Destination file.
        """
        text = src.read_text(encoding="utf-8", errors="ignore")
        missing = [imp for imp in INSERT_IMPORTS if imp not in text]
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

    def build_with_gradle(self, repo_dir: Path) -> Tuple[bool, str]:
        """
        Build repository bytecode with synthetic Gradle.

        Args:
            repo_dir (Path): Repository directory.

        Returns:
            Success status and error message (if any).
        """
        kt_files = list(repo_dir.rglob("*.kt"))
        if not kt_files:
            return True, ""

        imports = [imp for kt in kt_files for imp in self.parse_kotlin_imports(kt)]
        dependencies = self.guess_dependencies(imports)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src_root = tmp / "src/main/kotlin"
            src_root.mkdir(parents=True, exist_ok=True)

            for kt in kt_files:
                rel = kt.relative_to(repo_dir)
                dst_kt = src_root / rel
                dst_kt.parent.mkdir(parents=True, exist_ok=True)
                self.copy_with_auto_imports(kt, dst_kt)

            (tmp / "settings.gradle.kts").write_text(
                'rootProject.name = "synthetic"\n', encoding="utf-8"
            )
            deps_block = (
                "\n".join(f'    implementation("{c}")' for c in sorted(dependencies))
                or "    // no external deps"
            )
            (tmp / "build.gradle.kts").write_text(
                f"""
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
""",
                encoding="utf-8",
            )

            result = subprocess.run(
                ["gradle", "--no-daemon", "assemble", "-x", "test"],
                cwd=tmp,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return False, result.stderr

            target_dir = self.bytecode_root / repo_dir.name
            for f in (tmp / "build/classes").rglob("*.class"):
                rel = f.relative_to(tmp / "build/classes")
                out_f = target_dir / rel
                out_f.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out_f)

            return True, ""

    @staticmethod
    def manual_kotlinc_compile(repo_dir: Path, target_dir: Path) -> Tuple[bool, str]:
        """
        Compile Kotlin files manually with kotlinc.

        Args:
            repo_dir (Path): Repository directory.
            target_dir (Path): Output bytecode directory.

        Returns:
            Success status and error message (if any).
        """
        kt_files = list(repo_dir.rglob("*.kt"))
        if not kt_files:
            return True, ""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir) / "out"
            tmp_out.mkdir()
            cmd = [
                "kotlinc",
                "-jvm-target",
                "23",
                *map(str, kt_files),
                "-d",
                str(tmp_out),
            ]
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

    def compile_repository(self, repo_dir: Path) -> Tuple[str, Optional[str]]:
        """
        Compile a single repository.

        Args:
            repo_dir (Path): Repository directory.

        Returns:
            Repository name and optional error message.
        """
        target_dir = self.bytecode_root / repo_dir.name
        if target_dir.exists():
            return repo_dir.name, None

        ok, err = self.manual_kotlinc_compile(repo_dir, target_dir)
        if ok:
            return repo_dir.name, None

        ok, err = self.build_with_gradle(repo_dir)
        if ok:
            return repo_dir.name, None

        return repo_dir.name, err

    def compile_task(self, args: Tuple[Path, Path]) -> None:
        """
        Worker task for compilation.

        Args:
            args (Tuple[Path, Path]): Repository and output directories.
        """
        repo_dir, _ = args
        name, err = self.compile_repository(repo_dir)
        if err:
            self.log_error(repo_dir, err)

    @staticmethod
    def log_error(repo: Path, msg: str) -> None:
        """
        Log a compilation error to a shared log file.

        Args:
            repo (Path): Repository path.
            msg (str): Error message.
        """
        with log_lock:
            with (repo.parent / "compile_errors.log").open("a", encoding="utf-8") as fh:
                fh.write(f"{repo}:\n{msg}\n\n")

    def process_all(self) -> None:
        """
        Find, shuffle, and compile all repositories.
        """
        repos = self.find_repositories()
        print(f"Found {len(repos)} repositories.")

        tasks = [
            (repo, self.bytecode_root)
            for repo in repos
            if not (self.bytecode_root / repo.name).exists()
        ]
        random.shuffle(tasks)
        print(f"Found {len(tasks)} tasks to compile.")

        process_map(
            self.compile_task,
            tasks,
            max_workers=cpu_count() - 1,
            chunksize=1,
            desc="Compiling",
        )

        print("Compilation finished. See compile_errors.log for errors.")


def main() -> None:
    dataset_path = Path(input("Path to dataset: ").strip())
    compiler = KotlinBytecodeCompiler(dataset_path)
    compiler.process_all()


if __name__ == "__main__":
    main()
