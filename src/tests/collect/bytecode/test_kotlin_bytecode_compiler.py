from pathlib import Path

import pytest

from src.main.collect.bytecode.kotlin_bytecode_compiler import KotlinBytecodeCompiler


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temp repo with a .kt file and return its path."""
    repo = tmp_path / "originals" / "repo1"
    repo.mkdir(parents=True)
    kt = repo / "Main.kt"
    kt.write_text("package foo\nimport kotlin.math.PI\nfun main() {}")
    return repo


def test_init_sets_paths(tmp_path):
    c = KotlinBytecodeCompiler(tmp_path)
    assert c.dataset_path == tmp_path
    assert c.source_root == tmp_path / "originals"
    assert c.bytecode_root == tmp_path / "bytecode"


def test_find_repositories(tmp_path):
    (tmp_path / "originals" / "repoA").mkdir(parents=True)
    (tmp_path / "originals" / "repoB").mkdir(parents=True)
    c = KotlinBytecodeCompiler(tmp_path)
    repos = c.find_repositories()
    assert set(r.name for r in repos) == {"repoA", "repoB"}


def test_parse_kotlin_imports(tmp_path):
    f = tmp_path / "test.kt"
    f.write_text("import kotlin.math.PI\nimport abc.def\nfun foo() {}\n")
    imports = KotlinBytecodeCompiler.parse_kotlin_imports(f)
    assert imports == ["kotlin.math.PI", "abc.def"]


def test_guess_dependencies_skips_standard(monkeypatch):
    c = KotlinBytecodeCompiler(Path("/tmp/foo"))
    # Patch resolve_artifact to a known result
    monkeypatch.setattr(c, "resolve_artifact", lambda p: "g:a:+" if p == "abc.def" else None)
    deps = c.guess_dependencies(["kotlin.math.PI", "abc.def", "java.io.File"])
    assert "g:a:+" in deps
    assert len(deps) == 1


def test_copy_with_auto_imports_inserts_missing(tmp_path):
    src = tmp_path / "a.kt"
    dest = tmp_path / "b.kt"
    src.write_text("fun x() = 1\n")
    KotlinBytecodeCompiler.copy_with_auto_imports(src, dest)
    out = dest.read_text()
    assert "import kotlin.math.*" in out
    assert "fun x() = 1" in out


def test_copy_with_auto_imports_does_not_duplicate(tmp_path):
    src = tmp_path / "c.kt"
    dest = tmp_path / "d.kt"
    src.write_text("import kotlin.math.*\nfun x() = 2")
    KotlinBytecodeCompiler.copy_with_auto_imports(src, dest)
    out = dest.read_text()
    # Should not insert duplicate import
    assert out.count("import kotlin.math.*") == 1


def test_build_with_gradle_handles_no_kt_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    c = KotlinBytecodeCompiler(tmp_path)
    ok, err = c.build_with_gradle(repo)
    assert ok and err == ""


def test_manual_kotlinc_compile_handles_no_kt(tmp_path):
    repo = tmp_path / "repo2"
    repo.mkdir()
    out = tmp_path / "bytecode"
    out.mkdir()
    ok, err = KotlinBytecodeCompiler.manual_kotlinc_compile(repo, out)
    assert ok and err == ""


def test_manual_kotlinc_compile_runs_subprocess(monkeypatch, tmp_repo, tmp_path):
    # Patch subprocess to simulate success
    def fake_run(cmd, capture_output, text):
        (Path(cmd[-1]) / "Dummy.class").parent.mkdir(parents=True, exist_ok=True)
        (Path(cmd[-1]) / "Dummy.class").write_text("dummy")

        class Res: returncode = 0; stderr = ""

        return Res()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("os.walk", lambda d: [(d, [], ["Dummy.class"])])
    out = tmp_path / "bytecode"
    out.mkdir()
    ok, err = KotlinBytecodeCompiler.manual_kotlinc_compile(tmp_repo, out)
    assert ok and err == ""
    # File copied
    assert (out / "Dummy.class").exists()


def test_resolve_artifact_success(monkeypatch):
    class Resp:
        status_code = 200

        def json(self): return {"response": {"docs": [{"g": "g", "a": "a"}]}}

    monkeypatch.setattr("requests.get", lambda *a, **k: Resp())
    KotlinBytecodeCompiler.resolve_artifact.cache_clear()
    out = KotlinBytecodeCompiler.resolve_artifact("success-case")
    assert out == "g:a:+"


def test_resolve_artifact_failure(monkeypatch):
    class Resp:
        status_code = 404

        def json(self): return {}

    monkeypatch.setattr("requests.get", lambda *a, **k: Resp())
    KotlinBytecodeCompiler.resolve_artifact.cache_clear()
    out = KotlinBytecodeCompiler.resolve_artifact("fail-case")
    assert out is None


def test_compile_repository_prefers_manual(tmp_repo, tmp_path, monkeypatch):
    c = KotlinBytecodeCompiler(tmp_path)
    # Patch manual_kotlinc_compile to succeed
    monkeypatch.setattr(c, "manual_kotlinc_compile", lambda repo, out: (True, ""))
    name, err = c.compile_repository(tmp_repo)
    assert name == tmp_repo.name and err is None


def test_compile_repository_fallbacks(tmp_repo, tmp_path, monkeypatch):
    c = KotlinBytecodeCompiler(tmp_path)
    # Fail manual, succeed gradle
    monkeypatch.setattr(c, "manual_kotlinc_compile", lambda repo, out: (False, "manual fail"))
    monkeypatch.setattr(c, "build_with_gradle", lambda repo: (True, ""))
    name, err = c.compile_repository(tmp_repo)
    assert name == tmp_repo.name and err is None


def test_compile_repository_all_fail(tmp_repo, tmp_path, monkeypatch):
    c = KotlinBytecodeCompiler(tmp_path)
    monkeypatch.setattr(c, "manual_kotlinc_compile", lambda repo, out: (False, "manual fail"))
    monkeypatch.setattr(c, "build_with_gradle", lambda repo: (False, "gradle fail"))
    name, err = c.compile_repository(tmp_repo)
    assert name == tmp_repo.name and err == "gradle fail"


def test_log_error_creates_log(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    KotlinBytecodeCompiler.log_error(repo, "fail here")
    log = repo.parent / "compile_errors.log"
    assert log.exists()
    txt = log.read_text()
    assert "fail here" in txt


def test_process_all(monkeypatch, tmp_path):
    c = KotlinBytecodeCompiler(tmp_path)
    (tmp_path / "originals" / "r1").mkdir(parents=True)
    (tmp_path / "originals" / "r2").mkdir(parents=True)
    monkeypatch.setattr(c, "find_repositories", lambda: [tmp_path / "originals" / "r1"])

    def fake_process_map(func, tasks, **kwargs):
        for t in tasks:
            func(t)
        return [None for _ in tasks]

    monkeypatch.setattr("main.collect.bytecode.kotlin_bytecode_compiler.process_map", fake_process_map)
    c.process_all()
