from unittest import mock

import pytest

from main.collect.process_models.compile_models import CompileModels


@pytest.fixture
def fake_dataset(tmp_path):
    # Creates the following structure:
    # tmp_path/
    #   modelA/originals/repo1/
    #   modelA/originals/repo2/
    #   modelB/originals/
    #   modelC/   (no originals)
    (tmp_path / "modelA" / "originals" / "repo1").mkdir(parents=True)
    (tmp_path / "modelA" / "originals" / "repo2").mkdir(parents=True)
    (tmp_path / "modelB" / "originals").mkdir(parents=True)
    (tmp_path / "modelC").mkdir()
    return tmp_path


def test_find_models_lists_only_with_originals(fake_dataset):
    cm = CompileModels(fake_dataset)
    found = set(cm.find_models())
    # Only modelA and modelB have "originals" directories
    assert "modelA" in found
    assert "modelB" in found
    assert "modelC" not in found


def test_compile_model_invokes_compiler(monkeypatch, fake_dataset):
    cm = CompileModels(fake_dataset)
    # Prepare modelA with two repos in originals
    model = "modelA"
    src = fake_dataset / model / "originals"
    dst = fake_dataset / model / "bytecode"
    # Patch KotlinBytecodeCompiler.find_repositories to just return the two repos
    dummy_repos = [src / "repo1", src / "repo2"]
    monkeypatch.setattr(
        "main.collect.bytecode.kotlin_bytecode_compiler.KotlinBytecodeCompiler.find_repositories",
        mock.Mock(return_value=dummy_repos)
    )
    # Patch process_map so it doesn't run parallel code, just checks arguments
    called = {}

    def fake_process_map(func, tasks, **kwargs):
        called["tasks"] = tasks
        return [None for _ in tasks]

    monkeypatch.setattr("main.collect.process_models.compile_models.process_map", fake_process_map)
    # Patch cpu_count to 2 for determinism
    monkeypatch.setattr("main.collect.process_models.compile_models.cpu_count", mock.Mock(return_value=2))
    # Patch compile_task so nothing is actually run
    monkeypatch.setattr(
        "main.collect.bytecode.kotlin_bytecode_compiler.KotlinBytecodeCompiler.compile_task",
        mock.Mock()
    )
    # Create a dummy bytecode dir for repo1 to simulate partial work
    (dst / "repo1").mkdir(parents=True)
    cm.compile_model(model)
    # Only repo2 should be in the "tasks" list (repo1/ already has bytecode)
    tasks = called["tasks"]
    assert len(tasks) == 1
    assert tasks[0][0].name == "repo2"
    assert tasks[0][1] == dst


def test_run_compiles_all_models(monkeypatch, fake_dataset):
    cm = CompileModels(fake_dataset)
    # Patch find_models to control exactly what is found
    monkeypatch.setattr(cm, "find_models", mock.Mock(return_value=["modelA", "modelB"]))
    called_models = []
    monkeypatch.setattr(cm, "compile_model", lambda m: called_models.append(m))
    cm.run()
    # Should call compile_model for both
    assert set(called_models) == {"modelA", "modelB"}
