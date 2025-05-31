import contextlib
import json
from unittest import mock

import pytest

from src.main.collect.process_models.process_model import ModelProcessor


class DummyTokenizer:
    pad_token = None
    eos_token_id = 42
    eos_token = "<eos>"
    padding_side = "left"

    def __call__(self, prompts, return_tensors=None, padding=None, truncation=None):
        class DummyEnc(dict):
            def __init__(self, batch_size):
                super().__init__()
                tensor = mock.Mock(shape=(batch_size, 5))
                self["input_ids"] = tensor
                self.input_ids = tensor

            def to(self, device):
                return self

        return DummyEnc(len(prompts))

    def batch_decode(self, outputs, skip_special_tokens=None):
        return ["GEN1", "GEN2"][: outputs.shape[0]]

    def apply_chat_template(self, tmpl, tokenize, add_generation_prompt):
        # Used by Qwen models
        return f"CHAT:{tmpl[0]['content']}"


class DummyModel:
    device = "cpu"

    def eval(self):
        return self

    def generate(self, **kwargs):
        class DummyOut:
            def __getitem__(self, idx):
                # Return a dummy tensor
                import numpy as np

                return np.array([[0, 1, 2, 3, 4, 5, 6]])

            shape = (2, 7)

        import numpy as np

        return np.array([[10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24, 25, 26]])


@contextlib.contextmanager
def fake_cm(*a, **k):
    yield


@pytest.fixture
def processor(tmp_path, monkeypatch):
    # Patch Config and Row so we can control dataset and config values
    class DummyConfig:
        dataset_name = "test_ds"
        split = "train"
        out_dir = tmp_path
        dataset_size = 2
        flush_every = 1
        est_scale = 1.0
        quant_4bit = None
        quant_8bit = None

    class DummyRow:
        def __init__(self, kt_path, kt_source, bytecode):
            self.kt_path = kt_path
            self.kt_source = kt_source
            self.bytecode = bytecode

    monkeypatch.setattr("main.collect.process_models.process_model.Config", DummyConfig)
    monkeypatch.setattr("main.collect.process_models.process_model.Row", DummyRow)
    return ModelProcessor("dummy/model")


def test_build_prompt_regular(processor):
    # Should make a regular prompt if model doesn't start with Qwen/
    tok = DummyTokenizer()
    bytecode = "BC"
    out = processor.build_prompt("notQwen/model", bytecode, tok)
    assert "Byte‑code" in out and "Kotlin" in out


def test_build_prompt_qwen(processor):
    tok = DummyTokenizer()
    bytecode = "QWENBYTE"
    out = processor.build_prompt("Qwen/fake", bytecode, tok)
    assert out.startswith("CHAT:")


def test_load_rows(monkeypatch, processor):
    # Patch load_dataset to yield test data
    dummy_data = [
        {"kt_path": "a.kt", "kt_source": "origA", "foo": 1},
        {"kt_path": "b.kt", "kt_source": "origB", "foo": 2},
    ]
    monkeypatch.setattr(
        "main.collect.process_models.process_model.load_dataset",
        lambda n, split, streaming: dummy_data,
    )
    monkeypatch.setattr(
        "main.collect.process_models.process_model.to_bytecode", lambda r: "BYTECODE"
    )
    rows = processor.load_rows()
    assert rows[0].kt_path == "a.kt"
    assert rows[0].bytecode == "BYTECODE"


def test_hf_generate(monkeypatch, processor):
    model = DummyModel()
    tokenizer = DummyTokenizer()
    fake_torch = mock.Mock()
    fake_torch.inference_mode = fake_cm
    fake_torch.amp = mock.Mock()
    fake_torch.amp.autocast = fake_cm
    monkeypatch.setattr("main.collect.process_models.process_model.torch", fake_torch)
    prompts = ["Prompt1", "Prompt2"]
    outs = processor._hf_generate(model, tokenizer, prompts, max_new=3)
    assert isinstance(outs, list) and len(outs) == 2


def test_load_model(monkeypatch, processor):
    # Patch AutoModelForCausalLM
    dummy_model = DummyModel()
    monkeypatch.setattr(
        "main.collect.process_models.process_model.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: dummy_model,
    )
    monkeypatch.setattr("main.collect.process_models.process_model.torch", mock.Mock())
    m = processor.load_model()
    assert m is dummy_model


def test_unload_model(monkeypatch):
    # Patch torch
    dummy_torch = mock.Mock()
    dummy_torch.cuda.is_available.return_value = False
    monkeypatch.setattr("main.collect.process_models.process_model.torch", dummy_torch)
    ModelProcessor.unload_model("dummy_model", "dummy_tokenizer")  # Should not error


def test_process(monkeypatch, processor, tmp_path):
    # Patch everything: dataset, model, tokenizer, gen, file writes, helper utils
    processor.cfg.dataset_size = 2
    processor.cfg.flush_every = 1
    processor.output_file = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        "main.collect.process_models.process_model.extract_kotlin", lambda x: x
    )
    monkeypatch.setattr(
        "main.collect.process_models.process_model.to_bytecode",
        lambda r: r["kt_path"] + "_BC",
    )
    monkeypatch.setattr(
        "main.collect.process_models.process_model.gen_len_stats",
        lambda rows, tok: (5, 0),
    )
    monkeypatch.setattr(
        "main.collect.process_models.process_model.model_batch_size",
        lambda model, scale: 1,
    )
    monkeypatch.setattr(
        "main.collect.process_models.process_model.AutoTokenizer.from_pretrained",
        lambda n, **k: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "main.collect.process_models.process_model.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: DummyModel(),
    )
    dummy_data = [
        {"kt_path": "a.kt", "kt_source": "origA"},
        {"kt_path": "b.kt", "kt_source": "origB"},
    ]
    monkeypatch.setattr(
        "main.collect.process_models.process_model.load_dataset",
        lambda n, split, streaming: dummy_data,
    )
    # Don't actually run tqdm for speed
    monkeypatch.setattr(
        "main.collect.process_models.process_model.tqdm", lambda x, desc=None: x
    )
    processor.process()
    lines = [json.loads(line) for line in (tmp_path / "out.jsonl").open()]
    assert {"kt_path": "a.kt", "model": "GEN1"} in lines
    assert {"kt_path": "b.kt", "model": "GEN1"} in lines
