"""Describe model registry and runner helpers: the 2026-11 refresh."""
import importlib.util
from pathlib import Path

from core import describe_models

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_describe_models_refreshed():
    repos = [r for _, r, _, _ in describe_models.DESCRIBE_MODELS]
    assert repos[0] == "fancyfeast/llama-joycaption-beta-one-hf-llava"
    assert "Qwen/Qwen3.5-4B" in repos and "Qwen/Qwen3.5-2B" in repos
    assert not any("Qwen2.5" in r or "SmolVLM" in r for r in repos)


def test_strip_think():
    run = _load("joycaption_run")
    assert run.strip_think("<think>\nhmm\n</think>\n\nA lighthouse.") == "A lighthouse."
    assert run.strip_think("<think></think>A cat.") == "A cat."
    assert run.strip_think("A plain caption.") == "A plain caption."
