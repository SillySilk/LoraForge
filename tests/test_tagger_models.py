"""Tagger registry and PixAI runner helpers: the 2026-11 refresh."""
import importlib.util
from pathlib import Path

import torch

from core.tagger import PIXAI_REPOS, TAGGER_MODELS, TaggerProcess, is_pixai

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tagger_list_has_no_keras_v2_models():
    repos = [r for _, r, _ in TAGGER_MODELS]
    assert not any("v1-4" in r for r in repos)
    assert "SmilingWolf/wd-eva02-large-tagger-v3" in repos
    assert "pixai-labs/pixai-tagger-v1.0" in repos


def test_tagger_default_and_slot_count_unchanged():
    # Saved settings are indexes; slot 0 stays the default and no slot vanished.
    assert TAGGER_MODELS[0][1] == "SmilingWolf/wd-swinv2-tagger-v3"
    assert len(TAGGER_MODELS) == 4


def test_only_wd_models_use_onnx():
    for _, repo, onnx in TAGGER_MODELS:
        assert onnx == (not is_pixai(repo))


def test_is_pixai():
    assert is_pixai("pixai-labs/pixai-tagger-v1.0")
    assert is_pixai("  pixai-labs/pixai-tagger-v1.0 ")
    assert not is_pixai("SmilingWolf/wd-swinv2-tagger-v3")
    assert not is_pixai("")
    assert PIXAI_REPOS


def test_pixai_args_route_to_pixai_runner():
    args = TaggerProcess._pixai_args("C:/set", "pixai-labs/pixai-tagger-v1.0", 0.35, 8,
                                     "only.txt", True)
    assert Path(args[0]).name == "pixai_tag_run.py"
    assert "--thresh=0.35" in args and "--only=only.txt" in args
    assert "--skip-existing" in args
    assert not any(a.startswith("--sdscripts") for a in args)


def test_pixai_threshold_is_offset_from_wd_default():
    run = _load("pixai_tag_run")
    base = torch.tensor([0.17, 0.27])
    assert torch.allclose(run.offset_thresholds(base, 0.35), base)
    assert torch.allclose(run.offset_thresholds(base, 0.45), base + 0.10)
    assert run.offset_thresholds(torch.tensor([0.05]), 0.0).item() > 0.009


def test_pixai_tag_format_matches_wd_runner():
    run = _load("pixai_tag_run")
    assert run.format_tags(["1girl", "long_hair", "blue_eyes"]) == "1girl, long hair, blue eyes"
    assert run.KEEP_CATEGORIES == ("character", "general")
