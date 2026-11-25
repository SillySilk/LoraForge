"""Tag images with PixAI Tagger v1.0 and write .tags sidecars.

The PixAI tagger is a transformers model (custom code shipped in its repo), not a
WD14 ONNX model, so sd-scripts' tagger can't run it. Output matches what
scripts/wd14_tag_run.py produces, so the rest of the caption chain can't tell the
two apart: comma-separated tags, underscores turned into spaces, character tags
first, then general tags, and a "[Tagger] (n/N) name: tags" line per image.

Thresholds: PixAI ships a tuned threshold per tag. The app's tagger threshold
(0.35 by default, tuned for WD) is applied as an OFFSET from that default, so the
untouched slider gives PixAI's own recommended settings and moving it up or down
makes PixAI stricter or looser by the same amount.

The model downloads into the standard Hugging Face cache on first use (~2 GB).
"""
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WD_DEFAULT_THRESH = 0.35
# Only these categories go into a training caption. Style / copyright / meta /
# rating tags would teach the LoRA artist names and quality labels.
KEEP_CATEGORIES = ("character", "general")


def log(msg: str):
    print(msg, flush=True)


def offset_thresholds(base, thresh: float):
    """Per-tag thresholds shifted by how far `thresh` sits from the WD default."""
    return (base + (thresh - WD_DEFAULT_THRESH)).clamp(0.01, 0.99)


def ensure_timm() -> bool:
    """PixAI's model code imports timm. Installs made before this tagger existed
    lack it, so install it once into this same environment instead of failing."""
    try:
        import timm  # noqa: F401
        return True
    except ImportError:
        pass
    import subprocess
    log("[Tagger] Installing timm (one-time, needed by the PixAI tagger)…")
    rc = subprocess.run([sys.executable, "-m", "pip", "install", "timm"]).returncode
    if rc != 0:
        log("[Tagger] ERROR: could not install timm. Re-run install.bat, then try again.")
        return False
    return True


def format_tags(tags) -> str:
    return ", ".join(t.replace("_", " ") for t in tags)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_folder")
    ap.add_argument("--repo_id", default="pixai-labs/pixai-tagger-v1.0")
    ap.add_argument("--thresh", type=float, default=WD_DEFAULT_THRESH)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    a = ap.parse_args()

    from core.dataset_manager import SUPPORTED_EXTENSIONS
    if a.only:
        todo = [ln.strip() for ln in Path(a.only).read_text(encoding="utf-8").splitlines()
                if ln.strip()]
    else:
        todo = [str(p) for p in sorted(Path(a.image_folder).iterdir())
                if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if a.skip_existing:
        def _tagged(p):
            try:
                return bool(Path(p).with_suffix(".tags").read_text(encoding="utf-8").strip())
            except (OSError, UnicodeDecodeError):
                return False
        todo = [p for p in todo if not _tagged(p)]
    if not todo:
        log("[Tagger] every image already tagged — nothing to do.")
        return 0

    if not ensure_timm():
        return 1
    log(f"[Tagger] Loading {a.repo_id} (first run downloads ~2 GB)…")
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
        model = AutoModel.from_pretrained(a.repo_id, trust_remote_code=True,
                                          torch_dtype=torch.bfloat16)
        processor = AutoImageProcessor.from_pretrained(a.repo_id, trust_remote_code=True)
    except Exception:
        log(f"[Tagger] ERROR loading {a.repo_id}:")
        log(traceback.format_exc())
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    cfg = model.config
    tags = cfg.tags
    if cfg.tags_best_threshold is not None:
        base = torch.tensor(cfg.tags_best_threshold, dtype=torch.float32)
    else:
        base = torch.cat([torch.full([n], float(cfg.category_best_threshold[c]))
                          for c, n in cfg.tags_split])
    thresholds = offset_thresholds(base, a.thresh).to(device)
    # Tag index ranges per category, in the order the model emits them.
    ranges, st = {}, 0
    for cat, n in cfg.tags_split:
        ranges[cat] = (st, st + n)
        st += n
    log(f"[Tagger] Model ready on {device}. Tagging {len(todo)} image(s)…")

    done = 0
    for i in range(0, len(todo), max(1, a.batch_size)):
        batch = todo[i:i + a.batch_size]
        images, kept = [], []
        for p in batch:
            try:
                images.append(Image.open(p))
                kept.append(p)
            except Exception as e:
                log(f"[Tagger] ERROR opening {Path(p).name}: {e}")
        if not images:
            continue
        try:
            pixels = processor(images)["pixel_values"].to(device, torch.bfloat16)
            with torch.no_grad():
                probs = model(pixels).float().sigmoid()
        except Exception:
            log("[Tagger] ERROR during tagging:")
            log(traceback.format_exc())
            return 1
        for p, prob in zip(kept, probs):
            picked = []
            for cat in KEEP_CATEGORIES:
                if cat not in ranges:
                    continue
                lo, hi = ranges[cat]
                hit = (prob[lo:hi] > thresholds[lo:hi]).nonzero().flatten()
                order = prob[lo:hi][hit].argsort(descending=True)
                picked += [tags[lo + int(j)] for j in hit[order]]
            line = format_tags(picked)
            Path(p).with_suffix(".tags").write_text(line, encoding="utf-8")
            done += 1
            log(f"[Tagger] ({done}/{len(todo)}) {Path(p).name}: {line[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
