"""One-shot find/replace across every caption file of a dataset set.

Unlike the Setup caption rules (applied at every Combine), this edits the files on disk
once. It edits the final .txt AND the .nl/.tags sidecars, because the .txt is rebuilt
from the sidecars on many actions and a .txt-only edit would silently revert.
"""
import re
from pathlib import Path

from core.caption_rules import tidy

CAPTION_EXTS = (".txt", ".nl", ".tags")


def _pattern(find: str, match_case: bool, whole_word: bool):
    body = re.escape(find)
    if whole_word:
        # Same boundary as caption_rules: \b misbehaves on terms like "(medium)".
        body = r"(?<!\w)" + body + r"(?!\w)"
    return re.compile(body, 0 if match_case else re.IGNORECASE)


def replace_text(text: str, find: str, repl: str, match_case: bool,
                 whole_word: bool) -> tuple:
    """(new_text, match_count). Blank `find` is a no-op."""
    if not text or not find.strip():
        return text, 0
    new, count = _pattern(find, match_case, whole_word).subn(lambda m: repl, text)
    if count and repl == "":
        new = tidy(new)
    return (new, count) if count else (text, 0)


def _caption_files(image_paths):
    for p in image_paths:
        for ext in CAPTION_EXTS:
            f = Path(p).with_suffix(ext)
            if f.is_file():
                yield Path(p), f


def _read(f: Path) -> str:
    return f.read_text(encoding="utf-8", errors="replace")


def scan_matches(image_paths, find, match_case, whole_word) -> tuple:
    """(total matches, images with at least one match). Writes nothing."""
    total, hit_images = 0, set()
    for img, f in _caption_files(image_paths):
        _, n = replace_text(_read(f), find, "", match_case, whole_word)
        if n:
            total += n
            hit_images.add(img)
    return total, len(hit_images)


def replace_in_set(image_paths, find, repl, match_case, whole_word) -> tuple:
    """Rewrite every caption file that contains a match. Returns (matches, files written)."""
    total, written = 0, 0
    for _, f in _caption_files(image_paths):
        new, n = replace_text(_read(f), find, repl, match_case, whole_word)
        if n:
            f.write_text(new, encoding="utf-8")
            total += n
            written += 1
    return total, written
