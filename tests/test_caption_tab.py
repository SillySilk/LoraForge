"""Single-image Caption tab: staging, result reading, and the run job it builds."""
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ui import caption_tab as ct

_app = QApplication.instance() or QApplication([])


def _png(path: Path) -> Path:
    from PySide6.QtGui import QImage
    img = QImage(8, 8, QImage.Format_RGB32)
    img.fill(0)
    img.save(str(path))
    return path


def test_stage_copies_into_private_folder(tmp_path):
    src = _png(tmp_path / "a.png")
    folder = ct.stage_image(str(src))
    try:
        assert (Path(folder) / "a.png").is_file()
        assert Path(folder) != tmp_path
        assert ct.read_result(folder, str(src)) == ""
        (Path(folder) / "a.txt").write_text("  a cat  \n", encoding="utf-8")
        assert ct.read_result(folder, str(src)) == "a cat"
    finally:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)


def test_is_image():
    assert ct.is_image("x.PNG") and ct.is_image("x.webp")
    assert not ct.is_image("x.txt")


def test_load_shows_existing_caption_and_save_writes_it(tmp_path):
    src = _png(tmp_path / "b.png")
    (tmp_path / "b.txt").write_text("old caption", encoding="utf-8")
    tab = ct.CaptionTab()
    tab.load_image(str(src))
    assert tab._result.toPlainText() == "old caption"
    assert tab._run_btn.isEnabled()
    tab._result.setPlainText("new caption")
    tab._save()
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "new caption\n"


def test_start_refused_when_gpu_busy(tmp_path):
    tab = ct.CaptionTab()
    tab.set_sdscripts_path(str(tmp_path))
    tab.set_gpu_busy_check(lambda: "training is in progress")
    tab.load_image(str(_png(tmp_path / "c.png")))
    tab._start()
    assert not tab.is_busy()
    assert "training is in progress" in tab._status.text()


def test_start_builds_single_image_job(tmp_path, monkeypatch):
    seen = {}
    tab = ct.CaptionTab()
    monkeypatch.setattr(tab._runner, "start", lambda job: seen.setdefault("job", job) and True)
    tab.set_sdscripts_path(str(tmp_path))
    tab.load_image(str(_png(tmp_path / "d.png")))
    tab._style.setCurrentIndex(tab._style.findData("tags"))
    tab._name.setText("mira")
    tab._start()
    job = seen["job"]
    try:
        assert job.chain == ["tag", "combine"]
        assert job.trigger == "mira"
        assert list(p.name for p in Path(job.dataset_folder).iterdir()) == ["d.png"]
        assert not (tmp_path / "d.txt").exists(), "the image's own folder is untouched"
    finally:
        tab._cleanup()
