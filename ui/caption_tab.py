"""Caption tab ("The Scribe"): caption one image, no training.

A standalone tool, not a per-run control, so it doesn't break the Home-owns-controls
rule: nothing here also exists on Home. The image is copied into a private temp folder
and the normal CaptionRunner chain runs there, so the image's own folder is untouched
until the user presses Save.
"""
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QSettings, Signal
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core import caption_policy as cp
from core.caption_rules import parse_rules
from core.caption_runner import CaptionJob, CaptionRunner, chain_for_style
from core.describe_models import read_describe_defaults
from core.settings import SETTINGS_APP, SETTINGS_ORG, AppSettings
from core.tagger import TAGGER_MODELS, read_tagger_defaults

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_STYLE_LABELS = [("Prose + tags", "prose+tags"), ("Prose only", "prose"), ("Tags only", "tags")]
_KEY_DIR = "caption_tab/last_dir"
_KEY_STYLE = "caption_tab/style"


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def stage_image(image_path: str) -> str:
    """Copy the image into a fresh temp folder; return that folder."""
    folder = tempfile.mkdtemp(prefix="af_caption_")
    shutil.copy2(image_path, Path(folder) / Path(image_path).name)
    return folder


def read_result(folder: str, image_name: str) -> str:
    txt = Path(folder) / Path(image_name).with_suffix(".txt").name
    try:
        return txt.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


class DropZone(QLabel):
    """Image preview that also takes a dropped image file."""
    image_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 360)
        self.setObjectName("af_card")
        self.setStyleSheet("border: 2px dashed #5a4a26; border-radius: 8px; color: #8a8a93;")
        self.setText("Drop an image here\nor use Browse…")
        self._pm = QPixmap()

    def show_image(self, path: str):
        self._pm = QPixmap(path)
        self._rescale()

    def _rescale(self):
        if self._pm.isNull():
            return
        self.setPixmap(self._pm.scaled(self.size() * 0.96, Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def dragEnterEvent(self, e):
        urls = e.mimeData().urls()
        if urls and urls[0].isLocalFile() and is_image(urls[0].toLocalFile()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        path = e.mimeData().urls()[0].toLocalFile()
        if is_image(path):
            self.image_dropped.emit(path)


class CaptionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._sdscripts_path = ""
        self._image = ""
        self._work_dir = ""
        self._gpu_busy_check = lambda: None
        self._runner = CaptionRunner(self)
        self._runner.log_line.connect(self._on_log)
        self._runner.finished.connect(self._on_finished)
        self._build_ui()

    # ---- public -----------------------------------------------------------

    def set_sdscripts_path(self, path: str):
        self._sdscripts_path = path or ""

    def set_gpu_busy_check(self, fn):
        """Injected by MainWindow: reason the GPU is busy elsewhere, or None."""
        self._gpu_busy_check = fn

    def is_busy(self) -> bool:
        return self._runner.is_running()

    def load_image(self, path: str):
        if self.is_busy() or not is_image(path) or not Path(path).is_file():
            return
        self._image = path
        self._drop.show_image(path)
        self._path_label.setText(path)
        QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(_KEY_DIR, str(Path(path).parent))
        existing = Path(path).with_suffix(".txt")
        self._result.setPlainText(
            existing.read_text(encoding="utf-8", errors="replace").strip()
            if existing.is_file() else "")
        self._status.setText("Existing caption loaded." if existing.is_file() else "")
        self._refresh_buttons()

    # ---- UI -----------------------------------------------------------------

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(16)

        left = QVBoxLayout()
        self._drop = DropZone()
        self._drop.image_dropped.connect(self.load_image)
        left.addWidget(self._drop, 1)
        row = QHBoxLayout()
        browse = QPushButton("Browse…")
        browse.setObjectName("af_btn_ghost")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        self._path_label = QLabel("")
        self._path_label.setObjectName("af_eyebrow_mute")
        self._path_label.setWordWrap(True)
        row.addWidget(self._path_label, 1)
        left.addLayout(row)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        title = QLabel("Caption one image")
        title.setObjectName("af_display_gold")
        right.addWidget(title)
        note = QLabel("Uses the tagger and describe model chosen in Setup. "
                      "Nothing is written next to the image until you press Save.")
        note.setObjectName("af_eyebrow_mute")
        note.setWordWrap(True)
        right.addWidget(note)

        opts = QHBoxLayout()
        opts.addWidget(QLabel("Style"))
        self._style = QComboBox()
        for label, key in _STYLE_LABELS:
            self._style.addItem(label, key)
        saved = QSettings(SETTINGS_ORG, SETTINGS_APP).value(_KEY_STYLE, "prose+tags")
        self._style.setCurrentIndex(max(0, self._style.findData(saved)))
        self._style.currentIndexChanged.connect(
            lambda: QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(
                _KEY_STYLE, self._style.currentData()))
        opts.addWidget(self._style)
        opts.addSpacing(12)
        opts.addWidget(QLabel("Name"))
        self._name = QLineEdit()
        self._name.setPlaceholderText("optional — what the caption calls the subject")
        opts.addWidget(self._name, 1)
        right.addLayout(opts)

        run_row = QHBoxLayout()
        self._run_btn = QPushButton("▶  Caption")
        self._run_btn.setObjectName("btn_start")
        self._run_btn.clicked.connect(self._start)
        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setObjectName("btn_stop")
        self._stop_btn.clicked.connect(self._runner.stop)
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._stop_btn)
        self._status = QLabel("")
        self._status.setObjectName("af_eyebrow_mute")
        run_row.addWidget(self._status, 1)
        right.addLayout(run_row)

        self._result = QPlainTextEdit()
        self._result.setPlaceholderText("The caption appears here. You can edit it.")
        self._result.textChanged.connect(self._refresh_buttons)
        right.addWidget(self._result, 1)

        out_row = QHBoxLayout()
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setObjectName("af_btn_ghost")
        self._copy_btn.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self._result.toPlainText()))
        self._save_btn = QPushButton("Save .txt next to image")
        self._save_btn.setObjectName("af_btn_ghost")
        self._save_btn.clicked.connect(self._save)
        out_row.addWidget(self._copy_btn)
        out_row.addWidget(self._save_btn)
        out_row.addStretch()
        right.addLayout(out_row)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setObjectName("log_output")
        self._log.setFixedHeight(120)
        self._log.setStyleSheet("font-family: Consolas, monospace; font-size: 12px; "
                                "background-color: #0c0b0a; color: #c6c6ce; "
                                "border: 1px solid #2a2a1e; border-radius: 5px;")
        right.addWidget(self._log)
        layout.addLayout(right, 1)
        self._refresh_buttons()

    def _refresh_buttons(self):
        busy = self.is_busy()
        has_text = bool(self._result.toPlainText().strip())
        self._run_btn.setEnabled(bool(self._image) and not busy)
        self._stop_btn.setEnabled(busy)
        self._copy_btn.setEnabled(has_text)
        self._save_btn.setEnabled(has_text and bool(self._image) and not busy)

    # ---- actions ------------------------------------------------------------

    def _browse(self):
        start = QSettings(SETTINGS_ORG, SETTINGS_APP).value(_KEY_DIR, "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an image", start,
            "Images (" + " ".join(f"*{e}" for e in IMAGE_EXTS) + ")")
        if path:
            self.load_image(path)

    def _start(self):
        if not self._image or self.is_busy():
            return
        reason = self._gpu_busy_check()
        if reason:
            self._status.setText(f"Can't start: {reason}.")
            return
        if not self._sdscripts_path:
            self._status.setText("Set the sd-scripts path in Setup first.")
            return
        model_index, threshold, _ = read_tagger_defaults()
        _label, tagger_id, use_onnx = TAGGER_MODELS[model_index]
        describe_id, describe_cache, _ = read_describe_defaults()
        style = self._style.currentData()
        self._work_dir = stage_image(self._image)
        job = CaptionJob(
            dataset_folder=self._work_dir,
            sdscripts_path=self._sdscripts_path,
            trigger=self._name.text().strip(),
            chain=chain_for_style(style),
            policy=cp.OVERWRITE,
            tagger_model_id=tagger_id,
            tagger_threshold=threshold,
            tagger_use_onnx=use_onnx,
            describe_model_id=describe_id,
            describe_cache_dir=describe_cache or "",
            caption_rules=parse_rules(AppSettings().get("caption_rules_json")),
            caption_style=style,
        )
        self._log.clear()
        self._status.setText("Captioning… (the first run loads the models)")
        if not self._runner.start(job):
            self._status.setText("Could not start captioning.")
            self._cleanup()
        self._refresh_buttons()

    def _on_log(self, line: str):
        self._log.append(line)

    def _on_finished(self, ok: bool):
        text = read_result(self._work_dir, self._image) if ok else ""
        if text:
            self._result.setPlainText(text)
            self._status.setText("Done. Edit it, then Copy or Save.")
        else:
            self._status.setText("Captioning stopped or failed — see the log.")
        self._cleanup()
        self._refresh_buttons()

    def _save(self):
        txt = Path(self._image).with_suffix(".txt")
        try:
            txt.write_text(self._result.toPlainText().strip() + "\n", encoding="utf-8")
        except OSError as e:
            self._status.setText(f"Save failed: {e}")
            return
        self._status.setText(f"Saved {txt.name}.")

    def _cleanup(self):
        if self._work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)
            self._work_dir = ""
