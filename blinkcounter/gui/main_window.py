"""Main application window for BlinkCounter."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QMimeData, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)

from blinkcounter.sample_videos import SAMPLE_VIDEOS

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


class AnalysisWorker(QThread):
    """Worker thread for running video analysis without blocking the GUI."""

    progress = pyqtSignal(float, str)  # progress 0-1, status message
    finished = pyqtSignal(object)  # AnalysisResult or BatchResult
    error = pyqtSignal(str)

    def __init__(
        self,
        video_path: str | None = None,
        video_paths: list[str] | None = None,
        parent: QThread | None = None,
    ) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self.video_paths = video_paths

    def run(self) -> None:
        try:
            if self.video_paths:
                self._run_batch()
            elif self.video_path:
                self._run_single()
            else:
                self.error.emit("No video path provided.")
        except Exception as e:
            self.error.emit(str(e))

    def _run_single(self) -> None:
        self.progress.emit(0.0, "Starting analysis...")
        try:
            from blinkcounter.core.video_analyzer import VideoAnalyzer
        except ImportError as e:
            self.error.emit(f"VideoAnalyzer not available: {e}")
            return

        analyzer = VideoAnalyzer()

        def on_progress(value: float, msg: str = "") -> None:
            self.progress.emit(value, msg)

        result = analyzer.analyze(self.video_path, progress_callback=on_progress)
        self.progress.emit(1.0, "Complete")
        self.finished.emit(result)

    def _run_batch(self) -> None:
        self.progress.emit(0.0, "Starting batch analysis...")
        try:
            from blinkcounter.core.batch_analyzer import BatchAnalyzer
        except ImportError as e:
            self.error.emit(f"BatchAnalyzer not available: {e}")
            return

        analyzer = BatchAnalyzer()

        def on_progress(value: float, msg: str = "") -> None:
            self.progress.emit(value, msg)

        result = analyzer.analyze(self.video_paths, progress_callback=on_progress)
        self.progress.emit(1.0, "Complete")
        self.finished.emit(result)


def _resolve_video_input(path_or_url: str) -> str:
    """If the input looks like a YouTube URL, download it first. Otherwise return as-is."""
    path_or_url = path_or_url.strip()
    if any(domain in path_or_url for domain in ("youtube.com", "youtu.be", "youtube.co")):
        try:
            from blinkcounter.services.youtube import download_video

            return download_video(path_or_url)
        except ImportError:
            raise RuntimeError(
                "YouTube download service is not available. "
                "Please install the required dependencies."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download YouTube video: {e}")
    return path_or_url


class DropZone(QFrame):
    """A drag-and-drop zone that accepts video files."""

    file_dropped = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
        self.setStyleSheet(
            "DropZone {"
            "  border: 2px dashed #aaaaaa;"
            "  border-radius: 8px;"
            "  background-color: #fafafa;"
            "}"
            "DropZone:hover {"
            "  border-color: #0078d7;"
            "  background-color: #f0f7ff;"
            "}"
        )

        layout = QVBoxLayout(self)
        label = QLabel("Drag & drop a video file here")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #888888; font-size: 14px; border: none;")
        layout.addWidget(label)

        sublabel = QLabel("Supported: .mp4, .avi, .mkv, .mov, .webm")
        sublabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sublabel.setStyleSheet("color: #bbbbbb; font-size: 11px; border: none;")
        layout.addWidget(sublabel)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData() and event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if any(path.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        if event.mimeData() and event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if any(path.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                        self.file_dropped.emit(path)
                        event.acceptProposedAction()
                        return
        event.ignore()


class CollapsibleSection(QWidget):
    """A collapsible section with a toggle button and content area."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_expanded = False

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.toggle_btn = QToolButton()
        self.toggle_btn.setText(f"  {title}")
        self.toggle_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_btn.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; font-size: 13px; padding: 4px; }"
            "QToolButton:hover { color: #0078d7; }"
        )
        self.toggle_btn.clicked.connect(self._toggle)
        main_layout.addWidget(self.toggle_btn)

        self.content_area = QWidget()
        self.content_area.setVisible(False)
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(16, 4, 0, 4)
        main_layout.addWidget(self.content_area)

    def _toggle(self) -> None:
        self._is_expanded = not self._is_expanded
        self.content_area.setVisible(self._is_expanded)
        arrow = Qt.ArrowType.DownArrow if self._is_expanded else Qt.ArrowType.RightArrow
        self.toggle_btn.setArrowType(arrow)

    def add_widget(self, widget: QWidget) -> None:
        self.content_layout.addWidget(widget)


class MainWindow(QMainWindow):
    """Main application window for BlinkCounter."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BlinkCounter - Blink Rate Analyzer")
        self.setMinimumSize(900, 700)

        self._worker: AnalysisWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        self._main_layout = QVBoxLayout(central)

        # Stacked views: input vs results
        self._input_widget = QWidget()
        self._results_container = QWidget()
        self._results_layout = QVBoxLayout(self._results_container)
        self._results_layout.setContentsMargins(0, 0, 0, 0)
        self._results_container.setVisible(False)

        self._setup_input_ui()
        self._setup_results_back_button()

        self._main_layout.addWidget(self._input_widget)
        self._main_layout.addWidget(self._results_container)

    def _setup_input_ui(self) -> None:
        layout = QVBoxLayout(self._input_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # --- Single Video Tab ---
        single_tab = QWidget()
        single_layout = QVBoxLayout(single_tab)
        single_layout.setSpacing(12)
        single_layout.setContentsMargins(16, 16, 16, 16)

        # Drop zone
        self._drop_zone = DropZone()
        self._drop_zone.file_dropped.connect(self._on_file_selected)
        single_layout.addWidget(self._drop_zone)

        # Select file button
        select_btn = QPushButton("Select File...")
        select_btn.setFixedWidth(140)
        select_btn.setStyleSheet(
            "QPushButton { background-color: #0078d7; color: white; border-radius: 4px;"
            "padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background-color: #005fa3; }"
        )
        select_btn.clicked.connect(self._browse_file)
        single_layout.addWidget(select_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # YouTube URL row
        url_row = QWidget()
        url_layout = QHBoxLayout(url_row)
        url_layout.setContentsMargins(0, 0, 0, 0)

        url_label = QLabel("YouTube URL:")
        url_label.setStyleSheet("font-weight: bold;")
        url_layout.addWidget(url_label)

        self._url_input = QLineEdit()
        self._url_input.setPlaceholderText("https://www.youtube.com/watch?v=...")
        self._url_input.returnPressed.connect(self._on_analyze_url)
        url_layout.addWidget(self._url_input)

        analyze_url_btn = QPushButton("Analyze URL")
        analyze_url_btn.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; border-radius: 4px;"
            "padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background-color: #1e7e34; }"
        )
        analyze_url_btn.clicked.connect(self._on_analyze_url)
        url_layout.addWidget(analyze_url_btn)

        single_layout.addWidget(url_row)

        # Sample videos collapsible section
        samples_section = CollapsibleSection("Sample Videos")
        for sample in SAMPLE_VIDEOS:
            btn = QPushButton(sample["label"])
            btn.setToolTip(sample.get("note", ""))
            btn.setStyleSheet(
                "QPushButton { text-align: left; padding: 6px 10px; }"
                "QPushButton:hover { background-color: #e8f4fd; }"
            )
            url = sample["url"]
            btn.clicked.connect(lambda checked, u=url: self._start_analysis_url(u))
            samples_section.add_widget(btn)
        single_layout.addWidget(samples_section)

        # Progress area
        self._single_progress = QProgressBar()
        self._single_progress.setRange(0, 1000)
        self._single_progress.setValue(0)
        self._single_progress.setVisible(False)
        single_layout.addWidget(self._single_progress)

        self._single_status = QLabel("")
        self._single_status.setStyleSheet("color: #666666; font-size: 12px;")
        single_layout.addWidget(self._single_status)

        single_layout.addStretch()
        self._tabs.addTab(single_tab, "Single Video")

        # --- Batch Analysis Tab ---
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)
        batch_layout.setSpacing(12)
        batch_layout.setContentsMargins(16, 16, 16, 16)

        batch_label = QLabel("Queue videos for batch analysis:")
        batch_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        batch_layout.addWidget(batch_label)

        self._batch_list = QListWidget()
        self._batch_list.setMinimumHeight(200)
        batch_layout.addWidget(self._batch_list)

        # Batch buttons row
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        add_file_btn = QPushButton("Add File...")
        add_file_btn.clicked.connect(self._batch_add_file)
        add_file_btn.setStyleSheet(
            "QPushButton { background-color: #0078d7; color: white; border-radius: 4px;"
            "padding: 8px 12px; }"
            "QPushButton:hover { background-color: #005fa3; }"
        )
        btn_layout.addWidget(add_file_btn)

        add_url_btn = QPushButton("Add YouTube URL...")
        add_url_btn.clicked.connect(self._batch_add_url)
        add_url_btn.setStyleSheet(
            "QPushButton { background-color: #0078d7; color: white; border-radius: 4px;"
            "padding: 8px 12px; }"
            "QPushButton:hover { background-color: #005fa3; }"
        )
        btn_layout.addWidget(add_url_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._batch_remove_selected)
        remove_btn.setStyleSheet(
            "QPushButton { background-color: #dc3545; color: white; border-radius: 4px;"
            "padding: 8px 12px; }"
            "QPushButton:hover { background-color: #c82333; }"
        )
        btn_layout.addWidget(remove_btn)

        btn_layout.addStretch()

        analyze_all_btn = QPushButton("Analyze All")
        analyze_all_btn.clicked.connect(self._batch_analyze)
        analyze_all_btn.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; border-radius: 4px;"
            "padding: 8px 16px; font-weight: bold; font-size: 14px; }"
            "QPushButton:hover { background-color: #1e7e34; }"
        )
        btn_layout.addWidget(analyze_all_btn)

        batch_layout.addWidget(btn_row)

        # Batch progress area
        self._batch_progress = QProgressBar()
        self._batch_progress.setRange(0, 1000)
        self._batch_progress.setValue(0)
        self._batch_progress.setVisible(False)
        batch_layout.addWidget(self._batch_progress)

        self._batch_status = QLabel("")
        self._batch_status.setStyleSheet("color: #666666; font-size: 12px;")
        batch_layout.addWidget(self._batch_status)

        batch_layout.addStretch()
        self._tabs.addTab(batch_tab, "Batch Analysis")

    def _setup_results_back_button(self) -> None:
        back_btn = QPushButton("New Analysis")
        back_btn.setFixedWidth(140)
        back_btn.setStyleSheet(
            "QPushButton { background-color: #6c757d; color: white; border-radius: 4px;"
            "padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background-color: #545b62; }"
        )
        back_btn.clicked.connect(self._back_to_input)
        self._results_layout.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignLeft)

    # --- Single video actions ---

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*)",
        )
        if path:
            self._on_file_selected(path)

    def _on_file_selected(self, path: str) -> None:
        self._start_analysis(path)

    def _on_analyze_url(self) -> None:
        url = self._url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Input Required", "Please enter a YouTube URL.")
            return
        self._start_analysis_url(url)

    def _start_analysis_url(self, url: str) -> None:
        self._single_status.setText("Downloading video...")
        self._single_progress.setVisible(True)
        self._single_progress.setValue(0)

        try:
            resolved = _resolve_video_input(url)
        except RuntimeError as e:
            self._single_status.setText(f"Error: {e}")
            self._single_progress.setVisible(False)
            QMessageBox.critical(self, "Download Error", str(e))
            return

        self._start_analysis(resolved)

    def _start_analysis(self, video_path: str) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "An analysis is already running.")
            return

        self._single_progress.setVisible(True)
        self._single_progress.setValue(0)
        self._single_status.setText("Starting analysis...")

        self._worker = AnalysisWorker(video_path=video_path)
        self._worker.progress.connect(self._on_single_progress)
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_single_progress(self, value: float, msg: str) -> None:
        self._single_progress.setValue(int(value * 1000))
        self._single_status.setText(msg)

    def _on_analysis_finished(self, result: object) -> None:
        self._single_progress.setVisible(False)
        self._single_status.setText("Analysis complete.")
        self._show_results(result)

    def _on_analysis_error(self, msg: str) -> None:
        self._single_progress.setVisible(False)
        self._single_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Analysis Error", msg)

    # --- Batch actions ---

    def _batch_add_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*)",
        )
        if path:
            self._batch_list.addItem(path)

    def _batch_add_url(self) -> None:
        url, ok = QInputDialog.getText(
            self, "Add YouTube URL", "YouTube URL:"
        )
        if ok and url.strip():
            self._batch_list.addItem(url.strip())

    def _batch_remove_selected(self) -> None:
        for item in self._batch_list.selectedItems():
            row = self._batch_list.row(item)
            self._batch_list.takeItem(row)

    def _batch_analyze(self) -> None:
        count = self._batch_list.count()
        if count == 0:
            QMessageBox.warning(self, "No Videos", "Please add at least one video to analyze.")
            return

        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "An analysis is already running.")
            return

        # Resolve all paths/URLs
        video_paths: list[str] = []
        self._batch_status.setText("Resolving video sources...")
        self._batch_progress.setVisible(True)
        self._batch_progress.setValue(0)

        try:
            for i in range(count):
                item_text = self._batch_list.item(i).text()
                resolved = _resolve_video_input(item_text)
                video_paths.append(resolved)
        except RuntimeError as e:
            self._batch_status.setText(f"Error: {e}")
            self._batch_progress.setVisible(False)
            QMessageBox.critical(self, "Error", str(e))
            return

        self._batch_status.setText("Starting batch analysis...")
        self._worker = AnalysisWorker(video_paths=video_paths)
        self._worker.progress.connect(self._on_batch_progress)
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.error.connect(self._on_batch_error)
        self._worker.start()

    def _on_batch_progress(self, value: float, msg: str) -> None:
        self._batch_progress.setValue(int(value * 1000))
        self._batch_status.setText(msg)

    def _on_batch_error(self, msg: str) -> None:
        self._batch_progress.setVisible(False)
        self._batch_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Batch Analysis Error", msg)

    # --- Results view ---

    def _show_results(self, result: object) -> None:
        from blinkcounter.gui.results_widget import ResultsWidget

        # Clear any previous results widget
        for i in reversed(range(self._results_layout.count())):
            widget = self._results_layout.itemAt(i).widget()
            if isinstance(widget, ResultsWidget):
                widget.setParent(None)
                widget.deleteLater()

        results_widget = ResultsWidget(result)  # type: ignore[arg-type]
        self._results_layout.addWidget(results_widget)

        self._input_widget.setVisible(False)
        self._results_container.setVisible(True)

    def _back_to_input(self) -> None:
        self._results_container.setVisible(False)
        self._input_widget.setVisible(True)

        # Reset progress
        self._single_progress.setVisible(False)
        self._single_progress.setValue(0)
        self._single_status.setText("")
        self._batch_progress.setVisible(False)
        self._batch_progress.setValue(0)
        self._batch_status.setText("")
