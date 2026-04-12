"""Results display widget for blink analysis output."""

from __future__ import annotations

import csv

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from blinkcounter.core.models import AnalysisResult, BatchResult, MatchedPerson, Person
from blinkcounter.gui.charts import BatchBarChart, BlinkRateBarChart, BlinkTimelineChart
from blinkcounter.gui.person_card import MatchedPersonCard, PersonCard


class ResultsWidget(QWidget):
    """Widget displaying analysis results with person cards and charts."""

    def __init__(
        self,
        result: AnalysisResult | BatchResult,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.result = result
        self._setup_ui()

    def _setup_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setSpacing(16)
        self.content_layout.setContentsMargins(16, 16, 16, 16)

        if isinstance(self.result, BatchResult):
            self._build_batch_view()
        else:
            self._build_single_view(self.result)

        # Export button
        export_btn = QPushButton("Export CSV")
        export_btn.setFixedWidth(150)
        export_btn.setStyleSheet(
            "QPushButton { background-color: #0078d7; color: white; border-radius: 4px;"
            "padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background-color: #005fa3; }"
        )
        export_btn.clicked.connect(self._export_csv)
        self.content_layout.addWidget(export_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self.content_layout.addStretch()
        scroll.setWidget(content)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    def _add_section_header(self, text: str) -> None:
        label = QLabel(text)
        label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333333;")
        self.content_layout.addWidget(label)

    def _add_separator(self) -> None:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.content_layout.addWidget(line)

    def _build_single_view(self, result: AnalysisResult) -> None:
        # Header
        self._add_section_header("Analysis Results")

        # Source info
        source_text = result.video_source
        if len(source_text) > 80:
            source_text = "..." + source_text[-77:]
        info_label = QLabel(
            f"Source: {source_text}  |  "
            f"Duration: {result.duration_seconds:.1f}s  |  "
            f"FPS: {result.fps:.1f}  |  "
            f"Persons detected: {result.person_count}"
        )
        info_label.setStyleSheet("color: #666666; font-size: 12px;")
        info_label.setWordWrap(True)
        self.content_layout.addWidget(info_label)

        self._add_separator()

        # Person cards grid
        if result.persons:
            self._add_section_header("Detected Persons")
            cards_widget = self._build_person_cards_grid(result.persons)
            self.content_layout.addWidget(cards_widget)

            # Charts
            self._add_separator()
            self._add_section_header("Charts")

            bar_chart = BlinkRateBarChart()
            bar_chart.set_data(result.persons)
            bar_chart.setMinimumHeight(250)
            self.content_layout.addWidget(bar_chart)

            timeline_chart = BlinkTimelineChart()
            timeline_chart.set_data(result.persons)
            timeline_chart.setMinimumHeight(250)
            self.content_layout.addWidget(timeline_chart)
        else:
            no_data = QLabel("No persons detected in the video.")
            no_data.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_data.setStyleSheet("color: #999999; font-size: 14px; padding: 40px;")
            self.content_layout.addWidget(no_data)

    def _build_batch_view(self) -> None:
        batch: BatchResult = self.result  # type: ignore[assignment]

        self._add_section_header("Batch Analysis Results")
        info_label = QLabel(f"Videos analyzed: {len(batch.video_results)}")
        info_label.setStyleSheet("color: #666666; font-size: 12px;")
        self.content_layout.addWidget(info_label)

        self._add_separator()

        # Aggregate matched persons section
        if batch.matched_persons:
            self._add_section_header("Matched Persons (Across Videos)")
            matched_grid = self._build_matched_cards_grid(batch.matched_persons)
            self.content_layout.addWidget(matched_grid)

            # Batch chart
            batch_chart = BatchBarChart()
            batch_chart.set_data(batch.matched_persons)
            batch_chart.setMinimumHeight(300)
            self.content_layout.addWidget(batch_chart)

            # Also show rate comparison bar chart using matched persons
            bar_chart = BlinkRateBarChart()
            bar_chart.set_data(batch.matched_persons)
            bar_chart.setMinimumHeight(250)
            self.content_layout.addWidget(bar_chart)

        # Per-video breakdown
        self._add_separator()
        self._add_section_header("Per-Video Breakdown")

        for vr in batch.video_results:
            source_short = vr.video_source
            if len(source_short) > 60:
                source_short = "..." + source_short[-57:]
            video_label = QLabel(f"Video: {source_short}")
            video_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #444444; margin-top: 8px;"
            )
            self.content_layout.addWidget(video_label)

            if vr.persons:
                cards = self._build_person_cards_grid(vr.persons)
                self.content_layout.addWidget(cards)
            else:
                empty = QLabel("No persons detected.")
                empty.setStyleSheet("color: #999999; font-size: 12px; padding-left: 16px;")
                self.content_layout.addWidget(empty)

    def _build_person_cards_grid(self, persons: list[Person]) -> QWidget:
        """Build a grid of PersonCard widgets."""
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(12)
        cols = max(1, min(4, len(persons)))
        for i, person in enumerate(persons):
            card = PersonCard(person)
            grid.addWidget(card, i // cols, i % cols)
        return container

    def _build_matched_cards_grid(self, matched: list[MatchedPerson]) -> QWidget:
        """Build a grid of MatchedPersonCard widgets."""
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(12)
        cols = max(1, min(4, len(matched)))
        for i, mp in enumerate(matched):
            card = MatchedPersonCard(
                label=mp.label,
                avg_rate=mp.average_blinks_per_minute,
                classification=mp.classification,
                face_thumbnail=mp.face_thumbnail,
                video_count=len(mp.per_video_rates),
            )
            grid.addWidget(card, i // cols, i % cols)
        return container

    def _export_csv(self) -> None:
        """Export results to a CSV file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "blink_results.csv", "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Person", "Blinks/Min", "Classification",
                    "Total Blinks", "Duration(s)", "Video Source",
                ])

                if isinstance(self.result, BatchResult):
                    for vr in self.result.video_results:
                        for person in vr.persons:
                            writer.writerow([
                                person.label,
                                f"{person.blinks_per_minute:.1f}",
                                person.classification.value,
                                person.blink_count,
                                f"{person.total_visible_duration:.1f}",
                                vr.video_source,
                            ])
                else:
                    for person in self.result.persons:
                        writer.writerow([
                            person.label,
                            f"{person.blinks_per_minute:.1f}",
                            person.classification.value,
                            person.blink_count,
                            f"{person.total_visible_duration:.1f}",
                            self.result.video_source,
                        ])
        except OSError as e:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Export Error", f"Failed to save CSV:\n{e}")
