"""Matplotlib charts embedded in PyQt6 for blink rate visualization."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from blinkcounter.constants import (
    CLASSIFICATION_LOW_MAX,
    CLASSIFICATION_NORMAL_MAX,
    CLASSIFICATION_VERY_LOW_MAX,
    COLOR_HIGH,
    COLOR_LOW,
    COLOR_NORMAL,
    COLOR_VERY_LOW,
)
from blinkcounter.core.models import BlinkClassification, MatchedPerson, Person


def _color_for_classification(classification: BlinkClassification) -> str:
    """Return a matplotlib-compatible color string for a classification."""
    color_map = {
        BlinkClassification.VERY_LOW: COLOR_VERY_LOW,
        BlinkClassification.LOW: COLOR_LOW,
        BlinkClassification.NORMAL: COLOR_NORMAL,
        BlinkClassification.HIGH: COLOR_HIGH,
    }
    r, g, b = color_map.get(classification, COLOR_NORMAL)
    return f"#{r:02x}{g:02x}{b:02x}"


def _color_for_rate(rate: float) -> str:
    """Return a matplotlib color string for a given blink rate."""
    return _color_for_classification(BlinkClassification.from_rate(rate))


class BlinkRateBarChart(QWidget):
    """Horizontal bar chart comparing blink rates across persons."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 3), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def set_data(self, persons: list[Person] | list[MatchedPerson]) -> None:
        """Set chart data from a list of Person or MatchedPerson objects."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not persons:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw()
            return

        labels: list[str] = []
        rates: list[float] = []
        colors: list[str] = []

        for p in persons:
            if isinstance(p, MatchedPerson):
                labels.append(p.label)
                rate = p.average_blinks_per_minute
            else:
                labels.append(p.label)
                rate = p.blinks_per_minute
            rates.append(rate)
            colors.append(_color_for_rate(rate))

        y_pos = range(len(labels))
        ax.barh(y_pos, rates, color=colors, height=0.6, zorder=3)

        # Normal range band
        ax.axvspan(
            CLASSIFICATION_LOW_MAX, CLASSIFICATION_NORMAL_MAX,
            alpha=0.15, color="#28a745", zorder=1, label="Normal range (15-20)",
        )

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Blinks per Minute")
        ax.set_title("Blink Rate Comparison")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3, zorder=0)

        self.figure.tight_layout()
        self.canvas.draw()


class BlinkTimelineChart(QWidget):
    """Scatter plot showing blink events over time per person."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 3), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def set_data(self, persons: list[Person]) -> None:
        """Set chart data from a list of Person objects."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not persons:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw()
            return

        default_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf",
        ]

        labels: list[str] = []
        for i, person in enumerate(persons):
            labels.append(person.label)
            timestamps = [evt.timestamp for evt in person.blink_events]
            y_vals = [i] * len(timestamps)
            color = default_colors[i % len(default_colors)]
            ax.scatter(
                timestamps, y_vals, label=person.label,
                color=color, alpha=0.7, s=30, zorder=3,
            )

        ax.set_yticks(list(range(len(labels))))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Time (seconds)")
        ax.set_title("Blink Timeline")
        if len(labels) <= 8:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="x", alpha=0.3, zorder=0)

        self.figure.tight_layout()
        self.canvas.draw()


class BatchBarChart(QWidget):
    """Grouped horizontal bar chart for batch results showing per-video rates."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def set_data(self, matched_persons: list[MatchedPerson]) -> None:
        """Set chart data from matched persons across videos."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not matched_persons:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw()
            return

        # Collect all unique video sources
        all_videos: list[str] = []
        for mp in matched_persons:
            for vid in mp.per_video_rates:
                if vid not in all_videos:
                    all_videos.append(vid)

        bar_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        ]

        num_persons = len(matched_persons)
        num_bars_per_group = len(all_videos) + 1  # +1 for average
        bar_height = 0.8 / num_bars_per_group

        person_labels = [mp.label for mp in matched_persons]

        for v_idx, video_src in enumerate(all_videos):
            # Shorten the label for display
            short_label = video_src.split("/")[-1][:30] if "/" in video_src else video_src[:30]
            y_positions = []
            widths = []
            for p_idx, mp in enumerate(matched_persons):
                y_positions.append(p_idx + v_idx * bar_height - 0.4 + bar_height / 2)
                widths.append(mp.per_video_rates.get(video_src, 0.0))
            color = bar_colors[v_idx % len(bar_colors)]
            ax.barh(y_positions, widths, height=bar_height, label=short_label, color=color, zorder=3)

        # Average bars
        avg_y = []
        avg_w = []
        for p_idx, mp in enumerate(matched_persons):
            avg_y.append(p_idx + len(all_videos) * bar_height - 0.4 + bar_height / 2)
            avg_w.append(mp.average_blinks_per_minute)
        ax.barh(
            avg_y, avg_w, height=bar_height,
            label="Average", color="#333333", zorder=3, alpha=0.8,
        )

        # Normal range band
        ax.axvspan(
            CLASSIFICATION_LOW_MAX, CLASSIFICATION_NORMAL_MAX,
            alpha=0.15, color="#28a745", zorder=1, label="Normal (15-20)",
        )

        ax.set_yticks(list(range(num_persons)))
        ax.set_yticklabels(person_labels)
        ax.set_xlabel("Blinks per Minute")
        ax.set_title("Batch Blink Rate Comparison")
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(axis="x", alpha=0.3, zorder=0)

        self.figure.tight_layout()
        self.canvas.draw()
