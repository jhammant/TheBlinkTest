"""PersonCard widget displaying blink analysis results for a single person."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from blinkcounter.constants import (
    COLOR_HIGH,
    COLOR_LOW,
    COLOR_NORMAL,
    COLOR_VERY_LOW,
)
from blinkcounter.core.models import BlinkClassification, Person


def numpy_to_pixmap(img: np.ndarray) -> QPixmap:
    """Convert a numpy BGR image to a QPixmap."""
    try:
        import cv2

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:
        # If cv2 is not available, assume RGB already
        rgb = img
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _classification_color(classification: BlinkClassification) -> tuple[int, int, int]:
    """Return the RGB color tuple for a classification."""
    color_map = {
        BlinkClassification.VERY_LOW: COLOR_VERY_LOW,
        BlinkClassification.LOW: COLOR_LOW,
        BlinkClassification.NORMAL: COLOR_NORMAL,
        BlinkClassification.HIGH: COLOR_HIGH,
    }
    return color_map.get(classification, COLOR_NORMAL)


class PersonCard(QWidget):
    """Card widget showing blink analysis for a single person."""

    def __init__(self, person: Person, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.person = person
        self.setFixedSize(200, 250)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setObjectName("PersonCard")
        self.setStyleSheet(
            "#PersonCard {"
            "  background-color: #ffffff;"
            "  border: 1px solid #e0e0e0;"
            "  border-radius: 8px;"
            "}"
        )

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setOffset(0, 2)
        shadow.setColor(Qt.GlobalColor.gray)
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Face thumbnail
        thumb_label = QLabel()
        thumb_label.setFixedSize(80, 80)
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_label.setStyleSheet(
            "border: 2px solid #cccccc; border-radius: 4px; background: #f5f5f5;"
        )
        if self.person.face_thumbnail is not None:
            pixmap = numpy_to_pixmap(self.person.face_thumbnail)
            thumb_label.setPixmap(
                pixmap.scaled(
                    76, 76,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            thumb_label.setText("No\nImage")
            thumb_label.setStyleSheet(
                thumb_label.styleSheet() + " color: #999999; font-size: 11px;"
            )
        layout.addWidget(thumb_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Person label
        name_label = QLabel(self.person.label)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet("font-weight: bold; font-size: 14px; border: none;")
        layout.addWidget(name_label)

        # Blink rate
        rate_label = QLabel(f"{self.person.blinks_per_minute:.1f} blinks/min")
        rate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rate_label.setStyleSheet("font-size: 16px; font-weight: bold; border: none;")
        layout.addWidget(rate_label)

        # Classification badge
        classification = self.person.classification
        r, g, b = _classification_color(classification)
        badge = QLabel(classification.value)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedHeight(24)
        badge.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        badge.setStyleSheet(
            f"background-color: rgb({r},{g},{b});"
            "color: white;"
            "border-radius: 12px;"
            "font-size: 12px;"
            "font-weight: bold;"
            "padding: 2px 8px;"
            "border: none;"
        )
        layout.addWidget(badge, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Stats row
        stats_widget = QWidget()
        stats_widget.setStyleSheet("border: none;")
        stats_layout = QHBoxLayout(stats_widget)
        stats_layout.setContentsMargins(0, 0, 0, 0)

        blinks_label = QLabel(f"Blinks: {self.person.blink_count}")
        blinks_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        blinks_label.setStyleSheet("font-size: 11px; color: #666666; border: none;")
        stats_layout.addWidget(blinks_label)

        duration_label = QLabel(f"Duration: {self.person.total_visible_duration:.1f}s")
        duration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        duration_label.setStyleSheet("font-size: 11px; color: #666666; border: none;")
        stats_layout.addWidget(duration_label)

        layout.addWidget(stats_widget)


class MatchedPersonCard(QWidget):
    """Card widget showing aggregate blink analysis for a matched person across videos."""

    def __init__(
        self,
        label: str,
        avg_rate: float,
        classification: BlinkClassification,
        face_thumbnail: np.ndarray | None = None,
        video_count: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._avg_rate = avg_rate
        self._classification = classification
        self._face_thumbnail = face_thumbnail
        self._video_count = video_count
        self.setFixedSize(200, 250)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setObjectName("MatchedPersonCard")
        self.setStyleSheet(
            "#MatchedPersonCard {"
            "  background-color: #ffffff;"
            "  border: 1px solid #e0e0e0;"
            "  border-radius: 8px;"
            "}"
        )

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setOffset(0, 2)
        shadow.setColor(Qt.GlobalColor.gray)
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Face thumbnail
        thumb_label = QLabel()
        thumb_label.setFixedSize(80, 80)
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_label.setStyleSheet(
            "border: 2px solid #cccccc; border-radius: 4px; background: #f5f5f5;"
        )
        if self._face_thumbnail is not None:
            pixmap = numpy_to_pixmap(self._face_thumbnail)
            thumb_label.setPixmap(
                pixmap.scaled(
                    76, 76,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            thumb_label.setText("No\nImage")
            thumb_label.setStyleSheet(
                thumb_label.styleSheet() + " color: #999999; font-size: 11px;"
            )
        layout.addWidget(thumb_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Person label
        name_label = QLabel(self._label)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet("font-weight: bold; font-size: 14px; border: none;")
        layout.addWidget(name_label)

        # Average blink rate
        rate_label = QLabel(f"{self._avg_rate:.1f} blinks/min (avg)")
        rate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rate_label.setStyleSheet("font-size: 15px; font-weight: bold; border: none;")
        layout.addWidget(rate_label)

        # Classification badge
        r, g, b = _classification_color(self._classification)
        badge = QLabel(self._classification.value)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedHeight(24)
        badge.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        badge.setStyleSheet(
            f"background-color: rgb({r},{g},{b});"
            "color: white;"
            "border-radius: 12px;"
            "font-size: 12px;"
            "font-weight: bold;"
            "padding: 2px 8px;"
            "border: none;"
        )
        layout.addWidget(badge, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Video count
        videos_label = QLabel(f"Across {self._video_count} video(s)")
        videos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        videos_label.setStyleSheet("font-size: 11px; color: #666666; border: none;")
        layout.addWidget(videos_label)
