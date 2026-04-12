"""Entry point for BlinkCounter application."""

import sys


def main():
    from PyQt6.QtWidgets import QApplication

    from blinkcounter.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("BlinkCounter")
    app.setOrganizationName("BlinkCounter")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
