"""Entry point for the 3D File Analyzer application."""

import sys
import os


def main() -> None:
    # Must be set before creating QApplication on some Linux setups.
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("3D File Analyzer")

    from src.gui.main_window import MainWindow

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
