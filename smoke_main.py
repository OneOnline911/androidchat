import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget


class SmokeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt Smoke Test")

        self._count = 0
        self._label = QLabel(
            "QT SMOKE TEST\n\n"
            "If you can read this, Python + PySide6 + Qt started successfully."
        )
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)

        button = QPushButton("Tap test")
        button.clicked.connect(self._clicked)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(button)

    def _clicked(self):
        self._count += 1
        self._label.setText(f"QT SMOKE TEST RUNNING\n\nButton taps: {self._count}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SmokeWindow()
    window.show()
    sys.exit(app.exec())
