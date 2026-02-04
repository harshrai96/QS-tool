# from PySide6.QtWidgets import QApplication, QPushButton, QWidget, QVBoxLayout
# import subprocess
# import sys
#
# app = QApplication(sys.argv)
#
# window = QWidget()
# window.setWindowTitle("Vision QC")
# window.showFullScreen()
#
# layout = QVBoxLayout()
#
# btn = QPushButton("Live Predictions")
# btn.setStyleSheet("font-size: 40px; height: 200px;")
#
# def start_live():
#     subprocess.Popen(["python", "infer_4.py"])
#
# btn.clicked.connect(start_live)
# layout.addWidget(btn)
#
# window.setLayout(layout)
# window.show()
#
# sys.exit(app.exec())


import sys
import subprocess
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QSpacerItem,
    QSizePolicy
)
from PySide6.QtCore import Qt


app = QApplication(sys.argv)

# Main window
window = QWidget()
window.setWindowTitle("Opelka – Qualitätssicherungsmodul")
window.showFullScreen()

# Main layout
layout = QVBoxLayout()
layout.setContentsMargins(60, 40, 60, 40)
layout.setSpacing(30)

# ----- TITLE -----
title = QLabel("Opelka – Qualitätssicherungsmodul")
title.setAlignment(Qt.AlignCenter)
title.setStyleSheet("""
    QLabel {
        font-size: 36px;
        font-weight: bold;
    }
""")
layout.addWidget(title)

# Spacer (push button to center area)
layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

# ----- LIVE PREDICTIONS BUTTON -----
live_btn = QPushButton("Live-Prognosen")
live_btn.setFixedHeight(120)
live_btn.setStyleSheet("""
    QPushButton {
        font-size: 28px;
        border-radius: 12px;
        background-color: #2c7be5;
        color: white;
    }
    QPushButton:pressed {
        background-color: #1a5bb8;
    }
""")

def start_live():
    subprocess.Popen(["python", "infer_4.py"])

live_btn.clicked.connect(start_live)
layout.addWidget(live_btn, alignment=Qt.AlignCenter)

# Spacer (bottom)
layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

window.setLayout(layout)
window.show()

sys.exit(app.exec())
