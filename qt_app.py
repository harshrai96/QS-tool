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

# works but without exit pin
# import sys
# import subprocess
# from PySide6.QtWidgets import (
#     QApplication,
#     QWidget,
#     QPushButton,
#     QVBoxLayout,
#     QLabel,
#     QSpacerItem,
#     QSizePolicy
# )
# from PySide6.QtCore import Qt
#
#
# app = QApplication(sys.argv)
#
# # Main window
# window = QWidget()
# window.setWindowTitle("Opelka – Qualitätssicherungsmodul")
# window.showFullScreen()
#
# # Main layout
# layout = QVBoxLayout()
# layout.setContentsMargins(60, 40, 60, 40)
# layout.setSpacing(30)
#
# # ----- TITLE -----
# title = QLabel("Opelka – Qualitätssicherungsmodul")
# title.setAlignment(Qt.AlignCenter)
# title.setStyleSheet("""
#     QLabel {
#         font-size: 36px;
#         font-weight: bold;
#     }
# """)
# layout.addWidget(title)
#
# # Spacer (push button to center area)
# layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
#
# # ----- LIVE PREDICTIONS BUTTON -----
# live_btn = QPushButton("Live-Prognosen")
# live_btn.setFixedHeight(120)
# live_btn.setStyleSheet("""
#     QPushButton {
#         font-size: 28px;
#         border-radius: 12px;
#         background-color: #2c7be5;
#         color: white;
#     }
#     QPushButton:pressed {
#         background-color: #1a5bb8;
#     }
# """)
#
# def start_live():
#     subprocess.Popen(["python", "infer_4.py"])
#
# live_btn.clicked.connect(start_live)
# layout.addWidget(live_btn, alignment=Qt.AlignCenter)
#
# # Spacer (bottom)
# layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
#
# window.setLayout(layout)
# window.show()
#
# sys.exit(app.exec())


#works but has data labelling tab, could be of use later
# import os
# import sys
# import subprocess
#
# from PySide6.QtCore import Qt
# from PySide6.QtWidgets import (
#     QApplication,
#     QWidget,
#     QPushButton,
#     QVBoxLayout,
#     QHBoxLayout,
#     QLabel,
#     QMessageBox,
#     QInputDialog,
#     QLineEdit,
#     QSizePolicy,
# )
#
#
# # =========================
# # CONFIG
# # =========================
# DEV_PIN = "4321"  # <-- CHANGE THIS
# APP_TITLE = "Opelka – Qualitätssicherungsmodul"
#
#
# class MainWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle(APP_TITLE)
#         self.showFullScreen()
#
#         self.infer_process: subprocess.Popen | None = None
#
#         # ---------- Layouts ----------
#         root = QVBoxLayout()
#         root.setContentsMargins(60, 40, 60, 40)
#         root.setSpacing(25)
#
#         top_row = QHBoxLayout()
#         top_row.setContentsMargins(0, 0, 0, 0)
#
#         # Title
#         title = QLabel(APP_TITLE)
#         title.setAlignment(Qt.AlignCenter)
#         title.setStyleSheet("font-size: 36px; font-weight: 700;")
#         title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
#
#         # Developer/Exit button (small + discreet)
#         dev_btn = QPushButton("Developer")
#         dev_btn.setFixedSize(140, 52)
#         dev_btn.setStyleSheet("""
#             QPushButton {
#                 font-size: 16px;
#                 border-radius: 10px;
#                 background: #444;
#                 color: white;
#             }
#             QPushButton:pressed { background: #333; }
#         """)
#         dev_btn.clicked.connect(self.ask_exit_pin)
#
#         top_row.addWidget(title, stretch=1)
#         top_row.addWidget(dev_btn, alignment=Qt.AlignRight)
#
#         root.addLayout(top_row)
#
#         # ---------- Main buttons ----------
#         btn_style_primary = """
#             QPushButton {
#                 font-size: 30px;
#                 border-radius: 16px;
#                 background-color: #2c7be5;
#                 color: white;
#                 padding: 18px;
#             }
#             QPushButton:pressed {
#                 background-color: #1a5bb8;
#             }
#         """
#
#         btn_style_secondary = """
#             QPushButton {
#                 font-size: 28px;
#                 border-radius: 16px;
#                 background-color: #666;
#                 color: white;
#                 padding: 18px;
#             }
#             QPushButton:pressed {
#                 background-color: #555;
#             }
#         """
#
#         self.live_btn = QPushButton("Live Predictions")
#         self.live_btn.setFixedHeight(130)
#         self.live_btn.setStyleSheet(btn_style_primary)
#         self.live_btn.clicked.connect(self.start_live_predictions)
#
#         self.label_btn = QPushButton("Data Labeling")
#         self.label_btn.setFixedHeight(110)
#         self.label_btn.setStyleSheet(btn_style_secondary)
#         self.label_btn.clicked.connect(self.open_data_labeling_placeholder)
#
#         root.addStretch(1)
#         root.addWidget(self.live_btn)
#         root.addWidget(self.label_btn)
#         root.addStretch(2)
#
#         # Hint line
#         hint = QLabel("Tip: Use Developer button + PIN to exit to desktop")
#         hint.setAlignment(Qt.AlignCenter)
#         hint.setStyleSheet("font-size: 16px; color: #777;")
#         root.addWidget(hint)
#
#         self.setLayout(root)
#
#     # =========================
#     # Actions
#     # =========================
#     def _script_dir(self) -> str:
#         return os.path.dirname(os.path.abspath(__file__))
#
#     def start_live_predictions(self):
#         """Launch infer_4.py as a child process (touch button)."""
#         if self.infer_process and self.infer_process.poll() is None:
#             QMessageBox.information(self, "Already running", "Live Predictions is already running.")
#             return
#
#         infer_path = os.path.join(self._script_dir(), "infer_4.py")
#
#         if not os.path.exists(infer_path):
#             QMessageBox.critical(self, "Missing file", f"Cannot find infer_4.py at:\n{infer_path}")
#             return
#
#         # Use the same Python interpreter running this Qt app (venv-safe)
#         try:
#             self.infer_process = subprocess.Popen([sys.executable, infer_path])
#         except Exception as e:
#             QMessageBox.critical(self, "Failed to start", f"Could not start inference:\n{e}")
#
#     def open_data_labeling_placeholder(self):
#         QMessageBox.information(
#             self,
#             "Data Labeling",
#             "Data Labeling screen will be implemented later.\n\n(For now, use Live Predictions.)"
#         )
#
#     def ask_exit_pin(self):
#         """Ask for PIN (touch-friendly) then exit to desktop if correct."""
#         pin, ok = QInputDialog.getText(
#             self,
#             "Developer Access",
#             "Enter PIN:",
#             QLineEdit.Password
#         )
#
#         if not ok:
#             return
#
#         if pin != DEV_PIN:
#             QMessageBox.warning(self, "Access Denied", "Wrong PIN")
#             return
#
#         # Optional: stop inference before quitting launcher
#         self.stop_inference_if_running()
#         QApplication.quit()
#
#     def stop_inference_if_running(self):
#         """Try to stop infer_4.py gracefully."""
#         if not self.infer_process:
#             return
#
#         if self.infer_process.poll() is None:
#             try:
#                 # Try graceful termination
#                 self.infer_process.terminate()
#                 self.infer_process.wait(timeout=3)
#             except Exception:
#                 # Force kill if needed
#                 try:
#                     self.infer_process.kill()
#                 except Exception:
#                     pass
#
#         self.infer_process = None
#
#     # Optional: if user closes the window (rare in kiosk), still stop inference
#     def closeEvent(self, event):
#         self.stop_inference_if_running()
#         event.accept()
#
#
# def main():
#     app = QApplication(sys.argv)
#     w = MainWindow()
#     w.show()
#     sys.exit(app.exec())
#
#
# if __name__ == "__main__":
#     main()



import os
import sys
import subprocess

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QInputDialog,
    QLineEdit,
    QSizePolicy,
)

# =========================
# CONFIG
# =========================
DEV_PIN = "4321"  # <-- CHANGE THIS
APP_TITLE = "Opelka – Qualitätssicherungsmodul"


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.showFullScreen()

        self.infer_process: subprocess.Popen | None = None

        # ---------- Layout ----------
        root = QVBoxLayout()
        root.setContentsMargins(60, 40, 60, 40)
        root.setSpacing(25)

        # Top row: title + developer button
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)

        title = QLabel(APP_TITLE)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 36px; font-weight: 700;")
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        dev_btn = QPushButton("⚙  Entwickler-Modus")
        dev_btn.setFixedHeight(54)
        dev_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                border-radius: 10px;
                background: #444;
                color: white;
                padding: 6px 14px;
            }
            QPushButton:pressed { background: #333; }
        """)
        dev_btn.clicked.connect(self.ask_exit_pin)

        top_row.addWidget(title, stretch=1)
        top_row.addWidget(dev_btn, alignment=Qt.AlignRight)

        root.addLayout(top_row)

        # ---------- Main START button ----------
        start_btn = QPushButton("START")
        start_btn.setFixedHeight(150)
        start_btn.setStyleSheet("""
            QPushButton {
                font-size: 44px;
                font-weight: 700;
                border-radius: 18px;
                background-color: #2c7be5;
                color: white;
                padding: 18px;
            }
            QPushButton:pressed {
                background-color: #1a5bb8;
            }
        """)
        start_btn.clicked.connect(self.start_live_predictions)

        root.addStretch(2)
        root.addWidget(start_btn)
        root.addStretch(3)

        # (Data Labeling removed for now)
        # label_btn = QPushButton("Data Labeling")
        # label_btn.clicked.connect(self.open_data_labeling_placeholder)
        # root.addWidget(label_btn)

        hint = QLabel("Tip: Entwickler-Modus → PIN → zurück zum Desktop")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("font-size: 16px; color: #777;")
        root.addWidget(hint)

        self.setLayout(root)

    # =========================
    # Helpers
    # =========================
    def _script_dir(self) -> str:
        return os.path.dirname(os.path.abspath(__file__))

    # =========================
    # Actions
    # =========================
    def start_live_predictions(self):
        """Launch infer_4.py as a child process."""
        if self.infer_process and self.infer_process.poll() is None:
            QMessageBox.information(self, "Läuft bereits", "Live Predictions läuft bereits.")
            return

        infer_path = os.path.join(self._script_dir(), "infer_4.py")

        if not os.path.exists(infer_path):
            QMessageBox.critical(self, "Datei fehlt", f"infer_4.py nicht gefunden:\n{infer_path}")
            return

        try:
            # Use same interpreter as launcher (venv-safe)
            self.infer_process = subprocess.Popen([sys.executable, infer_path])
        except Exception as e:
            QMessageBox.critical(self, "Start fehlgeschlagen", f"Inferenz konnte nicht gestartet werden:\n{e}")

    def ask_exit_pin(self):
        """Ask for PIN then exit to desktop if correct."""
        pin, ok = QInputDialog.getText(
            self,
            "Entwickler-Modus",
            "PIN eingeben:",
            QLineEdit.Password
        )

        if not ok:
            return

        if pin != DEV_PIN:
            QMessageBox.warning(self, "Zugriff verweigert", "Falsche PIN")
            return

        self.stop_inference_if_running()
        QApplication.quit()

    def stop_inference_if_running(self):
        """Try to stop infer_4.py gracefully."""
        if not self.infer_process:
            return

        if self.infer_process.poll() is None:
            try:
                self.infer_process.terminate()
                self.infer_process.wait(timeout=3)
            except Exception:
                try:
                    self.infer_process.kill()
                except Exception:
                    pass

        self.infer_process = None

    def closeEvent(self, event):
        self.stop_inference_if_running()
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
