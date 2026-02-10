#
# import os
# import sys
# import subprocess
#
# from PySide6.QtCore import Qt, QTimer
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
#         # ---------- Layout ----------
#         root = QVBoxLayout()
#         root.setContentsMargins(60, 40, 60, 40)
#         root.setSpacing(25)
#
#         # Top row: title + developer button
#         top_row = QHBoxLayout()
#         top_row.setContentsMargins(0, 0, 0, 0)
#
#         title = QLabel(APP_TITLE)
#         title.setAlignment(Qt.AlignCenter)
#         title.setStyleSheet("font-size: 36px; font-weight: 700;")
#         title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
#
#         dev_btn = QPushButton("⚙  Entwickler-Modus")
#         dev_btn.setFixedHeight(54)
#         dev_btn.setStyleSheet("""
#             QPushButton {
#                 font-size: 18px;
#                 border-radius: 10px;
#                 background: #444;
#                 color: white;
#                 padding: 6px 14px;
#             }
#             QPushButton:pressed { background: #333; }
#         """)
#         dev_btn.clicked.connect(self.ask_exit_pin)
#
#         top_row.addWidget(title, stretch=1)
#         top_row.addWidget(dev_btn, alignment=Qt.AlignRight)
#         root.addLayout(top_row)
#
#         # ---------- Main START button ----------
#         self.start_btn = QPushButton("START")
#         self.start_btn.setFixedHeight(150)
#         self.start_btn.setStyleSheet("""
#             QPushButton {
#                 font-size: 44px;
#                 font-weight: 700;
#                 border-radius: 18px;
#                 background-color: #2c7be5;
#                 color: white;
#                 padding: 18px;
#             }
#             QPushButton:pressed {
#                 background-color: #1a5bb8;
#             }
#             QPushButton:disabled {
#                 background-color: #94bdf5;
#                 color: #ffffff;
#             }
#         """)
#         self.start_btn.clicked.connect(self.start_live_predictions)
#
#         # Helper text under button
#         info = QLabel("Nach dem Drücken von START wird das System geladen.\nBitte warten Sie einige Sekunden.")
#         info.setAlignment(Qt.AlignCenter)
#         info.setStyleSheet("font-size: 16px; color: #777;")
#
#         root.addStretch(2)
#         root.addWidget(self.start_btn)
#         root.addWidget(info)
#         root.addStretch(3)
#
#         hint = QLabel("Tip: Entwickler-Modus → PIN → zurück zum Desktop")
#         hint.setAlignment(Qt.AlignCenter)
#         hint.setStyleSheet("font-size: 16px; color: #777;")
#         root.addWidget(hint)
#
#         self.setLayout(root)
#
#         # ---------- Timer to reset UI when infer_4.py exits ----------
#         self.proc_timer = QTimer(self)
#         self.proc_timer.setInterval(500)  # ms
#         self.proc_timer.timeout.connect(self._poll_infer_process)
#         self.proc_timer.start()
#
#     def _script_dir(self) -> str:
#         return os.path.dirname(os.path.abspath(__file__))
#
#     def _poll_infer_process(self):
#         """If infer_4.py ended, reset the START button state."""
#         if self.infer_process and self.infer_process.poll() is not None:
#             self.infer_process = None
#             self.start_btn.setText("START")
#             self.start_btn.setEnabled(True)
#
#     def start_live_predictions(self):
#         """Launch infer_4.py as a child process."""
#         if self.infer_process and self.infer_process.poll() is None:
#             QMessageBox.information(self, "Läuft bereits", "Das System läuft bereits.")
#             return
#
#         infer_path = os.path.join(self._script_dir(), "infer_4.py")
#
#         if not os.path.exists(infer_path):
#             QMessageBox.critical(self, "Datei fehlt", f"infer_4.py nicht gefunden:\n{infer_path}")
#             return
#
#         # --- UI feedback ---
#         self.start_btn.setText("System startet … bitte warten")
#         self.start_btn.setEnabled(False)
#         QApplication.processEvents()  # force UI update
#
#         try:
#             self.infer_process = subprocess.Popen([sys.executable, infer_path])
#             # Once launched, show "running"
#             self.start_btn.setText("Läuft …")
#         except Exception as e:
#             QMessageBox.critical(self, "Start fehlgeschlagen", f"Inferenz konnte nicht gestartet werden:\n{e}")
#             self.infer_process = None
#             self.start_btn.setText("START")
#             self.start_btn.setEnabled(True)
#
#     def ask_exit_pin(self):
#         """Ask for PIN then exit to desktop if correct."""
#         pin, ok = QInputDialog.getText(
#             self,
#             "Entwickler-Modus",
#             "PIN eingeben:",
#             QLineEdit.Password
#         )
#
#         if not ok:
#             return
#
#         if pin != DEV_PIN:
#             QMessageBox.warning(self, "Zugriff verweigert", "Falsche PIN")
#             return
#
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
#                 self.infer_process.terminate()
#                 self.infer_process.wait(timeout=3)
#             except Exception:
#                 try:
#                     self.infer_process.kill()
#                 except Exception:
#                     pass
#
#         self.infer_process = None
#         self.start_btn.setText("START")
#         self.start_btn.setEnabled(True)
#
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
#


import os
import sys
import json
import uuid
import tempfile
import subprocess

from PySide6.QtCore import Qt, QTimer
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
        self.last_result_json: str = ""
        self.result_popup_shown: bool = False

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
        self.start_btn = QPushButton("START")
        self.start_btn.setFixedHeight(150)
        self.start_btn.setStyleSheet("""
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
            QPushButton:disabled {
                background-color: #94bdf5;
                color: #ffffff;
            }
        """)
        self.start_btn.clicked.connect(self.start_live_predictions)

        info = QLabel("Nach dem Drücken von START wird das System geladen.\nBitte warten Sie einige Sekunden.")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("font-size: 16px; color: #777;")

        root.addStretch(2)
        root.addWidget(self.start_btn)
        root.addWidget(info)
        root.addStretch(3)

        hint = QLabel("Tip: Entwickler-Modus → PIN → zurück zum Desktop")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("font-size: 16px; color: #777;")
        root.addWidget(hint)

        self.setLayout(root)

        # Poll child process so we can reset UI + show results popup after infer exits
        self.proc_timer = QTimer(self)
        self.proc_timer.setInterval(500)
        self.proc_timer.timeout.connect(self._poll_infer_process)
        self.proc_timer.start()

    def _script_dir(self) -> str:
        return os.path.dirname(os.path.abspath(__file__))

    def _poll_infer_process(self):
        if self.infer_process and self.infer_process.poll() is not None:
            # Process ended
            self.infer_process = None

            # Reset UI
            self.start_btn.setText("START")
            self.start_btn.setEnabled(True)

            # Show results once (if available)
            if not self.result_popup_shown:
                self.result_popup_shown = True
                self._show_results_popup()

    def start_live_predictions(self):
        if self.infer_process and self.infer_process.poll() is None:
            QMessageBox.information(self, "Läuft bereits", "Das System läuft bereits.")
            return

        infer_path = os.path.join(self._script_dir(), "infer_4.py")
        if not os.path.exists(infer_path):
            QMessageBox.critical(self, "Datei fehlt", f"infer_4.py nicht gefunden:\n{infer_path}")
            return

        # Fresh result path each run (avoid stale file)
        run_id = uuid.uuid4().hex[:10]
        self.last_result_json = os.path.join(tempfile.gettempdir(), f"qc_result_{run_id}.json")
        self.result_popup_shown = False

        # UI feedback
        self.start_btn.setText("System startet … bitte warten")
        self.start_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            # Pass JSON path to infer_4.py
            self.infer_process = subprocess.Popen(
                [sys.executable, infer_path, "--result-json", self.last_result_json]
            )
            self.start_btn.setText("Läuft …")
        except Exception as e:
            QMessageBox.critical(self, "Start fehlgeschlagen", f"Inferenz konnte nicht gestartet werden:\n{e}")
            self.infer_process = None
            self.start_btn.setText("START")
            self.start_btn.setEnabled(True)

    def _show_results_popup(self):
        # If infer didn't write a result file, don't block user
        if not self.last_result_json or not os.path.exists(self.last_result_json):
            QMessageBox.information(self, "Zählung", "Keine Ergebnisdatei gefunden.")
            return

        try:
            with open(self.last_result_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            total = int(data.get("total", 0))
            good = int(data.get("good", 0))
            bad = int(data.get("bad", 0))
            uncertain = int(data.get("uncertain", 0))

            msg = (
                f"Gesamt: {total}\n"
                f"Gut: {good}\n"
                f"Schlecht: {bad}\n"
                f"Unklar: {uncertain}"
            )
            QMessageBox.information(self, "Zählung abgeschlossen", msg)

        except Exception as e:
            QMessageBox.warning(self, "Zählung", f"Fehler beim Lesen der Ergebnisse:\n{e}")

    def ask_exit_pin(self):
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

        # Stop inference if still running
        self.stop_inference_if_running()
        QApplication.quit()

    def stop_inference_if_running(self):
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
        self.start_btn.setText("START")
        self.start_btn.setEnabled(True)

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
