import os
import random
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

import cv2
import numpy as np

from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt6.QtGui import QIcon, QAction, QCursor
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon, QWidget, QInputDialog

from mood_engine import MoodEngine, MoodConfig, Verdict
from notifier import Notifier

# GLOBAL CONFIGURATION
try:
    import tensorflow as tf
except Exception:
    tf = None

MOTIVATION_LINES = [
    "Keep up the good work",
    "Nice focus — stay consistent!",
    "You're doing great. One step at a time.",
    "Good vibe. Keep going",
]

BREAK_LINES = [
    "You seem a bit tense. Take a 5–10 min break.",
    "Consider a quick break: water + stretch.",
    "Time to reset. Walk for 2 minutes.",
]


# UTILITIES & LOGIC CLASSES
def majority_vote(verdicts: List[Verdict]) -> Verdict:
    if not verdicts:
        return Verdict.NO_FACE
    counts: Dict[Verdict, int] = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    if any(v != Verdict.NO_FACE for v in verdicts):
        counts.pop(Verdict.NO_FACE, None)
    return max(counts, key=counts.get)


class StudySession:

    def __init__(self, duration_minutes: int):
        self.duration = timedelta(minutes=duration_minutes)
        self.start_time: Optional[datetime] = None

        # Pause Logic
        self.is_paused = False
        self.pause_start_time: Optional[datetime] = None
        self.total_paused_duration = timedelta(0)

        # Notification Flags
        self.notified_60 = False
        self.notified_30 = False
        self.notified_10 = False

    def set_duration(self, minutes: int) -> None:
        self.duration = timedelta(minutes=minutes)

    def start(self) -> None:
        self.start_time = datetime.now()
        self.is_paused = False
        self.pause_start_time = None
        self.total_paused_duration = timedelta(0)

        self.notified_60 = False
        self.notified_30 = False
        self.notified_10 = False

    def stop(self) -> None:
        self.start_time = None
        self.is_paused = False

    def pause(self) -> None:
        if self.start_time and not self.is_paused:
            self.is_paused = True
            self.pause_start_time = datetime.now()

    def resume(self) -> None:
        if self.start_time and self.is_paused and self.pause_start_time:
            self.is_paused = False
            paused_slice = datetime.now() - self.pause_start_time
            self.total_paused_duration += paused_slice
            self.pause_start_time = None

    def remaining(self) -> Optional[timedelta]:
        if self.start_time is None:
            return None

        now = datetime.now()
        current_pause_drift = timedelta(0)
        if self.is_paused and self.pause_start_time:
            current_pause_drift = now - self.pause_start_time

        elapsed = (now - self.start_time) - self.total_paused_duration - current_pause_drift
        rem = self.duration - elapsed

        if rem.total_seconds() < 0:
            return timedelta(0)
        return rem

    def check_timer_notifications(self, notifier: Notifier) -> None:
        if self.is_paused or self.start_time is None:
            return

        rem = self.remaining()
        if rem is None:
            return

        minutes_left = int(rem.total_seconds() // 60)

        if minutes_left <= 60 and not self.notified_60 and minutes_left > 30:
            self.notified_60 = True
            notifier.info("Study timer", "1 hour left. Stay focused!")
        if minutes_left <= 30 and not self.notified_30 and minutes_left > 10:
            self.notified_30 = True
            notifier.info("Study timer", "30 minutes left. Almost there!")
        if minutes_left <= 10 and not self.notified_10 and minutes_left > 0:
            self.notified_10 = True
            notifier.info("Study timer", "10 minutes left. Finish strong!")

        if rem.total_seconds() <= 0:
            notifier.info("Study timer", "Session complete. Take a proper break.")
            self.stop()


class FaceDetector:
    def __init__(self):
        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_face_box(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0:
            return None
        x, y, fw, fh = faces[0]
        return (x, y, x + fw, y + fh)


class MoodModel:
    def __init__(self, model_path: str):
        if tf is None:
            raise RuntimeError("TensorFlow is not installed.")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found: %s" % model_path)

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.img_h = 224
        self.img_w = 224
        self.index_to_label = {0: "negative", 1: "neutral", 2: "positive"}

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        x = img_rgb.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        return x

    def predict_verdict_from_face(self, face_bgr: np.ndarray) -> Verdict:
        x = self._preprocess(face_bgr)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = self.index_to_label.get(idx, "neutral")
        if label == "negative":
            return Verdict.NEGATIVE
        if label == "positive":
            return Verdict.POSITIVE
        return Verdict.NEUTRAL


# BACKGROUND WORKER THREAD
class CameraWorker(QThread):
    result_ready = pyqtSignal(object)

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.detector = None
        self.model = None
        self.is_initialized = False

    def run(self):
        if not self.is_initialized:
            try:
                self.detector = FaceDetector()
                self.model = MoodModel(self.model_path)
                self.is_initialized = True
            except Exception as e:
                print(f"Initialization Error: {e}")
                self.result_ready.emit(Verdict.NO_FACE)
                return

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.result_ready.emit(Verdict.NO_FACE)
            return

        start = time.time()
        verdicts: List[Verdict] = []

        try:
            while time.time() - start < 4.0:
                ok, frame = cap.read()
                if not ok:
                    continue

                box = self.detector.detect_face_box(frame)
                if box is None:
                    verdicts.append(Verdict.NO_FACE)
                    time.sleep(0.15)
                    continue

                x1, y1, x2, y2 = box
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    verdicts.append(Verdict.NO_FACE)
                    time.sleep(0.15)
                    continue

                v = self.model.predict_verdict_from_face(face)
                verdicts.append(v)
                time.sleep(0.15)
        except Exception as e:
            pass
        finally:
            cap.release()

        v_final = majority_vote(verdicts)
        v_final = Verdict.NEUTRAL if v_final == Verdict.NO_FACE else v_final
        self.result_ready.emit(v_final)


# MAIN APPLICATION CLASS
class StudyBuddyApp(QWidget):
    def __init__(self):
        super().__init__()

        # 1. Background Execution Mode (Hidden Window)
        self.setWindowFlags(Qt.WindowType.Tool)
        self.hide()

        # 2. Paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, "assets", "icon.png")
        self.model_path = os.path.join(base_dir, "mobileNet(3 classes).h5")

        # 3. Setup Icon Globally
        if os.path.exists(icon_path):
            app_icon = QIcon(icon_path)
            self.setWindowIcon(app_icon)
            QApplication.setWindowIcon(app_icon)
        else:
            app_icon = self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon)

        # 4. Setup System Tray Icon
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(app_icon)
        self.tray.setToolTip("Study Buddy: Ready")

        # 5. Initialize Logic Controllers
        self.notifier = Notifier(self.tray)
        self.engine = MoodEngine(MoodConfig())
        self.session = StudySession(duration_minutes=120)
        self.camera_enabled = False
        self.worker = None

        # 6. Build the Context Menu
        self.menu = QMenu(self)

        # TIME DISPLAY ACTION
        self.act_time_display = QAction("Ready (120m)", self)
        self.act_time_display.setEnabled(False)

        # START
        self.act_start = QAction("Start Session", self)
        self.act_start.triggered.connect(self.start_session)

        # DURATION SUB-MENU
        self.menu_duration = QMenu("Set Duration", self)

        # Add preset durations
        presets = [25, 45, 60, 90, 120]
        for mins in presets:
            act = QAction(f"{mins} minutes", self)
            # Use default argument m=mins to capture the value correctly in the lambda
            act.triggered.connect(lambda checked, m=mins: self.set_duration(m))
            self.menu_duration.addAction(act)

        self.menu_duration.addSeparator()

        # Custom duration action
        act_custom = QAction("Custom...", self)
        act_custom.triggered.connect(self.ask_custom_duration)
        self.menu_duration.addAction(act_custom)

        # PAUSE
        self.act_pause = QAction("Pause Session", self)
        self.act_pause.setCheckable(True)
        self.act_pause.triggered.connect(self.toggle_pause)
        self.act_pause.setEnabled(False)

        # STOP
        self.act_stop = QAction("Stop Session", self)
        self.act_stop.triggered.connect(self.stop_session)
        self.act_stop.setEnabled(False)

        # CAMERA
        self.act_cam = QAction("Enable Camera Scanning", self)
        self.act_cam.setCheckable(True)
        self.act_cam.setChecked(False)
        self.act_cam.triggered.connect(self.toggle_camera)

        # QUIT
        self.act_quit = QAction("Quit App", self)
        self.act_quit.triggered.connect(QApplication.instance().quit)

        # ASSEMBLE MENU
        self.menu.addAction(self.act_time_display)
        self.menu.addSeparator()
        self.menu.addAction(self.act_start)
        self.menu.addMenu(self.menu_duration)
        self.menu.addAction(self.act_pause)
        self.menu.addAction(self.act_stop)
        self.menu.addSeparator()
        self.menu.addAction(self.act_cam)
        self.menu.addSeparator()
        self.menu.addAction(self.act_quit)

        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self.on_tray_activated)

        self.tray.show()

        self.tick_timer = QTimer(self)
        self.tick_timer.setInterval(1000)
        self.tick_timer.timeout.connect(self.on_tick)

        self.scan_timer = QTimer(self)
        self.scan_timer.setInterval(60000)
        self.scan_timer.timeout.connect(self.trigger_scan)

        print("App started")
        self.notifier.info("Study Buddy", "App running in background.")

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.menu.exec(QCursor.pos())

    # SESSION SETTINGS

    def set_duration(self, minutes):
        self.session.set_duration(minutes)
        self.notifier.info("Settings", f"Session duration set to {minutes} minutes.")

        if self.session.start_time is None:
            self.act_time_display.setText(f"Ready ({minutes}m)")
            self.tray.setToolTip(f"Study Buddy: Ready ({minutes}m)")
        else:
            pass

    def ask_custom_duration(self):
        minutes, ok = QInputDialog.getInt(
            None, "Custom Duration",
            "Enter study session duration (minutes):",
            value=60, min=1, max=1440
        )
        if ok:
            self.set_duration(minutes)

    # SESSION CONTROL

    def start_session(self):
        self.session.start()
        self.tick_timer.start()
        if self.camera_enabled:
            self.scan_timer.start()

        self.act_start.setEnabled(False)
        self.menu_duration.setEnabled(True)
        self.act_stop.setEnabled(True)
        self.act_pause.setEnabled(True)
        self.act_pause.setChecked(False)
        self.act_pause.setText("Pause Session")

        self.notifier.info("Study Buddy", "Session started. Good luck!")
        self.update_time_display()

    def stop_session(self):
        self.session.stop()
        self.tick_timer.stop()
        self.scan_timer.stop()

        self.act_start.setEnabled(True)
        self.act_stop.setEnabled(False)
        self.act_pause.setEnabled(False)
        self.act_pause.setChecked(False)
        self.act_time_display.setText("Session Stopped")
        self.tray.setToolTip("Study Buddy: Stopped")

        self.notifier.info("Study Buddy", "Session stopped.")

    def toggle_pause(self, checked):
        if self.session.start_time is None:
            return

        if checked:
            self.session.pause()
            self.scan_timer.stop()
            self.act_pause.setText("Resume Session")
            self.notifier.info("Paused", "Timer paused.")
        else:
            self.session.resume()
            if self.camera_enabled:
                self.scan_timer.start()
            self.act_pause.setText("Pause Session")
            self.notifier.info("Resumed", "Timer resumed.")

        self.update_time_display()

    def toggle_camera(self, checked):
        self.camera_enabled = bool(checked)
        if self.session.start_time is not None and not self.session.is_paused:
            if self.camera_enabled:
                self.scan_timer.start()
                self.notifier.info("Camera", "Scanning enabled.")
            else:
                self.scan_timer.stop()
                self.notifier.info("Camera", "Scanning disabled.")

    def on_tick(self):
        self.update_time_display()
        self.session.check_timer_notifications(self.notifier)

    def update_time_display(self):
        rem = self.session.remaining()
        if rem is None:
            return

        total_seconds = int(rem.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

        status_icon = "[Paused]" if self.session.is_paused else ""
        self.act_time_display.setText(f"{status_icon} Time Left: {time_str}")

        status_text = "Paused" if self.session.is_paused else "Running"
        self.tray.setToolTip(f"Study Buddy: {status_text}\nTime Left: {time_str}")

    def trigger_scan(self):
        if not self.camera_enabled or self.session.is_paused:
            return

        self.worker = CameraWorker(self.model_path)
        self.worker.result_ready.connect(self.handle_scan_result)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def handle_scan_result(self, verdict):
        now = datetime.now()

        self.engine.push(verdict)

        print(f"[{now.strftime('%H:%M:%S')}] Verdict: {verdict.name}")

        if self.engine.should_break_alert(now):
            self.engine.mark_break_shown(now)
            self.notifier.warn("Take a break", random.choice(BREAK_LINES))
            return

        if self.engine.should_motivate(now):
            self.engine.mark_motivation_shown(now)
            self.notifier.info("Nice!", random.choice(MOTIVATION_LINES))


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    window = StudyBuddyApp()

    sys.exit(app.exec())