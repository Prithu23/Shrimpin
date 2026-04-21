"""
Posture Detector - Shrimpin'
Ratio-based posture detection with calibration, forward hunch + sideways tilt.

Setup:
    pip install opencv-python mediapipe numpy

Usage:
    python posture_detector.py

Controls:
    q - Quit
    c - Recalibrate
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
import threading
import urllib.request
import os


# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------

NOSE           = 0
LEFT_EYE       = 2
RIGHT_EYE      = 5
LEFT_EAR       = 7
RIGHT_EAR      = 8
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP       = 23
RIGHT_HIP      = 24

TRACKED_INDICES = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
                   LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

SKELETON_CONNECTIONS = [
    (LEFT_EAR, LEFT_SHOULDER), (RIGHT_EAR, RIGHT_SHOULDER),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (NOSE, LEFT_EAR), (NOSE, RIGHT_EAR),
    (LEFT_EYE, NOSE), (RIGHT_EYE, NOSE),
]

# ---------------------------------------------------------------------------
# Sensitivity presets  (entry_offset, exit_offset, required_frames)
# ---------------------------------------------------------------------------

SENSITIVITY = {
    "strict":  (0.05, 0.02, 8),
    "normal":  (0.08, 0.04, 10),
    "relaxed": (0.12, 0.06, 15),
}

# ---------------------------------------------------------------------------
# Shared state (read by HTTP server)
# ---------------------------------------------------------------------------

shared_state = {
    "count": 0,
    "is_hunching": False,
    "calibrated": False,
    "countdown": 3,
    "cal_progress": 0.0,
}
latest_frame = None
frame_lock = threading.Lock()


# ---------------------------------------------------------------------------
# HTTP server — /status (JSON) + /video (MJPEG stream)
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/video":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    if frame is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.033)
            except Exception:
                pass
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(shared_state).encode())

    def log_message(self, *_):
        pass


def start_status_server(port=8765):
    ThreadingHTTPServer(("localhost", port), _Handler).serve_forever()


# ---------------------------------------------------------------------------
# PostureDetector
# ---------------------------------------------------------------------------

class PostureDetector:
    CALIBRATION_WAIT   = 3.0
    CALIBRATION_FRAMES = 45

    def __init__(self, sensitivity="normal"):
        self.calibrated             = False
        self.calibration_data       = []
        self.calibration_tilt_data  = []
        self.calibration_wait_start = None
        self.baseline_ratio         = None
        self.baseline_tilt          = None
        self.entry_threshold        = 999.0
        self.exit_threshold         = 999.0
        self.required_frames        = 10
        self.is_shrimping           = False
        self.consecutive_bad        = 0
        self.shrimp_count           = 0
        self.bad_posture_start      = None
        self.set_sensitivity(sensitivity)

    def set_sensitivity(self, level):
        self.sensitivity = level
        entry_off, exit_off, req = SENSITIVITY[level]
        self.required_frames = req
        if self.baseline_ratio is not None:
            self.entry_threshold = self.baseline_ratio + entry_off
            self.exit_threshold  = self.baseline_ratio + exit_off

    def reset_calibration(self):
        self.calibrated             = False
        self.calibration_data       = []
        self.calibration_tilt_data  = []
        self.calibration_wait_start = None
        self.baseline_ratio         = None
        self.baseline_tilt          = None
        self.entry_threshold        = 999.0
        self.exit_threshold         = 999.0
        self.is_shrimping           = False
        self.consecutive_bad        = 0
        self.bad_posture_start      = None

    def _calc_shrimp_ratio(self, lm):
        nose_y           = lm[NOSE].y
        eye_avg_y        = (lm[LEFT_EYE].y + lm[RIGHT_EYE].y) / 2.0
        shoulder_avg_y   = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2.0
        eyes_to_nose     = nose_y - eye_avg_y
        nose_to_shoulder = shoulder_avg_y - nose_y
        if nose_to_shoulder <= 0.001:
            return 999.0
        return eyes_to_nose / nose_to_shoulder

    def _calc_sideways_tilt(self, lm):
        return abs(lm[LEFT_EAR].y - lm[RIGHT_EAR].y)

    def calibration_progress(self):
        if self.calibration_wait_start is None:
            return self.CALIBRATION_WAIT, 0, self.CALIBRATION_FRAMES
        elapsed   = time.time() - self.calibration_wait_start
        countdown = max(0.0, self.CALIBRATION_WAIT - elapsed)
        return countdown, len(self.calibration_data), self.CALIBRATION_FRAMES

    def calibrate_frame(self, lm):
        now = time.time()
        if self.calibration_wait_start is None:
            self.calibration_wait_start = now

        if now - self.calibration_wait_start < self.CALIBRATION_WAIT:
            return False

        self.calibration_data.append(self._calc_shrimp_ratio(lm))
        self.calibration_tilt_data.append(self._calc_sideways_tilt(lm))

        if len(self.calibration_data) >= self.CALIBRATION_FRAMES:
            self.baseline_ratio = sum(self.calibration_data) / len(self.calibration_data)
            self.baseline_tilt  = sum(self.calibration_tilt_data) / len(self.calibration_tilt_data)
            entry_off, exit_off, _ = SENSITIVITY[self.sensitivity]
            self.entry_threshold = self.baseline_ratio + entry_off
            self.exit_threshold  = self.baseline_ratio + exit_off
            self.calibrated      = True
            return True

        return False

    def analyze_posture(self, lm):
        ratio = self._calc_shrimp_ratio(lm)
        tilt  = self._calc_sideways_tilt(lm)

        entry_off, exit_off, _ = SENSITIVITY[self.sensitivity]
        forward_bad  = ratio > (self.exit_threshold  if self.is_shrimping else self.entry_threshold)
        sideways_bad = tilt  > (self.baseline_tilt + exit_off if self.is_shrimping else self.baseline_tilt + entry_off)
        is_bad = forward_bad or sideways_bad

        if is_bad:
            self.consecutive_bad += 1
        else:
            self.consecutive_bad = 0

        if not self.is_shrimping and self.consecutive_bad >= self.required_frames:
            self.is_shrimping    = True
            self.consecutive_bad = 0
            self.shrimp_count   += 1
            self.bad_posture_start = time.time()

        if self.is_shrimping and self.consecutive_bad == 0 and ratio <= self.exit_threshold:
            self.is_shrimping      = False
            self.bad_posture_start = None

        return {
            "is_shrimping":    self.is_shrimping,
            "shrimp_count":    self.shrimp_count,
            "ratio":           ratio,
            "entry_threshold": self.entry_threshold,
            "baseline_ratio":  self.baseline_ratio,
        }

    def draw_skeleton(self, frame, lm, is_bad):
        h, w = frame.shape[:2]
        pt_col   = (0, 0, 220) if is_bad else (0, 200, 0)
        line_col = (0, 0, 180) if is_bad else (0, 160, 0)
        for i, j in SKELETON_CONNECTIONS:
            a, b = lm[i], lm[j]
            if getattr(a, 'visibility', 1) > 0.5 and getattr(b, 'visibility', 1) > 0.5:
                cv2.line(frame, (int(a.x*w), int(a.y*h)), (int(b.x*w), int(b.y*h)), line_col, 2, cv2.LINE_AA)
        for idx in TRACKED_INDICES:
            p = lm[idx]
            if getattr(p, 'visibility', 1) > 0.5:
                cx, cy = int(p.x*w), int(p.y*h)
                cv2.circle(frame, (cx, cy), 6, pt_col, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _put(frame, text, pos, scale=0.65, color=(255, 255, 255), thickness=2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (pos[0]+1, pos[1]+1), font, scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
        cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)

    def draw_calibration_overlay(self, frame):
        h, w = frame.shape[:2]
        countdown, collected, needed = self.calibration_progress()
        if countdown > 0:
            self._put(frame, f"Sit straight... {int(countdown)+1}",
                      (w//2 - 160, h//2), 0.85, (0, 255, 255), 2)
        else:
            self._put(frame, "Calibrating...", (w//2 - 120, h//2 - 20), 0.75, (0, 255, 255), 2)
            bar_w = 300
            bx, by = (w - bar_w) // 2, h//2 + 10
            prog = collected / needed if needed > 0 else 0
            cv2.rectangle(frame, (bx, by), (bx+bar_w, by+20), (100, 100, 100), 2)
            cv2.rectangle(frame, (bx, by), (bx+int(bar_w*prog), by+20), (0, 255, 0), -1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_FILE = "pose_landmarker_lite.task"


def main():
    if not os.path.exists(MODEL_FILE):
        print("Downloading pose model (~5 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("Done.")

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_base.BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    global latest_frame

    threading.Thread(target=start_status_server, daemon=True).start()
    print("Status server on http://localhost:8765/status")
    print("Video stream on  http://localhost:8765/video")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    detector    = PostureDetector(sensitivity="normal")
    calibrating = True
    print("Sit in your best posture for calibration...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            if calibrating:
                detector.draw_skeleton(frame, lm, False)
                detector.draw_calibration_overlay(frame)
                countdown, collected, needed = detector.calibration_progress()
                shared_state.update({
                    "calibrated":    False,
                    "countdown":     round(countdown, 1),
                    "cal_progress":  round(collected / needed, 2) if needed > 0 else 0,
                    "is_hunching":   False,
                })
                if detector.calibrate_frame(lm):
                    calibrating = False
                    print(f"Calibrated! Baseline ratio: {detector.baseline_ratio:.3f}")
                    shared_state["calibrated"] = True
            else:
                analysis = detector.analyze_posture(lm)
                detector.draw_skeleton(frame, lm, analysis["is_shrimping"])
                shared_state.update({
                    "count":        analysis["shrimp_count"],
                    "is_hunching":  analysis["is_shrimping"],
                    "calibrated":   True,
                    "countdown":    0,
                    "cal_progress": 1.0,
                    "ratio":        round(analysis["ratio"], 3),
                    "threshold":    round(analysis["entry_threshold"], 3),
                    "baseline":     round(analysis["baseline_ratio"], 3),
                })
                if analysis["is_shrimping"]:
                    h, w = frame.shape[:2]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
                    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        else:
            detector._put(frame, "No pose detected", (20, 40), 0.65, (0, 0, 255))

        _, jpeg = cv2.imencode(".jpg", frame)
        with frame_lock:
            latest_frame = jpeg.tobytes()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            detector.reset_calibration()
            calibrating = True
            print("Recalibrating...")

    cap.release()
    landmarker.close()
    print(f"\nTotal shrimp events: {detector.shrimp_count}")


if __name__ == "__main__":
    main()
