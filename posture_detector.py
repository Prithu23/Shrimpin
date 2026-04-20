"""
Posture Detector - Are You Shrimpin'?
Detects hunching using OpenCV + MediaPipe Pose.
Exposes hunch count on http://localhost:8765/status so posture.html can poll it.

Install:
    pip install opencv-python mediapipe

Controls:
    q - Quit
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
import math
import threading
import urllib.request
import os
import winsound


# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------

NOSE          = 0
LEFT_EAR      = 7
RIGHT_EAR     = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER= 12
LEFT_HIP      = 23
RIGHT_HIP     = 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def angle_at_b(a, b, c):
    """Angle (degrees) at point b formed by a-b-c."""
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.sqrt(ba[0]**2+ba[1]**2) * math.sqrt(bc[0]**2+bc[1]**2)
    if mag < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag))))


def neck_angle(lm):
    """Average ear-shoulder-hip angle. Increases when you hunch forward."""
    left  = angle_at_b(lm[LEFT_EAR],   lm[LEFT_SHOULDER],  lm[LEFT_HIP])
    right = angle_at_b(lm[RIGHT_EAR],  lm[RIGHT_SHOULDER], lm[RIGHT_HIP])
    return (left + right) / 2.0


def play_beep():
    threading.Thread(
        target=lambda: winsound.Beep(880, 300),
        daemon=True
    ).start()


def put_text(frame, text, pos, scale=0.65, color=(255,255,255), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (pos[0]+1, pos[1]+1), font, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(frame, text, pos,                  font, scale, color,   thickness,   cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Shared state (read by HTTP server, written by detector)
# ---------------------------------------------------------------------------

shared_state  = {"count": 0, "is_hunching": False}
latest_frame  = None
frame_lock    = threading.Lock()


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
# Detector state
# ---------------------------------------------------------------------------

HUNCH_THRESHOLD = 145   # angle below this = hunching
EXIT_THRESHOLD  = 150   # angle must rise above this to clear hunch (hysteresis)
REQUIRED_FRAMES = 8     # consecutive bad frames before counting as a hunch


class PostureDetector:
    def __init__(self):
        self.is_hunching = False
        self.consec_bad  = 0
        self.hunch_count = 0
        self.last_beep   = 0.0

    def analyze(self, lm):
        angle = neck_angle(lm)

        is_bad = angle < (EXIT_THRESHOLD if self.is_hunching else HUNCH_THRESHOLD)

        if is_bad:
            self.consec_bad += 1
        else:
            self.consec_bad = 0

        # Good → hunching
        if not self.is_hunching and self.consec_bad >= REQUIRED_FRAMES:
            self.is_hunching = True
            self.consec_bad  = 0
            self.hunch_count += 1
            shared_state["count"]       = self.hunch_count
            shared_state["is_hunching"] = True
            now = time.time()
            if now - self.last_beep > 3.0:
                play_beep()
                self.last_beep = now

        # Hunching → good
        if self.is_hunching and not is_bad:
            self.is_hunching            = False
            shared_state["is_hunching"] = False

        return angle


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_hud(frame, detector, angle):
    h, w = frame.shape[:2]

    if detector.is_hunching:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    status_color = (0, 0, 255) if detector.is_hunching else (0, 220, 0)
    status_text  = "HUNCHING!" if detector.is_hunching else "Good posture"
    put_text(frame, f"Hunch count: {detector.hunch_count}", (15, 35), 0.75, (255, 255, 255))
    put_text(frame, f"Status: {status_text}",               (15, 65), 0.65, status_color)
    put_text(frame, f"Angle: {angle:.1f}  (hunch < {HUNCH_THRESHOLD})", (15, 92), 0.5, (180, 180, 180))

    if detector.is_hunching and time.time() % 1.0 < 0.55:
        msg = "STRAIGHTEN UP!"
        sz  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
        put_text(frame, msg, ((w - sz[0]) // 2, (h + sz[1]) // 2), 1.1, (0, 0, 255), 3)

    put_text(frame, "q: Quit", (15, h - 15), 0.45, (150, 150, 150), 1)


POSE_CONNECTIONS = [
    (LEFT_EAR, LEFT_SHOULDER), (RIGHT_EAR, RIGHT_SHOULDER),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (NOSE, LEFT_EAR), (NOSE, RIGHT_EAR),
]

def _vis(lm):
    return getattr(lm, 'visibility', 1.0) or 1.0


def draw_skeleton(frame, lm_list, is_bad):
    h, w = frame.shape[:2]
    pt_col   = (0, 0, 220) if is_bad else (0, 200, 0)
    line_col = (0, 0, 180) if is_bad else (0, 160, 0)

    for i, j in POSE_CONNECTIONS:
        a, b = lm_list[i], lm_list[j]
        if _vis(a) > 0.5 and _vis(b) > 0.5:
            cv2.line(frame,
                     (int(a.x*w), int(a.y*h)),
                     (int(b.x*w), int(b.y*h)),
                     line_col, 2, cv2.LINE_AA)

    for idx in [NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]:
        lm = lm_list[idx]
        if _vis(lm) > 0.5:
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(frame, (cx, cy), 6, pt_col,       -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 6, (255,255,255),  1, cv2.LINE_AA)


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
    print("Status server running on http://localhost:8765/status")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    detector = PostureDetector()

    print("=== Are You Shrimpin'? ===")
    print(f"Hunch threshold: angle < {HUNCH_THRESHOLD} degrees")
    print("Controls: q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = int(time.time() * 1000)
        result   = landmarker.detect_for_video(mp_image, ts_ms)

        if result.pose_landmarks:
            lm    = result.pose_landmarks[0]
            angle = detector.analyze(lm)
            draw_skeleton(frame, lm, detector.is_hunching)
            draw_hud(frame, detector, angle)
        else:
            put_text(frame, "No pose detected — move into frame", (20, 40), 0.65, (0, 0, 255))

        _, jpeg = cv2.imencode(".jpg", frame)
        with frame_lock:
            latest_frame = jpeg.tobytes()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    print(f"\nSession ended — total hunches: {detector.hunch_count}")


if __name__ == "__main__":
    main()
