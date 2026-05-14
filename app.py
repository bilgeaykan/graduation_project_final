import time
import threading
import csv
import json
import webbrowser
from collections import deque
from pathlib import Path

import numpy as np
import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_file,
)
from ultralytics import YOLO
from werkzeug.utils import secure_filename

from security import ensure_default_admin, verify_login, login_required, role_required
from storage import append_secure_event, verify_log_integrity, LOG_ENC, TIMELINE_CSV
from config import (
    ROOT,
    DATA,
    UPLOAD_DIR,
    MODEL_PATH,
    JSON_LOG,
    ALLOWED_VIDEO_EXT,
    CONFIG,
    CAM_INDEX,
    WARMUP_HITS,
    MOTIONLESS_SPEED,
    MOTIONLESS_FRAMES,
    LOITERING_FRAMES,
    FALL_ASPECT,
    POSE_VIS_MIN,
    MIN_BOX_W,
    MIN_BOX_H,
    MAX_MISSING_FRAMES,
    LOG_BEHAVIORS,
)

from behavior_engine import classify_behavior

CSV_LOG = TIMELINE_CSV

# =========================================================
# OPTIONAL MEDIAPIPE SUPPORT
# =========================================================
# MediaPipe is only used as an optional helper for pose analysis.
# The project still works without MediaPipe because the main behavior
# detection logic is based on YOLO + rule-based movement analysis.
try:
    import mediapipe as mp

    MP_OK = hasattr(mp, "solutions")
    if not MP_OK:
        mp = None
except Exception:
    MP_OK = False
    mp = None



# =========================================================
# FLASK APP
# =========================================================

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)

# Demo/test secret key for the graduation project.
# In a real system, this should be stored as an environment variable.
app.secret_key = "graduation-project-secret-key"

# Creates the default admin user if it does not already exist.
ensure_default_admin()


# =========================================================
# SHARED RUNTIME STATE
# =========================================================
# These variables are shared between the camera thread, inference thread
# and Flask routes. A lock is used when reading/writing shared frame data.

lock = threading.Lock()

latest_frame = None
latest_annot = None
events = deque(maxlen=300)

running = True
camera_enabled = False
analysis_enabled = False

fps_est = 0.0
fps_window = 30
last_fps_time = time.time()
t0 = time.time()

persons = {}
next_person_id = 1
frame_count = 0

yolo = None
capture_thread = None
inference_thread = None


# =========================================================
# OPTIONAL POSE INITIALIZATION
# =========================================================

mp_pose = mp.solutions.pose if MP_OK and mp is not None else None
mp_draw = mp.solutions.drawing_utils if MP_OK and mp is not None else None
POSE = (
    mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    if mp_pose is not None
    else None
)


# =========================================================
# BASIC HELPER FUNCTIONS
# =========================================================

def clamp(v, lo, hi):
    """Keeps a value between minimum and maximum limits."""
    return max(lo, min(hi, v))


def dist(p1, p2):
    """Calculates Euclidean distance between two points."""
    if p1 is None or p2 is None:
        return 0.0
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def bbox_iou(a, b):
    """
    Calculates Intersection over Union between two bounding boxes.
    This helps us understand whether two detected person boxes overlap.
    """
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    denom = areaA + areaB - inter
    return (inter / denom) if denom > 0 else 0.0


def box_center(bbox):
    """Returns the center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def status_frame(text1: str, text2: str = ""):
    """Creates a simple black status frame for the video stream."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(img, text1, (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    if text2:
        cv2.putText(img, text2, (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    return img


def append_timeline_csv(ev: dict):
    """Writes an alert event to timeline.csv."""
    header = ["time_sec", "frame", "person_id", "state", "behavior", "score", "speed", "timestamp"]
    new_file = not TIMELINE_CSV.exists()

    with open(TIMELINE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow({k: ev.get(k, "") for k in header})


def reset_runtime_state():
    """
    Resets person tracking and event timeline for a clean new demo run.
    This prevents old IDs and old alerts from affecting the next analysis.
    """
    global persons, next_person_id, frame_count, t0

    persons = {}
    next_person_id = 1
    frame_count = 0
    t0 = time.time()

    with lock:
        events.clear()


# =========================================================
# MODEL LOADING
# =========================================================

def load_model():
    """
    Loads YOLOv8 model only once.
    This improves startup behavior and avoids reloading the model every frame.
    """
    global yolo

    if yolo is not None:
        return yolo

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

    yolo = YOLO(str(MODEL_PATH))
    return yolo


# =========================================================
# DETECTION FILTERING + SIMPLE ID STABILIZATION
# =========================================================

def filter_duplicate_detections(detections):
    """
    Removes duplicate boxes for the same person.
    If two boxes are very close or strongly overlapping, keep the bigger/confident one.
    """
    kept = []

    for det in sorted(detections, key=lambda d: (d.get("conf", 0), (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1])), reverse=True):
        x1, y1, x2, y2 = det["bbox"]
        c1 = box_center(det["bbox"])
        area1 = max(1, (x2-x1) * (y2-y1))

        is_duplicate = False

        for old in kept:
            ox1, oy1, ox2, oy2 = old["bbox"]
            c2 = box_center(old["bbox"])
            area2 = max(1, (ox2-ox1) * (oy2-oy1))

            center_distance = dist(c1, c2)
            iou = bbox_iou(det["bbox"], old["bbox"])

            # Same person usually has close centers or high overlap.
            if iou > 0.35 or center_distance < 45:
                is_duplicate = True
                break

            # One box inside/near another box: also treat as duplicate.
            smaller = min(area1, area2)
            inter_x1 = max(x1, ox1)
            inter_y1 = max(y1, oy1)
            inter_x2 = min(x2, ox2)
            inter_y2 = min(y2, oy2)
            inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            if smaller > 0 and inter / smaller > 0.55:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(det)

    return kept


def match_detection_to_person(bbox, used_ids):
    """
    Simple project-level ID stabilization.

    This is not face recognition. It only compares:
    - center distance
    - bounding box overlap
    - how recently a person was seen

    Purpose:
    reduce ID jumps and make RUN/FIGHT/FALL decisions more stable.
    """

    center = box_center(bbox)

    best_id = None
    best_score = -1.0

    for pid, p in persons.items():

        if pid in used_ids:
            continue

        frames_missing = frame_count - p.get("last_seen", frame_count)

        if frames_missing > MAX_MISSING_FRAMES:
            continue

        old_box = p.get("bbox", bbox)
        old_center = p.get("center", center)

        iou = bbox_iou(bbox, old_box)
        d = dist(center, old_center)

        max_dist = 170 if frames_missing <= 5 else 110

        if iou < 0.03 and d > max_dist:
            continue

        score = (iou * 4.0) + max(0.0, 1.0 - d / max_dist)

        if score > best_score:
            best_score = score
            best_id = pid

    return best_id


# =========================================================
# OPTIONAL POSE ANALYSIS
# =========================================================

def pose_on_roi(frame, bbox):
    """
    Runs MediaPipe pose analysis on a detected person area.
    This is optional. If disabled or unavailable, the function returns 0.
    """
    if (not MP_OK) or (POSE is None) or (not CONFIG.get("use_pose", False)):
        return 0

    x1, y1, x2, y2 = bbox
    H, W = frame.shape[:2]

    pad = int(0.15 * max(1, x2 - x1))
    rx1 = clamp(x1 - pad, 0, W - 1)
    ry1 = clamp(y1 - pad, 0, H - 1)
    rx2 = clamp(x2 + pad, 0, W - 1)
    ry2 = clamp(y2 + pad, 0, H - 1)

    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return 0

    try:
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pose_result = POSE.process(rgb)

        if pose_result.pose_landmarks:
            mp_draw.draw_landmarks(roi, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            visibilities = [lm.visibility for lm in pose_result.pose_landmarks.landmark]
            visible_points = sum(v > 0.5 for v in visibilities)
            return 1 if visible_points >= POSE_VIS_MIN else 0
    except Exception:
        return 0

    return 0


# =========================================================
# VIDEO SOURCE MANAGEMENT
# =========================================================

def open_camera():
    """Opens the default webcam."""
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened. Close other camera apps or change CAM_INDEX.")
    return cap


def open_file(path: str):
    """Opens a selected video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Video file could not be opened: {path}")
    return cap


def capture_loop():
    """
    Reads frames according to selected mode:
    - camera: live webcam
    - file: uploaded/local video file
    - dashboard: no frame capture
    """
    global latest_frame, running, camera_enabled

    cap = None
    cap_kind = None
    cap_path = None

    while running:
        if not camera_enabled:
            if cap is not None:
                cap.release()
                cap = None
                cap_kind = None
                cap_path = None
            with lock:
                latest_frame = None
            time.sleep(0.05)
            continue

        mode = CONFIG.get("mode", "dashboard")

        if mode == "dashboard":
            if cap is not None:
                cap.release()
                cap = None
                cap_kind = None
                cap_path = None
            with lock:
                latest_frame = None
            time.sleep(0.05)
            continue

        desired_kind = "camera" if mode == "camera" else "file"
        desired_path = None if desired_kind == "camera" else CONFIG.get("source_path", "").strip()

        needs_reopen = False
        if cap is None:
            needs_reopen = True
        elif desired_kind != cap_kind:
            needs_reopen = True
        elif desired_kind == "file" and desired_path != cap_path:
            needs_reopen = True

        if needs_reopen:
            if cap is not None:
                cap.release()

            cap = None
            cap_kind = None
            cap_path = None

            try:
                if desired_kind == "camera":
                    cap = open_camera()
                    cap_kind = "camera"
                else:
                    if not desired_path:
                        with lock:
                            latest_frame = None
                        time.sleep(0.05)
                        continue
                    cap = open_file(desired_path)
                    cap_kind = "file"
                    cap_path = desired_path
            except Exception:
                with lock:
                    latest_frame = None
                time.sleep(0.08)
                continue

        ret, frame = cap.read()

        if not ret or frame is None:
            if cap_kind == "file":
                # Loop uploaded video continuously during demo.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.02)
                continue
            time.sleep(0.02)
            continue

        resize_w = int(CONFIG.get("resize_w", 1280))
        frame = cv2.resize(frame, (resize_w, int(frame.shape[0] * (resize_w / frame.shape[1]))))

        with lock:
            latest_frame = frame

        time.sleep(0.001)

    if cap is not None:
        cap.release()


# =========================================================
# MAIN ANALYSIS LOOP
# =========================================================

def inference_loop():
    """
    Main analysis loop.

    Steps:
    1. Get latest frame.
    2. Detect people using YOLOv8.
    3. Stabilize person IDs.
    4. Calculate speed, closeness and overlap.
    5. Classify behavior as RUN, FIGHT, FALL or UNKNOWN.
    6. Draw results and log meaningful alerts.
    """
    global latest_annot, frame_count, fps_est, last_fps_time, persons, next_person_id, t0

    while running:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if (not camera_enabled) or (CONFIG.get("mode") == "dashboard") or frame is None:
            msg2 = (
                "Click 'Start Live Analyzer' to begin."
                if not camera_enabled
                else "No video source. Select a mode and choose a file."
            )
            with lock:
                latest_annot = status_frame("Camera is OFF" if not camera_enabled else "Waiting for source...", msg2)
            time.sleep(0.05)
            continue

        if not analysis_enabled:
            with lock:
                latest_annot = frame
            time.sleep(0.02)
            continue

        try:
            model = load_model()
        except Exception as e:
            with lock:
                latest_annot = status_frame("Model error", str(e))
            time.sleep(0.5)
            continue

        frame_count += 1
        H, W = frame.shape[:2]

        # Estimate FPS approximately every fps_window frames.
        if frame_count % fps_window == 0:
            now = time.time()
            dt = now - last_fps_time
            if dt > 0:
                fps_est = fps_window / dt
            last_fps_time = now

        conf = float(CONFIG.get("conf", 0.50))
        tracker = str(CONFIG.get("tracker", "bytetrack.yaml"))
        dist_th = float(CONFIG.get("dist_th", 85))
        iou_th = float(CONFIG.get("iou_th", 0.25))
        cooldown = int(CONFIG.get("cooldown", 45))

        try:
            results = model.track(
                frame,
                conf=conf,
                classes=[0],
                persist=True,
                tracker=tracker,
                verbose=False,
            )
        except Exception:
            time.sleep(0.02)
            continue

        r0 = results[0]
        boxes = r0.boxes
        detections = []

        # Collect only person detections with acceptable box sizes.
        if boxes is not None and len(boxes) > 0:
            xyxys = boxes.xyxy
            confs = boxes.conf if getattr(boxes, "conf", None) is not None else None

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, xyxys[i].tolist())

                x1 = clamp(x1, 0, W - 1)
                y1 = clamp(y1, 0, H - 1)
                x2 = clamp(x2, 0, W - 1)
                y2 = clamp(y2, 0, H - 1)

                bw = x2 - x1
                bh = y2 - y1

                if bw < MIN_BOX_W or bh < MIN_BOX_H:
                    continue

                conf_val = float(confs[i].item()) if confs is not None else 1.0

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf_val,
                })

        detections = filter_duplicate_detections(detections)
        matched_ids = set()

        # Assign project-level stable IDs to detections.
        for det in detections:
            bbox = det["bbox"]
            cx, cy = box_center(bbox)

            pid = match_detection_to_person(bbox, matched_ids)

            if pid is None:
                pid = next_person_id
                next_person_id += 1

                persons[pid] = {
                    "center": (cx, cy),
                    "prev": None,
                    "bbox": bbox,
                    "warmup": 0,
                    "last_seen": frame_count,
                    "last_event_frame": -10**9,
                    "fight_hold": 0,
                    "run_hold": 0,
                    "fall_hold": 0,
                    "motionless": 0,
                    "loiter": 0,
                    "pose_ok": 0,
                }

            matched_ids.add(pid)

            p = persons[pid]
            p["last_seen"] = frame_count
            p["warmup"] += 1
            p["prev"] = p.get("center")
            p["center"] = (cx, cy)
            p["bbox"] = bbox
            p["pose_ok"] = pose_on_roi(frame, bbox)

        # Remove people who disappeared for too long.
        for old_id in list(persons.keys()):
            if frame_count - persons[old_id].get("last_seen", frame_count) > MAX_MISSING_FRAMES:
               del persons[old_id]

        active_ids = [
            pid for pid, p in persons.items()
            if p.get("last_seen") == frame_count
        ]

        for pid in active_ids:
            p = persons[pid]

            # Warmup avoids making a decision from the first few unstable frames.
            if p["warmup"] < WARMUP_HITS:
                continue

            speed = dist(p.get("prev"), p.get("center"))
            overlaps = 0
            close_count = 0
            score = 0

            if speed > CONFIG["speed_th"]:
                score += 1

            # Compare current person with other active people.
            for oid in active_ids:
                if oid == pid:
                    continue

                other = persons[oid]

                if dist(p["center"], other["center"]) < dist_th:
                    close_count += 1
                    score += 1

                if bbox_iou(p["bbox"], other["bbox"]) > iou_th:
                    overlaps += 1
                    score += 1

            # FIGHT hold: close/overlap condition must continue for multiple frames.
            if overlaps >= 1 or close_count >= 1:
                p["fight_hold"] += 1
            else:
                p["fight_hold"] = max(0, p["fight_hold"] - 1)

            # RUN hold: fast movement must continue for multiple frames.
            if CONFIG["speed_th"] <= speed <= 220 and overlaps == 0 and close_count == 0:
                p["run_hold"] += 1
            else:
                p["run_hold"] = max(0, p["run_hold"] - 1)

            # FALL hold: a clearly horizontal bounding box must continue for multiple frames.
            x1, y1, x2, y2 = p["bbox"]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            aspect = bw / float(bh)

            # FALL should represent a falling movement, not a person simply standing still.
            # Therefore, the box must be horizontal and movement must be small-to-medium.
            if aspect >= FALL_ASPECT and 12 <= speed <= 180:
                p["fall_hold"] += 1
            else:
                p["fall_hold"] = max(0, p["fall_hold"] - 1)

            # Motionless/loitering values are calculated but not logged as alerts.
            if speed < MOTIONLESS_SPEED:
                p["motionless"] += 1
            else:
                p["motionless"] = 0
                p["loiter"] = 0

            if p["motionless"] > MOTIONLESS_FRAMES:
                p["loiter"] += 1

            behavior = classify_behavior(
                speed=speed,
                overlaps=overlaps,
                close_count=close_count,
                bbox=p["bbox"],
                fight_hold=p["fight_hold"],
                run_hold=p["run_hold"],
                fall_hold=p["fall_hold"],
                pose_ok=p["pose_ok"],
            )

            if behavior in LOG_BEHAVIORS:
                state = "ALERT"
                color = (0, 0, 255)
            else:
                state = "NORMAL"
                color = (0, 255, 0)

            should_log = (
                state == "ALERT"
                and behavior in LOG_BEHAVIORS
                and 5 <= speed <= 220
                and frame_count - p["last_event_frame"] > cooldown
            )

            if should_log:
                ev = {
                    "time_sec": round(time.time() - t0, 2),
                    "frame": int(frame_count),
                    "person_id": int(pid),
                    "state": state,
                    "behavior": behavior,
                    "score": int(score),
                    "speed": round(float(speed), 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                with lock:
                    events.append(ev)

                append_timeline_csv(ev)
                append_secure_event(ev)
                p["last_event_frame"] = frame_count

            # Draw result on the frame.
            thickness = 2 if state == "ALERT" else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if state == "ALERT":
                label = f"ID:{pid} {behavior}"
            else:
                label = f"ID:{pid} NORMAL"

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

        cv2.putText(
            frame,
            f"mode={CONFIG.get('mode')} fps~{fps_est:.1f} frame={frame_count}",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        with lock:
            latest_annot = frame

        time.sleep(0.005)


# =========================================================
# MJPEG STREAM
# =========================================================

def mjpeg_generator():
    """Streams the latest annotated frame to the browser as MJPEG."""
    while True:
        with lock:
            frame = None if latest_annot is None else latest_annot.copy()

        if frame is None:
            frame = status_frame("Initializing...", "Please wait.")

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            )

        time.sleep(0.03)


# =========================================================
# THREAD START
# =========================================================

def start_threads():
    """Starts capture and inference threads once."""
    global capture_thread, inference_thread

    if capture_thread is None:
        capture_thread = threading.Thread(target=capture_loop, daemon=True)
        capture_thread.start()

    if inference_thread is None:
        inference_thread = threading.Thread(target=inference_loop, daemon=True)
        inference_thread.start()


# =========================================================
# AUTH ROUTES
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = verify_login(request.form.get("username", ""), request.form.get("password", ""))
        if user:
            session["user"] = user
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid username or password.")

    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# =========================================================
# MAIN UI ROUTES
# =========================================================

@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


# =========================================================
# VIDEO UPLOAD + VIDEO LIST ROUTES
# =========================================================

@app.route("/api/videos")
@login_required
def api_videos():
    vids = []
    for p in UPLOAD_DIR.glob("*"):
        if p.suffix.lower() in ALLOWED_VIDEO_EXT:
            vids.append(p.name)
    vids.sort()
    return jsonify({"videos": vids})


@app.route("/api/upload_video", methods=["POST"])
@login_required
def api_upload_video():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file field"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "error": "Empty filename"}), 400

    name = secure_filename(f.filename)
    ext = Path(name).suffix.lower()

    if ext not in ALLOWED_VIDEO_EXT:
        return jsonify({"ok": False, "error": f"Unsupported type: {ext}"}), 400

    save_path = UPLOAD_DIR / name
    f.save(str(save_path))

    # Uploaded video becomes the active analysis source.
    CONFIG["mode"] = "file"
    CONFIG["source_path"] = str(save_path)

    return jsonify({"ok": True, "filename": name})


# =========================================================
# STATUS + CONTROL ROUTES
# =========================================================

@app.route("/api/status")
@login_required
def api_status():
    global analysis_enabled

    if not camera_enabled:
        analysis_enabled = False

    with lock:
        evs = list(events)

    return jsonify({
        "camera_enabled": camera_enabled,
        "analysis_enabled": analysis_enabled,
        "fps": round(float(fps_est), 2),
        "frame": int(frame_count),
        "events": evs[-200:],
        "pose_enabled": bool(MP_OK),
        "config": CONFIG,
    })


@app.route("/api/start_live", methods=["POST"])
@login_required
def start_live():
    """
    Starts analysis.
    If no source is selected, camera mode is used by default.
    If a video file was already selected, file mode is preserved.
    """
    global camera_enabled, analysis_enabled, t0

    reset_runtime_state()

    if CONFIG.get("mode") == "dashboard":
        CONFIG["mode"] = "camera"

    camera_enabled = True
    analysis_enabled = True
    t0 = time.time()

    return jsonify({
        "ok": True,
        "camera_enabled": camera_enabled,
        "analysis_enabled": analysis_enabled,
    })


@app.route("/api/stop_live", methods=["POST"])
@login_required
def stop_live():
    global camera_enabled, analysis_enabled

    analysis_enabled = False
    camera_enabled = False

    return jsonify({
        "ok": True,
        "camera_enabled": camera_enabled,
        "analysis_enabled": analysis_enabled,
    })


@app.route("/api/toggle_camera", methods=["POST"])
@role_required("admin")
def toggle_camera():
    global camera_enabled

    camera_enabled = not camera_enabled
    return jsonify({"ok": True, "camera_enabled": camera_enabled})


@app.route("/api/clear_timeline", methods=["POST"])
@role_required("admin")
def clear_timeline():
    with lock:
        events.clear()
    return jsonify({"ok": True})


# =========================================================
# CONFIG + MODE ROUTES
# =========================================================

@app.route("/api/config", methods=["POST"])
@login_required
def api_config():
    """Updates configurable threshold values from the dashboard."""
    global CONFIG

    data = request.get_json(force=True, silent=True) or {}

    for key in data.keys():
        if key not in CONFIG:
            continue

        value = data[key]

        if key in {"conf", "iou_th"}:
            CONFIG[key] = float(str(value).replace(",", "."))
        elif key in {"resize_w", "speed_th", "dist_th", "cooldown", "fight_hold", "run_hold", "fall_hold"}:
            CONFIG[key] = int(float(str(value).replace(",", ".")))
        elif key == "use_pose":
            CONFIG[key] = str(value).lower() in {"true", "1", "on", "yes"}
        else:
            CONFIG[key] = value

    return jsonify({"ok": True, "config": CONFIG})


@app.route("/api/mode", methods=["POST"])
@login_required
def api_mode():
    """Changes active source mode: dashboard, camera or file."""
    global CONFIG

    data = request.get_json(force=True, silent=True) or {}

    if "mode" in data:
        CONFIG["mode"] = str(data["mode"])

    if "source_path" in data:
        CONFIG["source_path"] = str(data["source_path"])

    return jsonify({"ok": True, "config": CONFIG})


# =========================================================
# LOG API ROUTES
# =========================================================

@app.route("/api/logs/csv")
@login_required
def api_logs_csv():
    """Reads timeline CSV records for the dashboard log table."""
    pid = request.args.get("pid", "").strip()
    beh = request.args.get("beh", "").strip().upper()

    if not CSV_LOG.exists() or CSV_LOG.stat().st_size == 0:
        return jsonify({"columns": [], "rows": []})

    with open(CSV_LOG, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return jsonify({"columns": [], "rows": []})

    cols = rows[0]
    data = rows[1:]

    pid_i = cols.index("person_id") if "person_id" in cols else None
    beh_i = cols.index("behavior") if "behavior" in cols else None

    def keep(row):
        if pid and pid_i is not None and str(row[pid_i]) != pid:
            return False
        if beh and beh_i is not None and str(row[beh_i]).upper() != beh:
            return False
        return True

    filtered = [row for row in data if keep(row)]
    return jsonify({"columns": cols, "rows": filtered[-80:]})


@app.route("/api/logs/json")
@login_required
def api_logs_json():
    """Reads old JSON logs if they exist. This is kept for compatibility."""
    if not JSON_LOG.exists() or JSON_LOG.stat().st_size == 0:
        return jsonify([])

    try:
        data = json.loads(JSON_LOG.read_text(encoding="utf-8"))
        return jsonify(data[-80:] if isinstance(data, list) else data)
    except Exception:
        return jsonify([])


# =========================================================
# DOWNLOADS + INTEGRITY
# =========================================================

@app.route("/download/timeline.csv")
@login_required
def download_timeline():
    if not TIMELINE_CSV.exists():
        return "No timeline yet", 404
    return send_file(TIMELINE_CSV, as_attachment=True, download_name="timeline.csv")


@app.route("/download/logs.enc")
@role_required("admin")
def download_logs_enc():
    if not LOG_ENC.exists():
        return "No encrypted logs yet", 404
    return send_file(LOG_ENC, as_attachment=True, download_name="logs.enc")


@app.route("/verify")
@role_required("admin")
def verify():
    return jsonify(verify_log_integrity())


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    start_threads()
    webbrowser.open("http://127.0.0.1:5000/login")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
