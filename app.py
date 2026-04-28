import time, threading, csv, json, webbrowser
from collections import deque
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, Response, jsonify, render_template, request, redirect, url_for, session, send_file
from ultralytics import YOLO

from security import ensure_default_admin, verify_login, login_required, role_required
from storage import append_secure_event, verify_log_integrity, LOG_ENC, TIMELINE_CSV

# Upload helper
from werkzeug.utils import secure_filename

# Optional MediaPipe (skeleton)
try:
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        MP_OK = True
    else:
        MP_OK = False
        mp = None
except Exception:
    MP_OK = False
    mp = None

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

UPLOAD_DIR = DATA / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_PATH = ROOT / "yolov8n.pt"

# Your existing logs (from your project tree)
CSV_LOG = ROOT / "suspicious_log.csv"
JSON_LOG = ROOT / "suspicious_log.json"

ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ---- Default Config (Streamlit sidebar equivalent) ----
CONFIG = {
    "mode": "dashboard",      # dashboard / camera / file
    "source_path": "",        # used when mode=file
    "conf": 0.55,             # ↑ default more strict (less false positives)
    "resize_w": 1280,
    "tracker": "bytetrack.yaml",
    "use_pose": True,
    "speed_th": 25,
    "dist_th": 60,            # ↓ default tighter to reduce “everyone is near”
    "iou_th": 0.45,           # ↑ default stricter overlap
    "cooldown": 30,
    "fight_hold": 10          # NEW: fight must persist N frames
}

# ---- Other constants (you can tune) ----
CAM_INDEX = 0
WARMUP_HITS = 5
CONFIRM_SUSP = 3
CONFIRM_ALERT = 2
MOTIONLESS_SPEED = 6
MOTIONLESS_FRAMES = 60
LOITERING_FRAMES = 180
FALL_ASPECT = 1.20
POSE_VIS_MIN = 6

app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
app.secret_key = "dev-secret-change-me"

ensure_default_admin()

# Shared state
lock = threading.Lock()
latest_frame = None
latest_annot = None
events = deque(maxlen=300)

running = True

# ✅ Start with camera OFF until user clicks Start
camera_enabled = False
analysis_enabled = False

# Stable fps estimate
fps_est = 0.0
fps_window = 30
last_fps_time = time.time()

# timeline time reference (resets on Start)
t0 = time.time()

# tracking state
persons = {}
frame_count = 0

# Load model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")
yolo = YOLO(str(MODEL_PATH))

# Pose init (optional)
mp_pose = mp.solutions.pose if MP_OK and mp is not None else None
mp_draw = mp.solutions.drawing_utils if MP_OK and mp is not None else None
POSE = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) if mp_pose is not None else None


# ---------------- Helpers ----------------
def clamp(v, lo, hi): return max(lo, min(hi, v))

def dist(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

def bbox_iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, (a[2]-a[0])) * max(0, (a[3]-a[1]))
    areaB = max(0, (b[2]-b[0])) * max(0, (b[3]-b[1]))
    denom = (areaA + areaB - inter)
    return (inter / denom) if denom > 0 else 0.0

def _status_frame(text1: str, text2: str = ""):
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(img, text1, (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
    if text2:
        cv2.putText(img, text2, (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,180), 2)
    return img

def append_timeline_csv(ev: dict):
    header = ["time_sec","frame","person_id","state","behavior","score","speed","timestamp"]
    new_file = not TIMELINE_CSV.exists()
    with open(TIMELINE_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow({k: ev.get(k, "") for k in header})


# ---------------- Capture Loop (Camera or File) ----------------
def _open_camera():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened. Close other camera apps or change CAM_INDEX.")
    return cap

def _open_file(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Video file could not be opened: {path}")
    return cap

def capture_loop():
    """
    Reads frames based on CONFIG['mode'] when camera_enabled=True:
      - mode='camera' -> webcam
      - mode='file'   -> CONFIG['source_path']
      - mode='dashboard' -> no capture
    """
    global latest_frame, running, camera_enabled

    cap = None
    cap_kind = None   # "camera" or "file"
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
                    cap = _open_camera()
                    cap_kind = "camera"
                else:
                    if not desired_path:
                        with lock:
                            latest_frame = None
                        time.sleep(0.05)
                        continue
                    cap = _open_file(desired_path)
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
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.02)
                continue
            time.sleep(0.02)
            continue

        resize_w = int(CONFIG.get("resize_w", 1280))
        frame = cv2.resize(frame, (resize_w, int(frame.shape[0]*(resize_w/frame.shape[1]))))

        with lock:
            latest_frame = frame
        time.sleep(0.001)

    if cap is not None:
        cap.release()


# ---------------- Inference Loop ----------------
def _pose_on_roi(frame, bbox):
    if (not MP_OK) or (POSE is None) or (not CONFIG.get("use_pose", True)):
        return 0
    x1,y1,x2,y2 = bbox
    H,W = frame.shape[:2]
    pad = int(0.15 * max(1, (x2-x1)))
    rx1 = clamp(x1-pad, 0, W-1); ry1 = clamp(y1-pad, 0, H-1)
    rx2 = clamp(x2+pad, 0, W-1); ry2 = clamp(y2+pad, 0, H-1)
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return 0
    try:
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pr = POSE.process(rgb)
        if pr.pose_landmarks:
            mp_draw.draw_landmarks(roi, pr.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            vis = [lm.visibility for lm in pr.pose_landmarks.landmark]
            ok = 1 if sum(v > 0.5 for v in vis) >= POSE_VIS_MIN else 0
            return ok
    except Exception:
        return 0
    return 0

def _behavior(speed, overlaps, pose_ok, bbox, motionless, loiter, fight_hold):
    x1,y1,x2,y2 = bbox
    w = max(1, x2-x1); h = max(1, y2-y1)
    aspect = w / float(h)

    if pose_ok and speed < MOTIONLESS_SPEED and aspect >= FALL_ASPECT:
        return "FALL"

    # ✅ Fight must persist (reduces false positives in crowded videos)
    if fight_hold >= int(CONFIG.get("fight_hold", 10)) and speed > CONFIG.get("speed_th", 25):
        return "FIGHT"

    if speed > CONFIG.get("speed_th", 25) and overlaps == 0:
        return "RUN"
    if motionless >= MOTIONLESS_FRAMES:
        return "MOTIONLESS"
    if loiter >= (LOITERING_FRAMES - MOTIONLESS_FRAMES):
        return "LOITERING"
    return "UNKNOWN"

def inference_loop():
    global latest_annot, frame_count, fps_est, last_fps_time, persons, t0

    while running:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if (not camera_enabled) or (CONFIG.get("mode") == "dashboard") or frame is None:
            msg2 = "Click 'Start Live Analyzer' to begin." if not camera_enabled else "No video source. Select mode and set a file."
            with lock:
                latest_annot = _status_frame("Camera is OFF" if not camera_enabled else "Waiting for source...", msg2)
            time.sleep(0.05)
            continue

        if not analysis_enabled:
            with lock:
                latest_annot = frame
            time.sleep(0.02)
            continue

        frame_count += 1
        H, W = frame.shape[:2]

        if frame_count % fps_window == 0:
            now = time.time()
            dt = now - last_fps_time
            if dt > 0:
                fps_est = fps_window / dt
            last_fps_time = now

        conf = float(CONFIG.get("conf", 0.55))
        tracker = str(CONFIG.get("tracker", "bytetrack.yaml"))
        dist_th = float(CONFIG.get("dist_th", 60))
        iou_th = float(CONFIG.get("iou_th", 0.45))
        cooldown = int(CONFIG.get("cooldown", 30))

        try:
            results = yolo.track(frame, conf=conf, classes=[0], persist=True, tracker=tracker, verbose=False)
        except Exception:
            time.sleep(0.02)
            continue

        r0 = results[0]
        boxes = r0.boxes

        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            xyxys = boxes.xyxy
            ids = boxes.id
            for i in range(len(boxes)):
                x1,y1,x2,y2 = map(int, xyxys[i].tolist())
                x1 = clamp(x1,0,W-1); y1 = clamp(y1,0,H-1)
                x2 = clamp(x2,0,W-1); y2 = clamp(y2,0,H-1)
                tid = int(ids[i].item())
                cx,cy = (x1+x2)//2, (y1+y2)//2

                if tid not in persons:
                    persons[tid] = {
                        "center": (cx,cy),
                        "prev": None,
                        "bbox": (x1,y1,x2,y2),
                        "susp": 0,
                        "alert": 0,
                        "warmup": 0,
                        "last_event_frame": -10**9,
                        "motionless": 0,
                        "loiter": 0,
                        "pose_ok": 0,
                        "fight_hold": 0
                    }

                p = persons[tid]
                p["warmup"] += 1
                p["prev"] = p["center"]
                p["center"] = (cx,cy)
                p["bbox"] = (x1,y1,x2,y2)

                p["pose_ok"] = _pose_on_roi(frame, p["bbox"])

        active_ids = list(persons.keys())

        for pid in active_ids:
            p = persons[pid]
            if p["warmup"] < WARMUP_HITS:
                continue

            speed = 0.0
            if p["prev"] is not None:
                speed = dist(p["prev"], p["center"])

            score = 0
            if speed > CONFIG.get("speed_th", 25):
                score += 1

            overlaps = 0
            for oid in active_ids:
                if oid == pid:
                    continue
                o = persons[oid]
                if dist(p["center"], o["center"]) < dist_th:
                    score += 1
                if bbox_iou(p["bbox"], o["bbox"]) > iou_th:
                    overlaps += 1
                    score += 1

            # hold logic
            if overlaps >= 1:
                p["fight_hold"] += 1
            else:
                p["fight_hold"] = max(0, p["fight_hold"] - 1)

            if score >= 3:
                state = "ALERT"; color = (0,0,255); p["alert"] += 1
            elif score == 2:
                state = "SUSPICIOUS"; color = (0,165,255); p["susp"] += 1
            else:
                state = "NORMAL"; color = (0,255,0)

            if speed < MOTIONLESS_SPEED:
                p["motionless"] += 1
            else:
                p["motionless"] = 0
                p["loiter"] = 0

            if p["motionless"] > MOTIONLESS_FRAMES:
                p["loiter"] += 1

            behavior = _behavior(speed, overlaps, p["pose_ok"], p["bbox"], p["motionless"], p["loiter"], p["fight_hold"])

            confirmed = (p["susp"] >= CONFIRM_SUSP or p["alert"] >= CONFIRM_ALERT or behavior == "FALL")
            if confirmed and (frame_count - p["last_event_frame"] > cooldown):
                tsec = round(time.time() - t0, 2)
                ev = {
                    "time_sec": tsec,
                    "frame": int(frame_count),
                    "person_id": int(pid),
                    "state": state,
                    "behavior": behavior,
                    "score": int(score),
                    "speed": round(float(speed), 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with lock:
                    events.append(ev)
                append_timeline_csv(ev)
                append_secure_event(ev)
                p["last_event_frame"] = frame_count

            x1,y1,x2,y2 = p["bbox"]
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID:{pid} {state} {behavior}",
                        (x1, max(20,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame,
                    f"mode={CONFIG.get('mode')} fps~{fps_est:.1f} frame={frame_count}",
                    (18,32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        with lock:
            latest_annot = frame
        time.sleep(0.005)


# ---------------- MJPEG Stream ----------------
def mjpeg_generator():
    while True:
        with lock:
            frame = None if latest_annot is None else latest_annot.copy()
        if frame is None:
            frame = _status_frame("Initializing...", "Please wait.")
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
        time.sleep(0.03)


def start_threads():
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()


# ---------------- Auth ----------------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        user = verify_login(request.form.get("username",""), request.form.get("password",""))
        if user:
            session["user"] = user
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html", error=None)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ---------------- UI ----------------
@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/video_feed")
@login_required
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------- Upload + list videos ----------------
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

    # switch to file mode but DO NOT auto-open webcam
    CONFIG["mode"] = "file"
    CONFIG["source_path"] = str(save_path)

    return jsonify({"ok": True, "filename": name})


# ---------------- API: Status + Controls ----------------
@app.route("/api/status")
@login_required
def api_status():
    global analysis_enabled
    if not camera_enabled:
        analysis_enabled = False  # keep UI consistent

    with lock:
        evs = list(events)
    return jsonify({
        "camera_enabled": camera_enabled,
        "analysis_enabled": analysis_enabled,
        "fps": round(float(fps_est), 2),
        "frame": int(frame_count),
        "events": evs[-200:],
        "pose_enabled": bool(MP_OK),
        "config": CONFIG
    })

@app.route("/api/start_live", methods=["POST"])
@login_required
def start_live():
    global camera_enabled, analysis_enabled, t0
    camera_enabled = True
    analysis_enabled = True
    t0 = time.time()
    return jsonify({"ok": True, "camera_enabled": camera_enabled, "analysis_enabled": analysis_enabled})

@app.route("/api/stop_live", methods=["POST"])
@login_required
def stop_live():
    global camera_enabled, analysis_enabled
    analysis_enabled = False
    camera_enabled = False
    return jsonify({"ok": True, "camera_enabled": camera_enabled, "analysis_enabled": analysis_enabled})

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


# ---------------- API: Config + Mode ----------------
@app.route("/api/config", methods=["POST"])
@login_required
def api_config():
    global CONFIG
    data = request.get_json(force=True, silent=True) or {}
    CONFIG.update({k: data[k] for k in data.keys() if k in CONFIG})
    return jsonify({"ok": True, "config": CONFIG})

@app.route("/api/mode", methods=["POST"])
@login_required
def api_mode():
    global CONFIG
    data = request.get_json(force=True, silent=True) or {}
    if "mode" in data:
        CONFIG["mode"] = str(data["mode"])
    if "source_path" in data:
        CONFIG["source_path"] = str(data["source_path"])
    return jsonify({"ok": True, "config": CONFIG})


# ---------------- API: Logs ----------------
@app.route("/api/logs/csv")
@login_required
def api_logs_csv():
    pid = request.args.get("pid","").strip()
    beh = request.args.get("beh","").strip().upper()

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
    beh_i = cols.index("behavior_type") if "behavior_type" in cols else None

    def keep(r):
        if pid and pid_i is not None and str(r[pid_i]) != pid:
            return False
        if beh and beh_i is not None and str(r[beh_i]).upper() != beh:
            return False
        return True

    filtered = [r for r in data if keep(r)]
    return jsonify({"columns": cols, "rows": filtered[-80:]})

@app.route("/api/logs/json")
@login_required
def api_logs_json():
    if not JSON_LOG.exists() or JSON_LOG.stat().st_size == 0:
        return jsonify([])
    try:
        data = json.loads(JSON_LOG.read_text(encoding="utf-8"))
        return jsonify(data[-80:] if isinstance(data, list) else data)
    except Exception:
        return jsonify([])


# ---------------- Downloads + Integrity ----------------
@app.route("/download/timeline.csv")
@login_required
def download_timeline():
    if not TIMELINE_CSV.exists():
        return ("No timeline yet", 404)
    return send_file(TIMELINE_CSV, as_attachment=True, download_name="timeline.csv")

@app.route("/download/logs.enc")
@role_required("admin")
def download_logs_enc():
    if not LOG_ENC.exists():
        return ("No encrypted logs yet", 404)
    return send_file(LOG_ENC, as_attachment=True, download_name="logs.enc")

@app.route("/verify")
@role_required("admin")
def verify():
    return jsonify(verify_log_integrity())


if __name__ == "__main__":
    start_threads()
    webbrowser.open("http://127.0.0.1:5000/login")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
