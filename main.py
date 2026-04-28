# main.py — FINAL (stable, presentation-ready)
# - No hardcoded paths (auto uses ./test.mp4 or CLI arg)
# - Works with: python main.py           (uses test.mp4)
#              python main.py 0         (webcam)
#              python main.py path/to/video.mp4
# - Skeleton is drawn on FULL FRAME (visible)
# - Fall detection improved (angle OR bbox aspect + votes)
# - Logs are NOT reset each run (CSV header only if missing)
# - Clear debug overlay + robust error messages

import os
import sys
import time
import json
import csv
import math
from pathlib import Path

import cv2
import mediapipe as mp
from ultralytics import YOLO

# ============================================================
# PATHS / INPUT SOURCE
# ============================================================
ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "test.mp4"
MODEL_PATH = ROOT / "yolov8n.pt"

def resolve_video_source():
    arg = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_VIDEO)

    # Webcam mode
    if str(arg).strip() == "0":
        return 0

    # If relative -> relative to project folder
    src = arg
    if not os.path.isabs(src):
        src = str((ROOT / src).resolve())

    if not os.path.exists(src):
        print("❌ Video dosyası bulunamadı:", src)
        print("✅ Çözüm:")
        print("   - test.mp4 dosyasını main.py ile aynı klasöre koy")
        print("   - veya: python main.py <tam_yol_video>")
        raise SystemExit
    return src

VIDEO_SOURCE = resolve_video_source()

# ============================================================
# VIDEO CAPTURE
# ============================================================
def open_capture(source):
    # Webcam: DirectShow helps on Windows
    if source == 0:
        cap_ = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap_ = cv2.VideoCapture(source)  # file => don't force CAP_DSHOW
    return cap_

cap = open_capture(VIDEO_SOURCE)
if not cap.isOpened():
    print("❌ Video/Kamera açılamadı:", VIDEO_SOURCE)
    print("✅ Kontrol:")
    print("   - Webcam ise: python main.py 0")
    print("   - Dosya ise: dosya yolu doğru mu? codec sorunu olabilir mi?")
    raise SystemExit

FPS = cap.get(cv2.CAP_PROP_FPS)
if not FPS or FPS == 0:
    FPS = 30.0

# ============================================================
# DISPLAY
# ============================================================
SCREEN_W, SCREEN_H = 1280, 720
WINDOW = "Hybrid Real-Time Behavior Detection (FINAL)"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, SCREEN_W, SCREEN_H)

# ============================================================
# MODELS
# ============================================================
if not MODEL_PATH.exists():
    print("❌ YOLO model bulunamadı:", MODEL_PATH)
    print("✅ Çözüm: yolov8n.pt dosyasını proje klasörüne koy.")
    raise SystemExit

yolo = YOLO(str(MODEL_PATH))

# Tracker selection (ultralytics built-ins)
TRACKER_YAML = "bytetrack.yaml"  # try "botsort.yaml" if IDs unstable

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ============================================================
# LOG FILES (DO NOT RESET)
# ============================================================
CSV_LOG = ROOT / "suspicious_log.csv"
JSON_LOG = ROOT / "suspicious_log.json"

CSV_HEADER = [
    "person_id", "classification", "behavior_type",
    "start_frame", "end_frame", "video_time_sec",
    "suspicious_frames", "alert_frames", "timestamp"
]

def init_logs():
    if not CSV_LOG.exists():
        with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)
    if not JSON_LOG.exists():
        JSON_LOG.write_text("[]", encoding="utf-8")

def append_record(record: dict):
    # CSV append
    with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([record.get(h, "") for h in CSV_HEADER])

    # JSON append
    try:
        recs = json.loads(JSON_LOG.read_text(encoding="utf-8"))
        if not isinstance(recs, list):
            recs = []
    except Exception:
        recs = []
    recs.append(record)
    JSON_LOG.write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")

init_logs()

# ============================================================
# PARAMETERS (tunable)
# ============================================================
YOLO_CONF = 0.40
MAX_MISSED = 18

POSE_INTERVAL = 1          # 1 = skeleton always (heavier), 2 = faster
POSE_VIS_THR = 0.30        # lower => more likely to draw

SPEED_THRESHOLD = 25       # px/frame
DIST_THRESHOLD = 80        # px
IOU_THRESHOLD = 0.30       # overlap threshold

# FALL detection: improved
FALL_ANGLE_DEG = 50        # torso tilt
FALL_ASPECT_THR = 1.30     # bbox width/height suggests lying
FALL_VOTES_NEED = 3        # persistence

# Confirmation thresholds (persistence to reduce false positives)
SUSPICIOUS_CONFIRM = 3
ALERT_CONFIRM = 2

# ============================================================
# HELPERS
# ============================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def dist(p1, p2):
    return math.dist(p1, p2)

def compute_speed(prev, curr):
    if prev is None or curr is None:
        return 0.0
    return dist(prev, curr)

def bbox_iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, (a[2] - a[0])) * max(0, (a[3] - a[1]))
    areaB = max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))
    denom = (areaA + areaB - inter)
    if denom <= 0:
        return 0.0
    return inter / denom

def body_angle_deg(landmarks):
    """
    Torso angle from shoulder-mid to hip-mid.
    Returns angle in degrees (0..90 approx), or None if insufficient visibility.
    """
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    if min(ls.visibility, rs.visibility, lh.visibility, rh.visibility) < POSE_VIS_THR:
        return None

    sx = (ls.x + rs.x) / 2.0
    sy = (ls.y + rs.y) / 2.0
    hx = (lh.x + rh.x) / 2.0
    hy = (lh.y + rh.y) / 2.0

    dx = (sx - hx)
    dy = (sy - hy)
    angle = abs(math.degrees(math.atan2(dx, dy)))  # 0..90
    return angle

def draw_pose_on_frame(frame, landmarks, rx1, ry1, rx2, ry2):
    """
    Draw landmarks computed on ROI onto the full frame by mapping ROI coords -> frame coords.
    """
    roi_w = max(1, rx2 - rx1)
    roi_h = max(1, ry2 - ry1)

    pts = {}
    for i, lm in enumerate(landmarks):
        if lm.visibility < POSE_VIS_THR:
            continue
        x = int(rx1 + lm.x * roi_w)
        y = int(ry1 + lm.y * roi_h)
        pts[i] = (x, y)

    # draw points
    for p in pts.values():
        cv2.circle(frame, p, 3, (255, 255, 0), -1)

    # draw connections
    for a, b in mp_pose.POSE_CONNECTIONS:
        ia = a.value
        ib = b.value
        if ia in pts and ib in pts:
            cv2.line(frame, pts[ia], pts[ib], (255, 255, 0), 2)

# ============================================================
# MAIN LOOP
# ============================================================
persons = {}  # track_id -> state dict
frame_count = 0

print("▶ Sistem Başladı | Tracker:", TRACKER_YAML, "| ESC / Q çıkış")
print("▶ Kaynak:", VIDEO_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (SCREEN_W, SCREEN_H))
    frame_count += 1

    # YOLO TRACK
    results = yolo.track(
        frame,
        conf=YOLO_CONF,
        classes=[0],           # person
        persist=True,
        tracker=TRACKER_YAML,
        verbose=False
    )

    r0 = results[0]
    boxes = r0.boxes
    seen_ids = set()

    if boxes is not None and len(boxes) > 0:
        if boxes.id is None:
            print("⚠️ Tracker ID üretemedi. TRACKER_YAML değiştir: botsort.yaml deneyebilirsin.")
            break

        xyxys = boxes.xyxy
        ids = boxes.id

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, xyxys[i].tolist())
            x1 = clamp(x1, 0, SCREEN_W - 1)
            y1 = clamp(y1, 0, SCREEN_H - 1)
            x2 = clamp(x2, 0, SCREEN_W - 1)
            y2 = clamp(y2, 0, SCREEN_H - 1)

            tid = int(ids[i].item())
            seen_ids.add(tid)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if tid not in persons:
                persons[tid] = {
                    "center": (cx, cy),
                    "prev_center": None,
                    "bbox": (x1, y1, x2, y2),
                    "missed": 0,

                    "suspicious": 0,
                    "alert": 0,
                    "normal": 0,
                    "start_frame": None,
                    "logged": False,

                    "pose_ok": 0,
                    "fall_votes": 0,
                }

            p = persons[tid]
            p["prev_center"] = p["center"]
            p["center"] = (cx, cy)
            p["bbox"] = (x1, y1, x2, y2)
            p["missed"] = 0

            # POSE (ROI) + DRAW ON FULL FRAME (always visible)
            p["pose_ok"] = 0
            angle = None

            if frame_count % POSE_INTERVAL == 0:
                pad = int(0.15 * max(1, (x2 - x1)))
                rx1 = clamp(x1 - pad, 0, SCREEN_W - 1)
                ry1 = clamp(y1 - pad, 0, SCREEN_H - 1)
                rx2 = clamp(x2 + pad, 0, SCREEN_W - 1)
                ry2 = clamp(y2 + pad, 0, SCREEN_H - 1)

                roi = frame[ry1:ry2, rx1:rx2]
                if roi.size > 0:
                    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    pr = pose.process(rgb)

                    if pr.pose_landmarks:
                        lms = pr.pose_landmarks.landmark
                        p["pose_ok"] = 1

                        # draw skeleton on full frame (mapped)
                        draw_pose_on_frame(frame, lms, rx1, ry1, rx2, ry2)

                        # torso angle
                        angle = body_angle_deg(lms)

            # FALL votes (angle OR bbox aspect)
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            aspect = bw / bh

            fall_signal = False
            if angle is not None and angle >= FALL_ANGLE_DEG:
                fall_signal = True
            if aspect >= FALL_ASPECT_THR:
                fall_signal = True

            if fall_signal:
                p["fall_votes"] += 1
            else:
                p["fall_votes"] = max(0, p["fall_votes"] - 1)

    # MISSED / CLEANUP
    to_delete = []
    for pid, p in persons.items():
        if pid not in seen_ids:
            p["missed"] += 1
            if p["missed"] > MAX_MISSED:
                to_delete.append(pid)
    for pid in to_delete:
        del persons[pid]

    # DECISION + DRAW + LOG
    active_ids = list(persons.keys())
    alerts_now = 0

    for pid in active_ids:
        p = persons[pid]
        score = 0

        speed = compute_speed(p["prev_center"], p["center"])
        if speed > SPEED_THRESHOLD:
            score += 1

        overlaps = 0
        for oid in active_ids:
            if oid == pid:
                continue
            o = persons[oid]
            if dist(p["center"], o["center"]) < DIST_THRESHOLD:
                score += 1
            if bbox_iou(p["bbox"], o["bbox"]) > IOU_THRESHOLD:
                overlaps += 1
                score += 1

        if p["pose_ok"]:
            score += 1

        # Classification state
        if score >= 3:
            state = "ALERT"
            color = (0, 0, 255)
            p["alert"] += 1
            p["normal"] = 0
            alerts_now += 1
        elif score == 2:
            state = "SUSPICIOUS"
            color = (0, 165, 255)
            p["suspicious"] += 1
            p["normal"] = 0
        else:
            state = "NORMAL"
            color = (0, 255, 0)
            p["normal"] += 1

        if p["start_frame"] is None and state != "NORMAL":
            p["start_frame"] = frame_count

        # Behavior label
        behavior = "UNKNOWN"
        if p["fall_votes"] >= FALL_VOTES_NEED:
            behavior = "FALL"
        elif overlaps >= 1 and speed > SPEED_THRESHOLD:
            behavior = "FIGHT"
        elif speed > SPEED_THRESHOLD and overlaps == 0:
            behavior = "RUN"

        # LOG once when confirmed
        confirm = (behavior == "FALL") or (p["suspicious"] >= SUSPICIOUS_CONFIRM) or (p["alert"] >= ALERT_CONFIRM)
        if (not p["logged"]) and confirm:
            video_time = frame_count / FPS
            record = {
                "person_id": pid,
                "classification": "Fall Detected" if behavior == "FALL" else "Confirmed Suspicious",
                "behavior_type": behavior,
                "start_frame": p["start_frame"] if p["start_frame"] is not None else frame_count,
                "end_frame": frame_count,
                "video_time_sec": round(float(video_time), 2),
                "suspicious_frames": int(p["suspicious"]),
                "alert_frames": int(p["alert"]),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            append_record(record)
            p["logged"] = True

        # DRAW bbox + text
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID:{pid} {state} {behavior}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            color,
            2
        )
        # debug line under bbox
        cv2.putText(
            frame,
            f"spd:{speed:.1f} pose:{p['pose_ok']} fallV:{p['fall_votes']}",
            (x1, min(SCREEN_H - 10, y2 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    # Global debug overlay
    cv2.putText(
        frame,
        f"frame={frame_count} tracks={len(persons)} alerts_now={alerts_now} FPS(src)={FPS:.1f}",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord("q")]:
        break

# CLEANUP
cap.release()
cv2.destroyAllWindows()
print("▶ Sistem kapandı | Loglar:", CSV_LOG.name, "ve", JSON_LOG.name)
