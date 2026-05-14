from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

UPLOAD_DIR = DATA / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_PATH = ROOT / "yolov8n.pt"
JSON_LOG = ROOT / "suspicious_log.json"

ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

CONFIG = {
    "mode": "dashboard",
    "source_path": "",
    "conf": 0.50,
    "resize_w": 1280,
    "tracker": "bytetrack.yaml",
    "use_pose": False,

    "speed_th": 100,
    "dist_th": 65,
    "iou_th": 0.30,
    "cooldown": 60,

    "fight_hold": 5,
    "run_hold": 3,
    "fall_hold": 4,
}

CAM_INDEX = 0
WARMUP_HITS = 4

MOTIONLESS_SPEED = 6
MOTIONLESS_FRAMES = 60
LOITERING_FRAMES = 180

FALL_ASPECT = 1.35
POSE_VIS_MIN = 6

MIN_BOX_W = 25
MIN_BOX_H = 55
MAX_MISSING_FRAMES = 35

LOG_BEHAVIORS = {"RUN", "FIGHT", "FALL"}