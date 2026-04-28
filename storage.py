# storage.py
import json, hashlib
from pathlib import Path
from cryptography.fernet import Fernet

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

KEY_FILE = DATA / "fernet.key"
LOG_ENC = DATA / "logs.enc"
CHAIN_FILE = DATA / "logs.chain"
TIMELINE_CSV = DATA / "timeline.csv"

def _load_or_create_key() -> bytes:
    if KEY_FILE.exists():
        k = KEY_FILE.read_bytes().strip()
        Fernet(k)  # validates format
        return k
    k = Fernet.generate_key()
    KEY_FILE.write_bytes(k)
    return k

FERNET = Fernet(_load_or_create_key())

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def append_secure_event(ev: dict) -> None:
    payload = json.dumps(ev, ensure_ascii=False).encode("utf-8")
    token = FERNET.encrypt(payload)

    with open(LOG_ENC, "ab") as f:
        f.write(token + b"\n")

    prev = "0" * 64
    if CHAIN_FILE.exists() and CHAIN_FILE.stat().st_size > 0:
        prev = CHAIN_FILE.read_text(encoding="utf-8").splitlines()[-1].strip()

    h = _sha256((prev + token.decode("utf-8")).encode("utf-8"))
    with open(CHAIN_FILE, "a", encoding="utf-8") as f:
        f.write(h + "\n")

def verify_log_integrity() -> dict:
    if not LOG_ENC.exists():
        return {"ok": True, "reason": "no encrypted logs yet", "count": 0}

    tokens = LOG_ENC.read_bytes().splitlines()
    chain = CHAIN_FILE.read_text(encoding="utf-8").splitlines() if CHAIN_FILE.exists() else []

    if len(chain) != len(tokens):
        return {"ok": False, "reason": "chain length mismatch", "tokens": len(tokens), "chain": len(chain)}

    prev = "0" * 64
    for i, token in enumerate(tokens):
        try:
            _ = FERNET.decrypt(token)
        except Exception as e:
            return {"ok": False, "reason": f"decrypt failed at line {i}", "error": str(e)}

        expected = _sha256((prev + token.decode("utf-8")).encode("utf-8"))
        if chain[i].strip() != expected:
            return {"ok": False, "reason": f"hash mismatch at line {i}"}
        prev = expected

    return {"ok": True, "count": len(tokens)}
