# security.py
import sqlite3
from functools import wraps
from pathlib import Path
import bcrypt
from flask import session, redirect, url_for, request

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
DB = DATA / "users.db"

def _conn():
    return sqlite3.connect(DB)

def _init_db():
    with _conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
        """)
        con.commit()

def ensure_default_admin(username="admin", password="admin123"):
    _init_db()
    with _conn() as con:
        cur = con.execute("SELECT username FROM users WHERE username=?", (username,))
        if cur.fetchone():
            return
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        con.execute(
            "INSERT INTO users(username, password_hash, role) VALUES(?,?,?)",
            (username, pw_hash, "admin")
        )
        con.commit()

def verify_login(username: str, password: str):
    _init_db()
    if not username or not password:
        return None
    with _conn() as con:
        cur = con.execute(
            "SELECT username, password_hash, role FROM users WHERE username=?",
            (username.strip(),)
        )
        row = cur.fetchone()
        if not row:
            return None
        u, pw_hash, role = row
        if bcrypt.checkpw(password.encode("utf-8"), pw_hash):
            return {"username": u, "role": role}
    return None

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapper

def role_required(role: str):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            u = session.get("user")
            if not u:
                return redirect(url_for("login"))
            if u.get("role") != role:
                return ("Forbidden", 403)
            return fn(*args, **kwargs)
        return wrapper
    return deco
