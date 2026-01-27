import sqlite3
import os
from core.security import hash_password

# Use os.path.join for cross-platform compatibility and ensure directory exists
DB_DIR = "results"
DB_NAME = os.path.join(DB_DIR, "users.db")

if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        role TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()

def ensure_admin_exists():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
    count = cur.fetchone()[0]

    if count == 0:
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ("nikhil", hash_password("123"), "admin")
        )
        conn.commit()

    conn.close()
