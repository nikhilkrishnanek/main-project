from core.db import get_connection
from core.security import hash_password
from core.logger import log_event
import sqlite3

def authenticate(username, password):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT password, role FROM users WHERE username=?",
        (username,)
    )
    row = cur.fetchone()
    conn.close()

    if row and row[0] == hash_password(password):
        return True, row[1]

    return False, None

def register_user(username, password, role="viewer"):
    """
    Public user registration with role support.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hash_password(password), role)
        )
        conn.commit()
        log_event(f"REGISTER | New user '{username}' registered as {role}")
        return True, "User registered successfully"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()
