from core.db import get_connection
from core.security import hash_password
from core.logger import log_event


from core.auth import register_user as auth_register_user


def register_user(username, password):
    """
    Public user registration (Legacy wrapper).
    """
    return auth_register_user(username, password, role="viewer")


def create_user(username, password, role):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
        (username, hash_password(password), role)
    )

    conn.commit()
    conn.close()
    log_event(f"ADMIN created user '{username}' with role '{role}'")


def delete_user(username):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()
    log_event(f"ADMIN deleted user '{username}'")


def change_password(username, old_password, new_password):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT password FROM users WHERE username=?",
        (username,)
    )
    row = cur.fetchone()

    if not row or row[0] != hash_password(old_password):
        conn.close()
        return False, "Old password incorrect"

    cur.execute(
        "UPDATE users SET password=? WHERE username=?",
        (hash_password(new_password), username)
    )

    conn.commit()
    conn.close()
    log_event(f"User '{username}' changed password")
    return True, "Password updated successfully"


def update_user_role(username, new_role):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "UPDATE users SET role=? WHERE username=?",
        (new_role, username)
    )

    conn.commit()
    conn.close()
    log_event(f"ADMIN updated role of '{username}' to '{new_role}'")

def list_users():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, role FROM users")
    users = cur.fetchall()
    conn.close()
    return users
