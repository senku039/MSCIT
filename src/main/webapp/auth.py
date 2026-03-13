"""Authentication and session utilities."""

from __future__ import annotations

import re
import sqlite3
from functools import wraps
from pathlib import Path
from typing import Any

from flask import current_app, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _get_connection() -> sqlite3.Connection:
    db_path = Path(current_app.config["USER_DB_PATH"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_user_table() -> None:
    """Ensure the minimal user table exists."""
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.commit()


def validate_credentials(email: str, password: str) -> str | None:
    normalized_email = email.strip().lower()
    if not normalized_email or not EMAIL_PATTERN.match(normalized_email):
        return "Please provide a valid email address."
    if not password:
        return "Password is required."
    return None


def create_user(email: str, password: str) -> tuple[bool, str]:
    error = validate_credentials(email, password)
    if error:
        return False, error

    normalized_email = email.strip().lower()
    password_hash = generate_password_hash(password)

    try:
        with _get_connection() as conn:
            conn.execute(
                "INSERT INTO users(email, password_hash) VALUES (?, ?)",
                (normalized_email, password_hash),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."

    return True, "Account created successfully."


def authenticate_user(email: str, password: str) -> tuple[dict[str, Any] | None, str | None]:
    error = validate_credentials(email, password)
    if error:
        return None, error

    normalized_email = email.strip().lower()
    with _get_connection() as conn:
        user = conn.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?",
            (normalized_email,),
        ).fetchone()

    if user is None or not check_password_hash(user["password_hash"], password):
        return None, "Invalid email or password."

    return {"id": user["id"], "email": user["email"]}, None


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect(url_for("api.login_page", next=request.path))
        return view_func(*args, **kwargs)

    wrapped.__name__ = view_func.__name__
    return wrapped
