"""JWT authentication utilities for FastAPI service."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any
import jwt
from fastapi import HTTPException, Request
import string
import secrets


# === VARIABLES ===


SALT_BYTES = 32
PBKDF2_ITERATIONS = 100000
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


# === UTILITIES


def _get_secret() -> str:
    return ( os.getenv("BACKEND_SECRET")
        or "dev-secret"
    )


def create_jwt_token(
    user_id: str,
    username: str,
    *,
    token_id: str | None = None,
    expires_in_hours: int | None = None,
) -> str:
    """Create a JWT token for the given user.

    Args:
        user_id: Database identifier for the user.
        username: Username used for display/debugging.
        expires_in_hours: Optional override for the token lifetime.
    """
    now = datetime.utcnow()
    payload = {
        "user_id": user_id,
        "username": username,
        "iat": now,
    }
    if token_id:
        payload["jti"] = token_id
    if expires_in_hours is not None:
        if expires_in_hours <= 0:
            raise ValueError("Token expiry must be a positive number of hours")
        payload["exp"] = now + timedelta(hours=expires_in_hours)
    return jwt.encode(payload, _get_secret(), algorithm=JWT_ALGORITHM)


def _decode_jwt_token(token: str) -> dict[str, Any] | None:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, _get_secret(), algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.InvalidTokenError:
        return None


async def get_session(request: Request) -> dict[str, Any] | None:
    """Extract session from Authorization header with Bearer token."""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None

    try:
        scheme, token = auth_header.split(" ", 1)
        if scheme.lower() != "bearer":
            return None
    except ValueError:
        return None

    payload = _decode_jwt_token(token)
    if not payload:
        return None

    session = {
        "user_id": str(payload.get("user_id", "")),
        "username": str(payload.get("username", "")),
        "issued_at": int(payload.get("iat", 0)),
        "expires_at": int(payload.get("exp", 0)),
    }

    token_id = payload.get("jti")
    if token_id:
        from . import database as db  # Local import to avoid circular dependency

        token_record = await db.get_user_access_token(str(token_id))
        if not token_record:
            return None
        if token_record.get("revoked_at") is not None:
            return None

        expires_at = token_record.get("expires_at")
        if isinstance(expires_at, datetime):
            if expires_at.tzinfo is not None:
                if expires_at.astimezone(timezone.utc) < datetime.now(timezone.utc):
                    return None
            elif expires_at < datetime.utcnow():
                return None

        if str(token_record.get("user_id")) != session["user_id"]:
            return None

        session["token_id"] = str(token_id)

    return session


def require_session(request: Request) -> dict[str, Any]:
    session = getattr(request.state, "session", None)
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session


def hash_password(password: str) -> str:
    salt = os.urandom(SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    encoded_salt = base64.b64encode(salt).decode("ascii")
    encoded_hash = base64.b64encode(derived).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${encoded_salt}${encoded_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_str, encoded_salt, encoded_hash = stored_hash.split(
            "$", 3
        )
    except ValueError:
        return False

    if algorithm != "pbkdf2_sha256":
        return False

    try:
        iterations = int(iterations_str)
    except ValueError:
        return False

    try:
        salt = base64.b64decode(encoded_salt)
        expected_hash = base64.b64decode(encoded_hash)
    except (binascii.Error, ValueError):
        return False

    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )

    if len(derived) != len(expected_hash):
        return False

    return hmac.compare_digest(derived, expected_hash)


def random_password(length: int = 6) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


async def generate_unique_demo_username(db) -> str:
    """Generate a unique username of the form 'demo{id}'."""
    # We attempt a simple numeric suffix strategy to keep usernames readable.
    # Try successive integers until insert succeeds (bounded attempts for safety).
    for attempt in range(1, 10000):
        candidate = f"demo{attempt}"
        existing = await db.get_user_by_username(candidate)
        if not existing:
            return candidate
    # Fallback to a random suffix if sequential strategy fails unexpectedly
    return f"demo{secrets.randbelow(10_000_000)}"

__all__ = [
    "SALT_BYTES",
    "PBKDF2_ITERATIONS",
    "JWT_ALGORITHM",
    "JWT_EXPIRATION_HOURS",
    "create_jwt_token",
    "get_session",
    "require_session",
    "hash_password",
    "verify_password",
    "random_password",
    "generate_unique_demo_username"
]
