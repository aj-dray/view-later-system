#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from psycopg import InterfaceError, OperationalError


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import auth
from app import database as db


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a JWT access token for the Later backend."
    )
    parser.add_argument(
        "username",
        help="Username to issue the token for.",
    )
    parser.add_argument(
        "--ensure-user",
        action="store_true",
        help="Create the user if they do not exist.",
    )
    parser.add_argument(
        "--password",
        help="Password to use when creating the user (otherwise generated).",
    )
    parser.add_argument(
        "--expires-in-hours",
        type=int,
        default=None,
        help="Override the default token lifetime.",
    )
    return parser


async def _issue_token(
    username: str,
    ensure_user: bool,
    password: str | None,
    expires_in_hours: int | None,
) -> tuple[str, bool, str | None]:
    await db.init_pool()
    await db.init_database()

    created_user = False
    created_password: str | None = None
    try:
        user: dict[str, Any] | None = await db.get_user_by_username(username)
        if user is None:
            if not ensure_user:
                raise RuntimeError(
                    f"User '{username}' not found. Re-run with --ensure-user to create it."
                )
            created_user = True
            created_password = password or secrets.token_urlsafe(16)
            new_user_id = await db.create_user(username, created_password)
            user = {"id": str(new_user_id), "username": username}

        token = auth.create_jwt_token(
            user_id=str(user["id"]),
            username=str(user["username"]),
            expires_in_hours=expires_in_hours,
        )
        return token, created_user, created_password
    finally:
        await db.close_pool()


def _warn_if_default_secret() -> None:
    if os.getenv("BACKEND_SECRET"):
        return
    print(
        "Warning: BACKEND_SECRET environment variable not set; default development secret will be used.",
        file=sys.stderr,
    )


def main() -> int:
    load_dotenv()
    _warn_if_default_secret()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        token, created_user, created_password = asyncio.run(
            _issue_token(
                username=args.username,
                ensure_user=args.ensure_user,
                password=args.password,
                expires_in_hours=args.expires_in_hours,
            )
        )
    except (OperationalError, InterfaceError) as exc:
        print(f"Database connection failed: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Unable to issue token: {exc}", file=sys.stderr)
        return 3
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 4

    print(f"Username: {args.username}")
    if created_user:
        print("User created: yes")
        if created_password:
            print(f"Generated password: {created_password}")
    else:
        print("User created: no")
    print(f"Access token: {token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
