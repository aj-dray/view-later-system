from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
import uuid
from typing import Any, AsyncGenerator, Literal
import numpy as np
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from google_auth_oauthlib.flow import Flow
from psycopg import errors

load_dotenv()

from . import database as db
from . import services
from . import auth
from . import schemas
from .services import newsletters as newsletter_services


# === CONFIGURATION ===


logger = logging.getLogger(__name__)

_BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", "http://localhost:8000").rstrip("/")
if not _BACKEND_PUBLIC_URL:
    _BACKEND_PUBLIC_URL = "http://localhost:8000"
BACKEND_PUBLIC_URL = _BACKEND_PUBLIC_URL

_FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000").rstrip("/")
if not _FRONTEND_BASE_URL:
    _FRONTEND_BASE_URL = "http://localhost:3000"
FRONTEND_BASE_URL = _FRONTEND_BASE_URL

_raw_gmail_credentials = os.getenv("GMAIL_AUTH_CREDENTIALS")
if _raw_gmail_credentials:
    try:
        GMAIL_CREDENTIALS: dict[str, Any] | None = json.loads(_raw_gmail_credentials)
    except json.JSONDecodeError:
        logger.error("Invalid GMAIL_AUTH_CREDENTIALS JSON provided; Gmail integration disabled")
        GMAIL_CREDENTIALS = None
else:
    GMAIL_CREDENTIALS = None

GMAIL_ACCOUNT = os.getenv("GMAIL_ACCOUNT", "me")
GMAIL_NEWSLETTER_SENDERS = [
    sender.strip()
    for sender in os.getenv("GMAIL_NEWSLETTER_SENDERS", "").split(",")
    if sender.strip()
]

_OAUTH_STATE_TTL = timedelta(minutes=10)
_GMAIL_OAUTH_STATES: dict[str, tuple[str, datetime, str | None]] = {}

_GMAIL_POLL_INTERVAL_SECONDS = max(
    60,
    int(os.getenv("GMAIL_POLL_INTERVAL_SECONDS", "300")),
)

_GMAIL_POLL_TASKS: dict[str, asyncio.Task[Any]] = {}
_GMAIL_POLL_LAST_RUN: dict[str, datetime] = {}

GMAIL_OAUTH_REDIRECT_URI = os.getenv("GMAIL_OAUTH_REDIRECT_URI")
_raw_oauth_client_config = os.getenv("GMAIL_OAUTH_CLIENT_CONFIG")
if _raw_oauth_client_config:
    try:
        GOOGLE_OAUTH_CLIENT_CONFIG: dict[str, Any] | None = json.loads(_raw_oauth_client_config)
    except json.JSONDecodeError:
        logger.error("Invalid GMAIL_OAUTH_CLIENT_CONFIG JSON provided; Gmail OAuth disabled")
        GOOGLE_OAUTH_CLIENT_CONFIG = None
else:
    client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
    redirect_uri = GMAIL_OAUTH_REDIRECT_URI
    if client_id and client_secret and redirect_uri:
        GOOGLE_OAUTH_CLIENT_CONFIG = {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uris": [redirect_uri],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }
    else:
        GOOGLE_OAUTH_CLIENT_CONFIG = None

class AccessTokenRequest(BaseModel):
    username: str | None = None
    password: str | None = None
    label: str | None = None
    expires_in_hours: int | None = Field(
        default=None,
        gt=0,
        description="Override token lifetime in hours; omit for non-expiring token.",
    )


class GmailSenderPayload(BaseModel):
    email_address: str
    label: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    await db.init_pool()
    await db.init_database()
    yield
    # Shutdown
    await db.close_pool()


app = FastAPI(title="Later System Service", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_BASE_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

"""Track running background pipeline tasks per item to allow cancellation."""
_RUNNING_PIPELINE_TASKS: dict[str, asyncio.Task[None]] = {}


def _schedule_pipeline(item_id: str, *, url: str, user_id: str) -> None:
    task = asyncio.create_task(
        _process_item_pipeline(item_id=item_id, url=url, user_id=user_id),
        name=f"process-item-{item_id}",
    )
    _RUNNING_PIPELINE_TASKS[item_id] = task

    def _cleanup(_: asyncio.Task[None]) -> None:
        _RUNNING_PIPELINE_TASKS.pop(item_id, None)

    task.add_done_callback(_cleanup)


def _build_proxy_url(slug: str) -> str:
    return f"{BACKEND_PUBLIC_URL}/items/view/{slug}"


def _build_sender_favicon(sender_email: str | None) -> str | None:
    if not sender_email:
        return None
    _, at, domain_part = sender_email.partition("@")
    if domain_part == "smol.ai": # hot fix
        domain_part = "news.smol.ai"
    if at != "@":
        return None
    domain = domain_part.strip().lower().strip(">")
    if not domain or "." not in domain:
        return None
    sanitised = "".join(ch for ch in domain if ch.isalnum() or ch in {".", "-"})
    if not sanitised:
        return None
    return f"https://www.google.com/s2/favicons?sz=64&domain={sanitised}"


def _normalise_credentials_payload(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            logger.error("Stored Gmail credentials are not valid JSON")
            return None
        if isinstance(decoded, dict):
            return decoded
    return None


def _resolve_gmail_redirect_uri() -> str:
    if GMAIL_OAUTH_REDIRECT_URI:
        return GMAIL_OAUTH_REDIRECT_URI
    if GOOGLE_OAUTH_CLIENT_CONFIG:
        for key in ("web", "installed"):
            entry = GOOGLE_OAUTH_CLIENT_CONFIG.get(key)
            if isinstance(entry, dict):
                redirects = entry.get("redirect_uris")
                if isinstance(redirects, (list, tuple)) and redirects:
                    uri = redirects[0]
                    if isinstance(uri, str) and uri.strip():
                        return uri
    raise HTTPException(status_code=503, detail="Gmail OAuth redirect URI is not configured")


def _create_gmail_flow(*, state: str | None = None) -> Flow:
    if Flow is None:
        raise HTTPException(status_code=503, detail="Gmail OAuth libraries are unavailable")
    if GOOGLE_OAUTH_CLIENT_CONFIG is None:
        raise HTTPException(status_code=503, detail="Gmail OAuth client is not configured")
    flow_kwargs: dict[str, Any] = {"redirect_uri": _resolve_gmail_redirect_uri()}
    if state:
        flow_kwargs["state"] = state
    return Flow.from_client_config(
        GOOGLE_OAUTH_CLIENT_CONFIG,
        scopes=newsletter_services.GMAIL_SCOPES,
        **flow_kwargs,
    )


def _prune_oauth_states() -> None:
    if not _GMAIL_OAUTH_STATES:
        return
    now = datetime.now(timezone.utc)
    expired = [
        state
        for state, (_, expires_at, _) in _GMAIL_OAUTH_STATES.items()
        if expires_at <= now
    ]
    for state in expired:
        _GMAIL_OAUTH_STATES.pop(state, None)


def _register_oauth_state(state: str, user_id: str, *, code_verifier: str | None) -> None:
    _prune_oauth_states()
    expires_at = datetime.now(timezone.utc) + _OAUTH_STATE_TTL
    _GMAIL_OAUTH_STATES[state] = (user_id, expires_at, code_verifier)


def _consume_oauth_state(state: str) -> tuple[str, str | None] | None:
    entry = _GMAIL_OAUTH_STATES.pop(state, None)
    if not entry:
        return None
    user_id, expires_at, code_verifier = entry
    if expires_at <= datetime.now(timezone.utc):
        return None
    return user_id, code_verifier


async def _get_user_gmail_context(
    user_id: str,
    *,
    require_senders: bool = False,
) -> tuple[dict[str, Any], list[str], str | None]:
    credentials_record = await db.get_gmail_credentials(user_id)
    credentials_info = _normalise_credentials_payload(
        credentials_record["credentials"] if credentials_record else None
    )
    email_address = credentials_record.get("email_address") if credentials_record else None
    if credentials_record and credentials_info is None:
        logger.error("Stored Gmail credentials for user appear invalid", extra={"user_id": user_id})
        raise HTTPException(status_code=503, detail="Stored Gmail credentials are invalid")

    if credentials_info is None:
        credentials_info = _normalise_credentials_payload(GMAIL_CREDENTIALS)
    if credentials_info is None:
        raise HTTPException(status_code=503, detail="Gmail credentials are not configured")

    sender_rows = await db.list_gmail_senders(user_id)
    senders = [
        str(row["email_address"]).strip()
        for row in sender_rows
        if row.get("email_address")
    ]
    if not senders:
        senders = list(GMAIL_NEWSLETTER_SENDERS)

    if require_senders and not senders:
        raise HTTPException(status_code=503, detail="No Gmail newsletter senders configured")

    return credentials_info, senders, email_address


def _create_newsletter_service(
    *,
    credentials_info: dict[str, Any],
    senders: list[str],
    user_id: str | None = None,
) -> newsletter_services.GmailNewsletterService:
    effective_senders = senders if senders else ["placeholder@example.com"]
    return newsletter_services.GmailNewsletterService(
        credentials_info=credentials_info,
        senders=effective_senders,
        user_id=user_id or GMAIL_ACCOUNT or "me",
    )


async def _discover_gmail_email(credentials_info: dict[str, Any]) -> str | None:
    try:
        service = _create_newsletter_service(
            credentials_info=credentials_info,
            senders=["placeholder@example.com"],
        )
        client = await service._ensure_client()
        profile = await service._call_gmail(lambda: client.users().getProfile(userId="me").execute())
        email = profile.get("emailAddress") if isinstance(profile, dict) else None
        if isinstance(email, str) and email.strip():
            return email.strip()
    except Exception as exc:  # pragma: no cover - best effort lookup
        logger.warning("Failed to discover Gmail profile email", exc_info=exc)
    return None


def _is_gmail_polling_active(user_id: str) -> bool:
    task = _GMAIL_POLL_TASKS.get(user_id)
    return bool(task and not task.done())


def _cancel_gmail_polling(user_id: str) -> None:
    task = _GMAIL_POLL_TASKS.pop(user_id, None)
    if task and not task.done():
        task.cancel()
    _GMAIL_POLL_LAST_RUN.pop(user_id, None)


def _ensure_gmail_polling(user_id: str) -> None:
    existing = _GMAIL_POLL_TASKS.get(user_id)
    if existing and not existing.done():
        return

    async def _loop(uid: str) -> None:
        try:
            while True:
                try:
                    result = await _import_gmail_messages(uid)
                    imported_count = len(result.get("imported", []))
                    duplicates_count = len(result.get("duplicates", []))
                    skipped_count = len(result.get("skipped", []))
                    if imported_count or skipped_count:
                        logger.info(
                            "Completed Gmail poll",
                            extra={
                                "user_id": uid,
                                "imported": imported_count,
                                "duplicates": duplicates_count,
                                "skipped": skipped_count,
                            },
                        )
                except HTTPException as exc:
                    status = getattr(exc, "status_code", None)
                    if status in {401, 403}:
                        logger.warning(
                            "Stopping Gmail poll loop due to authentication failure",
                            extra={"user_id": uid},
                        )
                        break
                    if status == 503:
                        logger.debug(
                            "Gmail poll loop waiting for configuration",
                            extra={"user_id": uid},
                        )
                    else:
                        logger.error(
                            "Gmail poll loop encountered HTTP error",
                            extra={"user_id": uid, "status": status},
                            exc_info=exc,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error(
                        "Unexpected Gmail poll failure",
                        extra={"user_id": uid},
                        exc_info=exc,
                    )
                try:
                    await asyncio.sleep(_GMAIL_POLL_INTERVAL_SECONDS)
                except asyncio.CancelledError:
                    raise
        except asyncio.CancelledError:
            logger.debug("Gmail poll loop cancelled", extra={"user_id": uid})
            raise
        finally:
            _GMAIL_POLL_TASKS.pop(uid, None)

    task = asyncio.create_task(_loop(user_id), name=f"gmail-poll-{user_id}")
    _GMAIL_POLL_TASKS[user_id] = task


async def _import_gmail_messages(
    user_id: str,
    *,
    max_results: int | None = None,
) -> dict[str, Any]:
    try:
        credentials_info, senders, _ = await _get_user_gmail_context(
            user_id,
            require_senders=True,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "Failed to load Gmail configuration",
            extra={"user_id": user_id},
            exc_info=exc,
        )
        raise HTTPException(status_code=503, detail="Unable to load Gmail configuration") from exc

    try:
        service = _create_newsletter_service(
            credentials_info=credentials_info,
            senders=senders,
            user_id=GMAIL_ACCOUNT,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "Failed to initialise Gmail newsletter service",
            extra={"user_id": user_id},
            exc_info=exc,
        )
        raise HTTPException(status_code=503, detail="Unable to initialise Gmail client") from exc

    try:
        messages = await service.poll_newsletters(max_results=max_results)
    except Exception as exc:
        logger.error(
            "Gmail polling failed",
            extra={"user_id": user_id},
            exc_info=exc,
        )
        raise HTTPException(status_code=502, detail="Failed to query Gmail") from exc

    imported: list[dict[str, Any]] = []
    duplicates: list[str] = []
    skipped: list[dict[str, str]] = []

    for message in messages:
        existing = await db.get_email_source_by_message_id(message.message_id)
        if existing:
            duplicates.append(message.message_id)
            continue

        try:
            slug = await newsletter_services.generate_unique_slug(
                message.slug_hint(),
                db.email_slug_exists,
            )
        except Exception as exc:
            logger.error(
                "Failed to generate slug for Gmail message",
                extra={"message_id": message.message_id, "user_id": user_id},
                exc_info=exc,
            )
            skipped.append({"message_id": message.message_id, "reason": "slug_generation_failed"})
            continue

        html_content = message.html
        if not html_content:
            skipped.append({"message_id": message.message_id, "reason": "missing_html"})
            continue
        resolved_url: str | None = None
        target_url = _build_proxy_url(slug)

        favicon_url = _build_sender_favicon(message.sender_email)

        saved_at = datetime.now()
        base_item = {
            "url": target_url,
            "title": message.subject or "Untitled Newsletter",
            "format": "newsletter",
            "favicon_url": favicon_url,
            "client_status": "adding",
            "client_status_at": saved_at,
            "server_status": "saved",
            "server_status_at": saved_at,
            "user_id": user_id,
        }

        try:
            item = await db.create_item(base_item)
        except errors.UniqueViolation:
            skipped.append({"message_id": message.message_id, "reason": "duplicate_item"})
            continue
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail="Unable to create item") from exc

        email_source = await db.create_email_source(
            item_id=item["id"],
            message_id=message.message_id,
            slug=slug,
            title=message.subject,
            resolved_url=resolved_url,
            html_content=html_content,
        )

        _schedule_pipeline(item["id"], url=target_url, user_id=user_id)

        imported.append(
            {
                "item_id": item["id"],
                "message_id": message.message_id,
                "slug": email_source["slug"],
                "url": target_url,
                "resolved_url": resolved_url,
            }
        )

        # Mark the email as read now that we've successfully imported it
        try:
            await service.mark_as_read(message.message_id)
        except Exception as mark_exc:  # pragma: no cover - best effort
            logger.debug(
                "Failed to mark Gmail message as read (non-fatal)",
                extra={"message_id": message.message_id, "user_id": user_id},
                exc_info=mark_exc,
            )

    _GMAIL_POLL_LAST_RUN[user_id] = datetime.now(timezone.utc)

    return {
        "imported": imported,
        "duplicates": duplicates,
        "skipped": skipped,
    }


def _validate_sender_email(value: str) -> str:
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="Email address is required")
    candidate = value.strip().lower()
    if not candidate or "@" not in candidate or candidate.startswith("@") or candidate.endswith("@"):
        raise HTTPException(status_code=400, detail="Invalid email address")
    if len(candidate) > 320:
        raise HTTPException(status_code=400, detail="Email address is too long")
    return candidate


def _to_iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return None


def _normalise_token_label(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="Label must be a string")
    label = value.strip()
    if not label:
        return None
    if len(label) > 100:
        raise HTTPException(status_code=400, detail="Label must be 100 characters or fewer")
    return label


# === MIDDLEWARE ===


@app.middleware("http")
async def attach_session(request: Request, call_next):
    request.state.session = await auth.get_session(request)
    response = await call_next(request)
    return response


# === AUTH ===


@app.post("/auth/login")
async def login(username: str = Body(...), password: str = Body(...)) -> dict[str, Any]:
    user = await db.authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = auth.create_jwt_token(
        user["user_id"],
        user["username"],
        expires_in_hours=auth.JWT_EXPIRATION_HOURS,
    )

    try:
        credentials_record = await db.get_gmail_credentials(user["user_id"])
        has_credentials = bool(credentials_record or GMAIL_CREDENTIALS)
        if has_credentials:
            sender_rows = await db.list_gmail_senders(user["user_id"])
            if sender_rows or GMAIL_NEWSLETTER_SENDERS:
                _ensure_gmail_polling(user["user_id"])
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "Failed to evaluate Gmail polling state after login",
            extra={"user_id": user["user_id"]},
            exc_info=exc,
        )

    return {"access_token": token, "token_type": "bearer"}

@app.post("/user/access-token")
async def issue_user_access_token(
    request: Request,
    payload: AccessTokenRequest | None = Body(None),
) -> dict[str, Any]:
    session = getattr(request.state, "session", None)

    label = _normalise_token_label(payload.label) if payload else None
    expires_in_hours = payload.expires_in_hours if payload else None

    if session and session.get("user_id"):
        user_id = session["user_id"]
        username = session.get("username", "")
    else:
        if not payload or not payload.username or not payload.password:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        user = await db.authenticate_user(payload.username, payload.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        user_id = user["user_id"]
        username = user["username"]

    token_id = str(uuid.uuid4())
    expires_at: datetime | None = None
    if expires_in_hours is not None:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)
    try:
        token = auth.create_jwt_token(
            user_id,
            username,
            token_id=token_id,
            expires_in_hours=expires_in_hours,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        record = await db.create_user_access_token(
            user_id=user_id,
            token_id=token_id,
            expires_at=expires_at,
            label=label,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Failed to persist access token") from exc
    return {
        "access_token": token,
        "token_type": "bearer",
        "token_id": record.get("token_id"),
        "label": record.get("label"),
        "expires_at": _to_iso(record.get("expires_at")),
        "created_at": _to_iso(record.get("created_at")),
    }


def _serialise_token_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert database token row for JSON response."""
    serialised: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            serialised[key] = value.isoformat()
        else:
            serialised[key] = value
    return serialised


@app.get("/user/access-token")
async def list_user_access_tokens(session: dict = Depends(auth.require_session)) -> dict[str, Any]:
    tokens = await db.list_user_access_tokens(session["user_id"])
    return {"tokens": [_serialise_token_row(token) for token in tokens]}


@app.delete("/user/access-token/{token_id}", status_code=204)
async def delete_user_access_token(token_id: str, session: dict = Depends(auth.require_session)) -> None:
    revoked = await db.revoke_user_access_token(user_id=session["user_id"], token_id=token_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="Token not found")


# === USERS ===


@app.get("/user/me")
async def get_current_user(session: dict = Depends(auth.require_session)) -> dict:
    return {
        "user_id": session.get("user_id"),
        "username": session.get("username"),
    }


@app.post("/user/add", status_code=201)
async def add_user(username: str = Body(...), password: str = Body(...)) -> dict:
    try:
        user_id = await db.create_user(username, password)
        return {"user_id": user_id, "username": username}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/demo/request", status_code=201)
async def request_demo_account() -> dict:
    """Provision a new demo account cloned from the base 'demo' user.

    Returns the generated username and a 6-character password.
    """
    source = await db.get_user_by_username("demo")
    if not source:
        raise HTTPException(status_code=404, detail="Base demo account not found")

    username = await auth.generate_unique_demo_username(db)
    password = auth.random_password(6)

    try:
        new_user_id = await db.create_user(username, password)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Unable to create demo user") from exc

    # Clone items and item_chunks from base demo user
    try:
        await db.clone_user_data(source_user_id=str(source["id"]), target_user_id=str(new_user_id))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Failed to clone demo data") from exc

    return {"username": username, "password": password}


# === ITEMS ===


async def _process_item_pipeline(*, item_id: str, url: str, user_id: str) -> None:
    """Run extraction as a background task so the HTTP response so we can return
    immediately, allowing clients to poll for incremental updates.
    """
    log_extra = {"item_id": item_id, "user_id": user_id}
    stage = "initialization"
    is_newsletter = False
    newsletter_title: str | None = None
    newsletter_favicon: str | None = None

    async def mark_error(message: str, exc: Exception | None = None) -> None:
        """Mark item as failed and log error."""
        logger.error(f"{message} (stage: {stage})", extra=log_extra, exc_info=exc)
        try:
            await db.update_item(
                {"client_status": "error", "client_status_at": datetime.now()},
                item_id=item_id,
                user_id=user_id,
            )
        except Exception:
            pass  # Best effort to mark error state

    try:
        base_item = await db.get_item(item_id, ["format", "title", "favicon_url"], user_id)
        if base_item is None:
            return  # Item was deleted, silently exit
        if base_item.get("format") == "newsletter":
            is_newsletter = True
            newsletter_title = base_item.get("title")
            newsletter_favicon = base_item.get("favicon_url")

        # Stage 1: Extract metadata from URL
        stage = "extraction"
        item_updates = await services.extract_data(url)
        if not item_updates:
            await mark_error("Extraction returned no metadata")
            return

        if isinstance(item_updates, dict):
            if is_newsletter:
                # Preserve the original email subject/title and format
                item_updates.pop("title", None)
                item_updates["format"] = "newsletter"
                item_updates.pop("favicon_url", None)
                if newsletter_favicon:
                    item_updates["favicon_url"] = newsletter_favicon
            item_updates.pop("client_status", None)  # Strip client_status if present

        item = await db.update_item(item_updates, item_id=item_id, user_id=user_id)
        if item is None:
            return  # Item was deleted, silently exit

        # Stage 2: Generate summary
        stage = "summary"
        summary_updates = await services.generate_data(item, user_id=user_id)
        if isinstance(summary_updates, dict):
            summary_updates.pop("client_status", None)
            # Override type and preserve title for email imports
            if is_newsletter:
                summary_updates["type"] = "newsletter"
                summary_updates["format"] = "newsletter"
                summary_updates.pop("favicon_url", None)
                if newsletter_favicon:
                    summary_updates["favicon_url"] = newsletter_favicon
                subject_title = newsletter_title
                if not subject_title:
                    try:
                        async with db.get_connection() as conn:
                            async with conn.cursor(row_factory=db.dict_row) as cur:
                                await cur.execute(
                                    "SELECT title FROM email_sources WHERE item_id = %s",
                                    (item_id,)
                                )
                                row = await cur.fetchone()
                                if row:
                                    subject_title = row.get("title")
                    except Exception:
                        subject_title = None
                if subject_title:
                    summary_updates["title"] = subject_title
                else:
                    summary_updates.pop("title", None)

        item = await db.update_item(summary_updates, item_id=item_id, user_id=user_id)
        if item is None:
            return  # Item was deleted, silently exit

        # Stage 3: Create embeddings and chunks
        stage = "embedding"
        embed_updates, item_chunks = await services.index_item(item, item_id=item_id, user_id=user_id)
        if isinstance(embed_updates, dict):
            embed_updates.pop("client_status", None)

        # Mark as queued when successfully completed
        embed_updates.update({
            "client_status": "queued",
            "client_status_at": datetime.now(),
        })

        updated_item = await db.update_item(embed_updates, item_id=item_id, user_id=user_id)
        if updated_item is None:
            return  # Item was deleted, silently exit

        await db.add_item_chunks(item_id=item_id, chunks=item_chunks)

    except ValueError as exc:
        await mark_error(f"Failed to persist updates", exc)
    except Exception as exc:
        await mark_error(f"Pipeline failed", exc)


@app.post("/items/add", status_code=201)
async def add_item(
    url: str = Body(..., embed=True),
    session: dict = Depends(auth.require_session),
) -> dict[str, str]:
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")
    submitted_url_raw = url.strip()
    try:
        submitted_url = services.extracting.prepare_url(submitted_url_raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    saved_at = datetime.now()

    base_item = {
        "url": submitted_url,
        "client_status": "adding",
        "client_status_at": saved_at,
        "server_status": "saved",
        "server_status_at": saved_at,
        "user_id": user_id,
    }

    try:
        item = await db.create_item(base_item)
    except errors.UniqueViolation as exc:
        # URL already exists for this user – log for visibility then return 409
        logger.warning(
            "Duplicate item add attempt",
            extra={"user_id": user_id, "url": submitted_url},
            exc_info=None,
        )
        raise HTTPException(status_code=409, detail="URL already exists") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail="Unable to create item") from exc

    item_id = item["id"]

    # Trigger the pipeline asynchronously so clients can stream updates
    _schedule_pipeline(item_id, url=submitted_url, user_id=user_id)

    return {"item_id": item_id}


@app.get("/items/select")
async def get_items(
    *,
    columns: list[str] | None = Query(
        default=None,
        description="Columns to select (defaults to all public columns)",
    ),
    filters: list[str] | None = Query(
        default=None,
        description="Filters in format 'column:operator:value' (e.g., 'client_status:IN:saved,queued')",
        alias="filter"
    ),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    order_by: str | None = Query(
        default="created_at",
        description="Column to order by",
    ),
    order: Literal["asc", "desc"] = Query("desc"),
    session: dict = Depends(auth.require_session),
) -> list[dict]:
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")

    # Use default columns if none specified
    selected_columns = columns if columns else schemas.ITEM_PUBLIC_COLS

    # Validate columns
    invalid_columns = [col for col in selected_columns if col not in schemas.ITEM_PUBLIC_COLS]
    if invalid_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid columns: {invalid_columns}"
        )

    # Parse filters
    parsed_filters = []
    if filters:
        for filter_str in filters:
            try:
                parts = filter_str.split(":", 2)
                if len(parts) != 3:
                    raise ValueError("Filter must have format 'column:operator:value'")

                column, operator, value_str = parts

                # Validate column
                if column not in schemas.ITEM_PUBLIC_COLS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid filter column: {column}"
                    )

                # Parse value based on operator
                if operator.upper() == "IN":
                    value = value_str.split(",")
                else:
                    value = value_str

                parsed_filters.append((column, operator.upper(), value))

            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid filter format: {filter_str}. {str(e)}"
                )
    # Validate order_by column
    if order_by and order_by not in schemas.ITEM_PUBLIC_COLS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid order_by column: {order_by}"
        )

    rows = await db.get_items(
        columns=selected_columns,
        filters=parsed_filters,
        user_id=user_id,
        limit=limit,
        offset=offset,
        order_by=order_by,
        order_direction=order,
    )
    return rows


@app.post("/items/update")
async def update_items(
    *,
    item_ids: list[str] = Body(..., embed=True),
    updates: dict[str, Any] = Body(..., embed=True),
    session: dict = Depends(auth.require_session),
) -> dict:
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")

    if not item_ids:
        raise HTTPException(status_code=400, detail="No item_ids provided")
    if not isinstance(updates, dict) or not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    results: dict[str, dict[str, Any]] = {}
    for item_id in item_ids:
        try:
            item = await db.update_item(updates, item_id=item_id, user_id=user_id)
            if item is None:
                results[item_id] = {"updated": False, "error": "Not found"}
            else:
                results[item_id] = {"updated": True}
        except ValueError as exc:
            results[item_id] = {"updated": False, "error": str(exc)}
        except Exception as exc:
            results[item_id] = {"updated": False, "error": "Unexpected error"}

    return {"results": results}


@app.post("/items/delete")
async def delete_items(
    *,
    item_ids: list[str] = Body(..., embed=True),
    session: dict = Depends(auth.require_session),
) -> dict:
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")

    if not item_ids:
        raise HTTPException(status_code=400, detail="No item_ids provided")

    results: dict[str, bool] = {}
    for item_id in item_ids:
        try:
            deleted = await db.delete_item(item_id=item_id, user_id=user_id)
        except Exception:
            deleted = False
        results[item_id] = deleted

        # Cancel any running background task for this item
        task = _RUNNING_PIPELINE_TASKS.pop(item_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    return {"results": results}


# === GMAIL NEWSLETTERS ===


@app.post("/gmail/poll")
async def poll_gmail_newsletters(
    *,
    session: dict = Depends(auth.require_session),
    max_results: int | None = Query(None, ge=1, le=50),
) -> dict[str, Any]:
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")

    result = await _import_gmail_messages(user_id, max_results=max_results)
    _ensure_gmail_polling(user_id)
    return result


@app.get("/gmail/settings")
async def get_gmail_settings(*, session: dict = Depends(auth.require_session)) -> dict[str, Any]:
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")

    credentials_record = await db.get_gmail_credentials(user_id)
    sender_rows = await db.list_gmail_senders(user_id)

    senders = [
        {
            "id": row.get("id"),
            "email_address": row.get("email_address"),
            "label": row.get("label"),
            "created_at": _to_iso(row.get("created_at")),
        }
        for row in sender_rows
    ]

    response: dict[str, Any] = {
        "connected": bool(credentials_record),
        "email_address": credentials_record.get("email_address") if credentials_record else None,
        "token_expiry": _to_iso(credentials_record.get("token_expiry")) if credentials_record else None,
        "updated_at": _to_iso(credentials_record.get("updated_at")) if credentials_record else None,
        "senders": senders,
        "default_senders": list(GMAIL_NEWSLETTER_SENDERS),
        "oauth_available": bool(Flow and GOOGLE_OAUTH_CLIENT_CONFIG),
        "legacy_credentials": bool(GMAIL_CREDENTIALS),
        "has_senders": bool(senders) or bool(GMAIL_NEWSLETTER_SENDERS),
        "polling_active": _is_gmail_polling_active(user_id),
        "polling_interval_seconds": _GMAIL_POLL_INTERVAL_SECONDS,
        "last_poll_at": _to_iso(_GMAIL_POLL_LAST_RUN.get(user_id)),
    }
    return response


@app.post("/gmail/senders")
async def create_gmail_sender(
    payload: GmailSenderPayload,
    session: dict = Depends(auth.require_session),
) -> dict[str, Any]:
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")

    email_address = _validate_sender_email(payload.email_address)
    label = payload.label.strip() if isinstance(payload.label, str) and payload.label.strip() else None
    try:
        sender = await db.add_gmail_sender(
            user_id=user_id,
            email_address=email_address,
            label=label,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    credentials_record = await db.get_gmail_credentials(user_id)
    if credentials_record or GMAIL_CREDENTIALS:
        _ensure_gmail_polling(user_id)

    sender_response = {
        "id": sender.get("id"),
        "email_address": sender.get("email_address"),
        "label": sender.get("label"),
        "created_at": _to_iso(sender.get("created_at")),
    }
    return {"sender": sender_response}


@app.delete("/gmail/senders/{sender_id}")
async def delete_gmail_sender(sender_id: str, *, session: dict = Depends(auth.require_session)) -> dict[str, Any]:
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")

    deleted = await db.remove_gmail_sender(user_id=user_id, sender_id=sender_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Sender not found")
    remaining = await db.list_gmail_senders(user_id)
    if not remaining and not GMAIL_NEWSLETTER_SENDERS:
        _cancel_gmail_polling(user_id)
    return {"deleted": True}


@app.post("/gmail/auth/start")
async def gmail_auth_start(*, session: dict = Depends(auth.require_session)) -> dict[str, Any]:
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")

    flow = _create_gmail_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    code_verifier = getattr(flow, "code_verifier", None)
    _register_oauth_state(state, user_id, code_verifier=code_verifier)
    return {"auth_url": auth_url}


@app.post("/gmail/auth/disconnect")
async def gmail_auth_disconnect(*, session: dict = Depends(auth.require_session)) -> dict[str, Any]:
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")
    disconnected = await db.delete_gmail_credentials(user_id)
    if disconnected:
        _cancel_gmail_polling(user_id)
    return {"disconnected": disconnected}


@app.get("/gmail/auth/callback")
async def gmail_auth_callback(
    state: str = Query(...),
    code: str = Query(...),
    error: str | None = Query(None),
) -> RedirectResponse:
    if error:
        logger.error("Gmail OAuth callback reported error", extra={"error": error})
        return RedirectResponse(url=f"{FRONTEND_BASE_URL}/settings?gmail=error")

    consumed = _consume_oauth_state(state)
    if not consumed:
        logger.error("Gmail OAuth callback with invalid or expired state", extra={"state": state})
        return RedirectResponse(url=f"{FRONTEND_BASE_URL}/settings?gmail=error")

    user_id, code_verifier = consumed

    try:
        flow = _create_gmail_flow(state=state)
    except HTTPException as exc:
        logger.error("Failed to initialise Gmail OAuth flow", exc_info=exc, extra={"user_id": user_id})
        return RedirectResponse(url=f"{FRONTEND_BASE_URL}/settings?gmail=error")

    if code_verifier:
        flow.code_verifier = code_verifier

    try:
        await asyncio.to_thread(flow.fetch_token, code=code)
    except Exception as exc:
        logger.error("Failed to exchange Gmail OAuth code", exc_info=exc, extra={"user_id": user_id})
        return RedirectResponse(url=f"{FRONTEND_BASE_URL}/settings?gmail=error")

    credentials = getattr(flow, "credentials", None)
    if credentials is None:
        logger.error("Gmail OAuth flow returned no credentials", extra={"user_id": user_id})
        return RedirectResponse(url=f"{FRONTEND_BASE_URL}/settings?gmail=error")

    try:
        credentials_dict = json.loads(credentials.to_json())
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to serialise Gmail credentials", exc_info=exc, extra={"user_id": user_id})
        return RedirectResponse(url=f"{FRONTEND_BASE_URL}/settings?gmail=error")

    token_expiry = getattr(credentials, "expiry", None)
    if isinstance(token_expiry, datetime) and token_expiry.tzinfo is None:
        token_expiry = token_expiry.replace(tzinfo=timezone.utc)

    email_address = await _discover_gmail_email(credentials_dict)
    try:
        await db.upsert_gmail_credentials(
            user_id=user_id,
            credentials=credentials_dict,
            email_address=email_address,
            token_expiry=token_expiry,
        )
    except Exception as exc:
        logger.error("Failed to persist Gmail credentials", exc_info=exc, extra={"user_id": user_id})
        return RedirectResponse(url=f"{FRONTEND_BASE_URL}/settings?gmail=error")

    _ensure_gmail_polling(user_id)

    async def _kickoff_initial_poll() -> None:
        try:
            await _import_gmail_messages(user_id)
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug(
                "Initial Gmail poll after OAuth connect failed",
                extra={"user_id": user_id},
                exc_info=exc,
            )

    asyncio.create_task(_kickoff_initial_poll())

    return RedirectResponse(url=f"{FRONTEND_BASE_URL}/settings?gmail=connected")


@app.get("/items/view/{slug}")
async def view_newsletter_html(slug: str, request: Request) -> HTMLResponse:
    metadata = await db.get_email_source_by_slug(slug)
    if not metadata:
        raise HTTPException(status_code=404, detail="Newsletter not found")

    owner_id = metadata.get("user_id")
    session = getattr(request.state, "session", None)
    if session and owner_id and str(owner_id) != session.get("user_id"):
        raise HTTPException(status_code=404, detail="Newsletter not found")
    owner_id_str = str(owner_id) if owner_id else None

    html_content = metadata.get("html_content")
    if not html_content:
        if not owner_id_str:
            raise HTTPException(status_code=404, detail="Newsletter not found")
        try:
            credentials_info, senders, _ = await _get_user_gmail_context(owner_id_str, require_senders=False)
            service = _create_newsletter_service(
                credentials_info=credentials_info,
                senders=senders,
                user_id=GMAIL_ACCOUNT,
            )
            html_content = await service.fetch_message_html(metadata["message_id"])
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to fetch Gmail HTML",
                extra={"message_id": metadata.get("message_id"), "user_id": owner_id_str},
                exc_info=exc,
            )
            raise HTTPException(status_code=502, detail="Unable to retrieve newsletter HTML") from exc

        if not html_content:
            raise HTTPException(status_code=404, detail="Newsletter HTML unavailable")

        await db.update_email_source_html(
            email_source_id=metadata["id"],
            html_content=html_content,
        )

    return HTMLResponse(content=html_content)


# === CLUSTERING ===


@app.get("/clusters/dimensional-reduction")
async def generate_graph(
    request: Request,
    item_ids: list[str] = Query(...),
    mode: Literal["pca", "tsne", "umap"] = Query("umap"),
    session: dict = Depends(auth.require_session),
) -> dict:
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")
    rows = await db.get_items(
        columns=["id", "mistral_embedding"],
        filters=[("id", "IN", item_ids)],
        user_id=user_id,
        limit=None
    )
    # Extract extra parameters from query string
    kwargs = {}
    known_params = {"item_ids", "mode"}
    for key, value in request.query_params.items():
        if key not in known_params:
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    kwargs[key] = float(value)
                else:
                    kwargs[key] = int(value)
            except ValueError:
                kwargs[key] = value

    if mode == "pca":
        reduced_embeddings = services.clustering.pca(rows, d=2, **kwargs)
    elif mode == "tsne":
        reduced_embeddings = services.clustering.tsne(rows, d=2, **kwargs)
    elif mode == "umap":
        reduced_embeddings = services.clustering.umap(rows, d=2, **kwargs)

    ordered_ids = [row["id"] for row in rows]

    # Handle different return types from clustering functions
    try:
        # Convert numpy array to list
        embeddings_list = np.array(reduced_embeddings).tolist()
    except (ValueError, TypeError):
        # Fallback for other types - cast to Any to avoid type checker issues
        from typing import cast
        embeddings_list = cast(list, reduced_embeddings)

    return {
        "reduced_embeddings": embeddings_list,
        "item_ids": ordered_ids,
    }


@app.get("/clusters/generate")
async def get_clustering(
    request: Request,
    item_ids: list[str] = Query(...),
    mode: Literal["kmeans", "hca", "dbscan"] = Query("kmeans"),
    session: dict = Depends(auth.require_session),
) -> dict:
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")

    rows = await db.get_items(
        columns=["id", "mistral_embedding"],
        filters=[("id", "IN", item_ids)],
        user_id=user_id,
        limit=None
    )

    # Extract extra parameters from query string
    kwargs = {}
    known_params = {"item_ids", "mode"}
    for key, value in request.query_params.items():
        if key not in known_params:
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    kwargs[key] = float(value)
                else:
                    kwargs[key] = int(value)
            except ValueError:
                kwargs[key] = value

    if mode == "kmeans":
        clusters = services.clustering.kmeans(rows, **kwargs)
    elif mode == "hca":
        clusters = services.clustering.hca(rows, **kwargs)
    elif mode == "dbscan":
        clusters = services.clustering.dbscan(rows, **kwargs)

    ordered_ids = [row["id"] for row in rows]

    # Handle different return types from clustering functions
    try:
        # Convert numpy array to list
        clusters_list = np.array(clusters).tolist()
    except (ValueError, TypeError):
        # Fallback for other types - cast to Any to avoid type checker issues
        from typing import cast
        clusters_list = cast(list, clusters)

    return {
        "clusters": clusters_list,
        "item_ids": ordered_ids,
    }


@app.get("/clusters/label")
async def get_cluster_labels(
    *,
    item_ids: list[str] = Query(...),
    clusters: str = Query(..., description="JSON encoded list of cluster definitions"),
    session: dict = Depends(auth.require_session)
) -> dict:
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")

    rows = await db.get_items(
        columns=["id", "summary"],
        filters=[("id", "IN", item_ids)],
        user_id=user_id,
        limit=None
    )

    try:
        parsed_clusters = json.loads(clusters)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid clusters payload") from exc

    if not isinstance(parsed_clusters, list):
        raise HTTPException(status_code=400, detail="Clusters payload must be a list")

    if len(parsed_clusters) != len(item_ids):
        raise HTTPException(status_code=400, detail="Clusters payload length must match item IDs")

    # Database results for an IN clause are not guaranteed to preserve order, so align them
    # manually with the item_ids sequence the frontend used when preparing clusters.
    rows_by_id = {row["id"]: row for row in rows}
    ordered_rows: list[dict[str, Any]] = []
    ordered_clusters: list[int] = []

    for idx, item_id in enumerate(item_ids):
        row = rows_by_id.get(item_id)
        if row is None:
            # Skip missing rows to avoid misaligning cluster indices with summaries
            continue
        ordered_rows.append(row)
        ordered_clusters.append(parsed_clusters[idx])

    labels = services.clustering.label(ordered_clusters, ordered_rows)

    return {"labels": labels}


# === SEARCH ===


@app.get("/items/search")
async def search_items(
    query: str = Query(...),
    mode: Literal["lexical", "semantic", "agentic"] = Query("lexical"),
    limit: int = Query(10, ge=1, le=10),
    columns: list[str] | None = Query(None),
    session: dict = Depends(auth.require_session),
) -> dict:
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")

    # If the client didn't request specific columns, include helpful defaults
    effective_columns = columns or ["title", "summary"]

    if mode == "lexical":
        search_results = await services.searching.lexical(
            user_id=user_id,
            query=query,
            limit=limit,
            columns=effective_columns,
        )
    elif mode == "semantic":
        search_results = await services.searching.semantic(
            user_id=user_id,
            query=query,
            limit=limit,
            columns=effective_columns,
        )
    elif mode == "agentic":
        return StreamingResponse(
            services.searching.agentic(
                user_id=user_id,
                query=query,
                limit=limit,
                columns=effective_columns,
            ),
            media_type="text/event-stream",
        )

    return {"results": search_results}


# === USER SETTINGS ===


@app.get("/user/settings/{setting_type}/{setting_key}")
async def get_user_setting(
    setting_type: str,
    setting_key: str,
    session: dict = Depends(auth.require_session)
) -> dict[str, Any]:
    """Get a specific user setting."""
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")
    setting_value = await db.get_user_setting(user_id, setting_type, setting_key)
    return {"setting_value": setting_value or {}}


@app.get("/user/settings/{setting_type}")
async def get_user_settings_by_type(
    setting_type: str,
    session: dict = Depends(auth.require_session)
) -> dict[str, Any]:
    """Get all user settings of a specific type."""
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")
    settings = await db.get_user_settings_by_type(user_id, setting_type)
    return {"settings": settings}


@app.put("/user/settings/{setting_type}/{setting_key}")
async def set_user_setting(
    setting_type: str,
    setting_key: str,
    setting_value: dict[str, Any] = Body(...),
    session: dict = Depends(auth.require_session),
) -> dict[str, Any]:
    """Set a user setting."""
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")
    await db.set_user_setting(user_id, setting_type, setting_key, setting_value)
    return {"success": True}


@app.patch("/user/settings/{setting_type}/{setting_key}")
async def update_user_setting_field(
    setting_type: str,
    setting_key: str,
    field_key: str = Body(...),
    field_value: Any = Body(...),
    session: dict = Depends(auth.require_session),
) -> dict[str, Any]:
    """Update a single field within a user setting."""
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user context")
    await db.update_user_setting_field(user_id, setting_type, setting_key, field_key, field_value)
    return {"success": True}


if __name__ == "__main__":
    pass
