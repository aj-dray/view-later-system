"""Utilities for importing newsletter content from Gmail."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime, parseaddr
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, Sequence

try:  # pragma: no cover - optional dependency during tests
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:  # pragma: no cover - optional dependency
    build = None  # type: ignore[assignment]
    HttpError = Exception  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency during tests
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
except ImportError:  # pragma: no cover - optional dependency
    Credentials = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from google.oauth2.credentials import Credentials as GoogleCredentials
else:
    GoogleCredentials = Any

logger = logging.getLogger(__name__)

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]

@dataclass(slots=True)
class NewsletterMessage:
    """Representation of a newsletter discovered via Gmail."""

    message_id: str
    subject: str | None
    received_at: datetime | None
    sender_email: str | None
    html: str | None

    def slug_hint(self) -> str:
        if self.subject:
            return self.subject
        return self.message_id


class GmailNewsletterService:
    """Client wrapper for polling Gmail newsletter messages."""

    def __init__(
        self,
        *,
        credentials_info: dict[str, Any] | GoogleCredentials,
        senders: Sequence[str],
        user_id: str = "me",
        gmail_client: Any | None = None,
    ) -> None:
        if not senders:
            raise ValueError("At least one sender must be configured")
        self._credentials_info = credentials_info
        self._senders = [sender.strip() for sender in senders if sender.strip()]
        if not self._senders:
            raise ValueError("Sender list cannot be empty")
        self._user_id = user_id
        self._gmail_client = gmail_client
        self._client_lock = asyncio.Lock()

    async def poll_newsletters(self, *, max_results: int | None = None) -> list[NewsletterMessage]:
        """Return newsletter messages for the configured senders."""

        client = await self._ensure_client()
        query = " OR ".join(f"(from:{sender} is:unread)" for sender in self._senders)
        try:
            listing = await self._call_gmail(
                lambda: client.users()
                .messages()
                .list(userId=self._user_id, q=query, maxResults=max_results)
                .execute()
            )
        except HttpError as exc:
            logger.error("Failed to list Gmail messages", exc_info=exc)
            raise

        message_refs = listing.get("messages", []) if listing else []
        results: list[NewsletterMessage] = []
        for ref in message_refs:
            message_id = ref.get("id")
            if not message_id:
                continue
            metadata = await self._fetch_metadata(client, message_id)
            subject = _extract_header(metadata, "Subject")
            sender_email = _extract_email_address(_extract_header(metadata, "From"))
            internal_date = metadata.get("internalDate")
            received_at = _parse_internal_date(internal_date, metadata)
            html = await self._fetch_html(client, message_id)
            if not html:
                logger.debug("Message %s did not include HTML body", message_id)
            results.append(
                NewsletterMessage(
                    message_id=message_id,
                    subject=subject,
                    received_at=received_at,
                    sender_email=sender_email,
                    html=html,
                )
            )
        return results

    async def fetch_message_html(self, message_id: str) -> str | None:
        """Retrieve the HTML body for the given message id."""

        client = await self._ensure_client()
        return await self._fetch_html(client, message_id)

    async def mark_as_read(self, message_id: str) -> bool:
        """Mark a Gmail message as read.

        Returns True if successful, False otherwise.
        """
        client = await self._ensure_client()
        try:
            await self._call_gmail(
                lambda: client.users()
                .messages()
                .modify(
                    userId=self._user_id,
                    id=message_id,
                    body={"removeLabelIds": ["UNREAD"]}
                )
                .execute()
            )
            return True
        except HttpError as exc:
            logger.warning(
                "Failed to mark message as read",
                extra={"message_id": message_id},
                exc_info=exc,
            )
            return False

    async def _ensure_client(self):
        async with self._client_lock:
            if self._gmail_client is None:
                self._gmail_client = await self._call_gmail(self._build_client)
        return self._gmail_client

    def _build_client(self):
        if build is None:
            raise RuntimeError("google-api-python-client is required for Gmail integration")
        credentials = self._build_credentials(self._credentials_info)
        return build("gmail", "v1", credentials=credentials, cache_discovery=False)

    @staticmethod
    def _build_credentials(source: dict[str, Any] | GoogleCredentials):
        if Credentials is None or Request is None:
            raise RuntimeError("google-auth libraries are required for Gmail integration")

        if isinstance(source, Credentials):
            creds = source
        else:
            creds = Credentials.from_authorized_user_info(source, scopes=GMAIL_SCOPES)
        if not creds.valid and creds.refresh_token:
            creds.refresh(Request())
        return creds

    async def _fetch_metadata(self, client, message_id: str) -> dict[str, Any]:
        return await self._call_gmail(
            lambda: client.users()
            .messages()
            .get(userId=self._user_id, id=message_id, format="metadata")
            .execute()
        )

    async def _fetch_html(self, client, message_id: str) -> str | None:
        message = await self._call_gmail(
            lambda: client.users()
            .messages()
            .get(userId=self._user_id, id=message_id, format="full")
            .execute()
        )
        payload = message.get("payload") if message else None
        if not payload:
            return None
        html = _extract_html_from_payload(payload)
        return html

    async def _call_gmail(self, func: Callable[[], Any]) -> Any:
        if self._gmail_client is not None:
            # A pre-built client (typically for tests) can execute synchronously.
            return func()
        return await asyncio.to_thread(func)


def _extract_header(message: dict[str, Any], header_name: str) -> str | None:
    payload = message.get("payload") if message else None
    if not payload:
        return None
    headers = payload.get("headers", [])
    for header in headers:
        if header.get("name", "").lower() == header_name.lower():
            value = header.get("value")
            return value
    return None


def _extract_email_address(value: str | None) -> str | None:
    if not value:
        return None
    _, email = parseaddr(value)
    if not email or "@" not in email:
        return None
    return email.strip().lower()


def _parse_internal_date(internal_date: Any, message: dict[str, Any]) -> datetime | None:
    if isinstance(internal_date, str) and internal_date.isdigit():
        try:
            timestamp = int(internal_date) / 1000
            return datetime.fromtimestamp(timestamp)
        except ValueError:
            pass
    header_date = _extract_header(message, "Date")
    if header_date:
        try:
            return parsedate_to_datetime(header_date)
        except (TypeError, ValueError):
            return None
    return None


def _extract_html_from_payload(payload: dict[str, Any]) -> str | None:
    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {})
    data = body.get("data")
    if mime_type == "text/html" and data:
        return _decode_websafe_base64(data)

    parts: Iterable[dict[str, Any]] = payload.get("parts", []) or []
    for part in parts:
        result = _extract_html_from_payload(part)
        if result:
            return result
    return None


def _decode_websafe_base64(value: str) -> str:
    padding = '=' * (-len(value) % 4)
    decoded = base64.urlsafe_b64decode(value + padding)
    return decoded.decode("utf-8", errors="replace")


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned or "newsletter"


async def generate_unique_slug(
    candidate_text: str,
    exists: Callable[[str], Awaitable[bool]],
) -> str:
    base = slugify(candidate_text)
    slug = base
    suffix = 2
    while await exists(slug):
        slug = f"{base}-{suffix}"
        suffix += 1
    return slug


__all__ = [
    "GmailNewsletterService",
    "NewsletterMessage",
    "generate_unique_slug",
    "slugify",
]
