from __future__ import annotations

import base64
from datetime import datetime
from types import SimpleNamespace

import pytest

from app import database as db
from app import main
from app import services as app_services
from app.services import newsletters


@pytest.mark.asyncio
async def test_generate_unique_slug_increments(monkeypatch):
    existing: set[str] = {"newsletter", "newsletter-2"}

    async def exists(slug: str) -> bool:
        return slug in existing

    slug = await newsletters.generate_unique_slug("Newsletter!!!", exists)
    assert slug == "newsletter-3"


@pytest.mark.asyncio
async def test_gmail_service_returns_html_with_view_link():
    html_body = "<html><body><a href='https://example.com/article'>View in Browser</a></body></html>"
    encoded_html = base64.urlsafe_b64encode(html_body.encode("utf-8")).decode("ascii")

    message_record = {
        "id": "msg-1",
        "metadata": {
            "internalDate": "1700000000000",
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Weekly Update"},
                    {"name": "From", "value": "Sender <newsletter@example.com>"},
                ],
                "body": {},
            },
        },
        "full": {
            "payload": {
                "mimeType": "multipart/alternative",
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": ""}},
                    {"mimeType": "text/html", "body": {"data": encoded_html}},
                ],
            }
        },
    }

    class FakeMessages:
        def __init__(self, messages: list[dict]):
            self._messages = messages

        def list(self, userId: str, q: str, maxResults=None):  # noqa: N803 - mimic API
            return SimpleNamespace(execute=lambda: {"messages": [{"id": m["id"]} for m in self._messages]})

        def get(self, userId: str, id: str, format: str = "metadata"):  # noqa: A002,N803 - API compatibility
            def execute():
                record = next(m for m in self._messages if m["id"] == id)
                if format == "metadata":
                    return record["metadata"]
                return record["full"]

            return SimpleNamespace(execute=execute)

    class FakeUsers:
        def __init__(self, messages: list[dict]):
            self.messages = FakeMessages(messages)

    class FakeGmail:
        def __init__(self, messages: list[dict]):
            self._messages = messages

        def users(self):
            return FakeUsers(self._messages)

    fake_client = FakeGmail([message_record])
    service = newsletters.GmailNewsletterService(
        credentials_info={},
        senders=["sender@example.com"],
        user_id="me",
        gmail_client=fake_client,
    )

    messages = await service.poll_newsletters()
    assert len(messages) == 1
    message = messages[0]
    assert message.html == html_body
    assert message.subject == "Weekly Update"
    assert message.sender_email == "newsletter@example.com"


class DummyNewsletterService:
    def __init__(self, messages: list[newsletters.NewsletterMessage], html_lookup: dict[str, str] | None = None):
        self.messages = messages
        self.html_lookup = html_lookup or {}
        self.poll_calls = 0

    async def poll_newsletters(self, *, max_results: int | None = None):
        self.poll_calls += 1
        return self.messages[: max_results] if max_results else list(self.messages)

    async def fetch_message_html(self, message_id: str) -> str | None:
        return self.html_lookup.get(message_id)


@pytest.mark.asyncio
async def test_gmail_poll_creates_items_and_runs_pipeline(async_client, user_credentials, monkeypatch):
    external_html = "<html><body><h1>External Story</h1></body></html>"
    external = newsletters.NewsletterMessage(
        message_id="message-external",
        subject="External",
        received_at=datetime.utcnow(),
        sender_email="external@example.com",
        html=external_html,
    )
    inline_html = "<html><body><h1>Inline</h1></body></html>"
    inline = newsletters.NewsletterMessage(
        message_id="message-inline",
        subject="Inline",
        received_at=datetime.utcnow(),
        sender_email="news@updates.test",
        html=inline_html,
    )

    dummy_service = DummyNewsletterService([external, inline])

    async def fake_get_context(user_id: str, *, require_senders: bool = False):
        return ({"token": "test"}, ["sender@example.com"], "sender@example.com")

    def fake_create_newsletter_service(*, credentials_info, senders, user_id=None):
        return dummy_service

    monkeypatch.setattr(main, "_get_user_gmail_context", fake_get_context)
    monkeypatch.setattr(main, "_create_newsletter_service", fake_create_newsletter_service)

    scheduled: list[tuple[str, str, str]] = []

    def record_schedule(item_id: str, *, url: str, user_id: str) -> None:
        scheduled.append((item_id, url, user_id))

    monkeypatch.setattr(main, "_schedule_pipeline", record_schedule)

    response = await async_client.post(
        "/gmail/poll",
        headers=user_credentials["auth_headers"],
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["imported"]) == 2
    assert payload["duplicates"] == []
    assert payload["skipped"] == []

    # Ensure items exist and pipelines scheduled
    assert len(scheduled) == 2
    for _, url, _ in scheduled:
        assert url.startswith(main.BACKEND_PUBLIC_URL)

    # Inline HTML should be cached in the database
    inline_record = await db.get_email_source_by_message_id("message-inline")
    assert inline_record is not None
    assert inline_record["html_content"].strip() == inline_html
    external_record = await db.get_email_source_by_message_id("message-external")
    assert external_record is not None
    assert external_record["html_content"].strip() == external_html

    imported_ids = [entry["item_id"] for entry in payload["imported"]]
    stored_items = {
        item_id: await db.get_item(
            item_id,
            ["title", "format", "favicon_url"],
            user_credentials["user_id"],
        )
        for item_id in imported_ids
    }
    expected_favicons = {
        "External": "https://www.google.com/s2/favicons?sz=64&domain=example.com",
        "Inline": "https://www.google.com/s2/favicons?sz=64&domain=updates.test",
    }
    for item in stored_items.values():
        assert item is not None
        assert item["format"] == "newsletter"
        assert item["favicon_url"] == expected_favicons[item["title"]]

    # Second poll should mark messages as duplicates
    second = await async_client.post(
        "/gmail/poll",
        headers=user_credentials["auth_headers"],
    )
    assert second.status_code == 200
    assert set(second.json()["duplicates"]) == {"message-external", "message-inline"}


@pytest.mark.asyncio
async def test_newsletter_pipeline_preserves_subject_title(database, user_credentials, monkeypatch):
    subject = "Weekly Roundup"
    saved_at = datetime.utcnow()
    item_payload = {
        "url": main._build_proxy_url("weekly-roundup"),
        "title": subject,
        "format": "newsletter",
        "favicon_url": "https://www.google.com/s2/favicons?sz=64&domain=newsletter.example",
        "client_status": "adding",
        "client_status_at": saved_at,
        "server_status": "saved",
        "server_status_at": saved_at,
        "user_id": user_credentials["user_id"],
    }
    item = await db.create_item(item_payload)
    await db.create_email_source(
        item_id=item["id"],
        message_id="message-weekly",
        slug="weekly-roundup",
        title=subject,
        resolved_url=None,
        html_content="<html><body><h1>Weekly Roundup</h1></body></html>",
    )

    async def fake_extract(url: str):
        return {
            "canonical_url": url,
            "title": "Metadata Newsletter Title",
            "format": "webpage",
            "content_text": "Body content",
            "server_status": "extracted",
        }

    async def fake_generate(item_data: dict[str, object], *, user_id: str):
        return {
            "title": "Generated Summary Title",
            "type": "article",
            "summary": "Summary text",
            "server_status": "summarised",
        }

    async def fake_index(item_data: dict[str, object], *, item_id: str, user_id: str):
        embed_updates = {
            "server_status": "embedded",
            "mistral_embedding": [0.0] * 5,
        }
        return embed_updates, []

    monkeypatch.setattr(app_services, "extract_data", fake_extract)
    monkeypatch.setattr(app_services, "generate_data", fake_generate)
    monkeypatch.setattr(app_services, "index_item", fake_index)

    await main._process_item_pipeline(item_id=item["id"], url=item["url"], user_id=user_credentials["user_id"])

    updated = await db.get_item(
        item["id"],
        ["title", "format", "type", "favicon_url"],
        user_id=user_credentials["user_id"],
    )
    assert updated is not None
    assert updated["title"] == subject
    assert updated["format"] == "newsletter"
    assert updated["type"] == "newsletter"
    assert updated["favicon_url"] == "https://www.google.com/s2/favicons?sz=64&domain=newsletter.example"


@pytest.mark.asyncio
async def test_view_newsletter_html_serves_cached_content(async_client, user_credentials, monkeypatch):
    saved_at = datetime.utcnow()
    base_item = {
        "url": main._build_proxy_url("cached-newsletter"),
        "client_status": "adding",
        "client_status_at": saved_at,
        "server_status": "saved",
        "server_status_at": saved_at,
        "user_id": user_credentials["user_id"],
    }
    item = await db.create_item(base_item)
    await db.create_email_source(
        item_id=item["id"],
        message_id="cached-message",
        slug="cached-newsletter",
        title="Cached",
        resolved_url=None,
        html_content="<html><body>Cached</body></html>",
    )

    response = await async_client.get(f"/items/view/cached-newsletter")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Cached" in response.text


@pytest.mark.asyncio
async def test_view_newsletter_html_fetches_when_missing(async_client, user_credentials, monkeypatch):
    slug = "fresh-newsletter"
    saved_at = datetime.utcnow()
    item = await db.create_item(
        {
            "url": main._build_proxy_url(slug),
            "client_status": "adding",
            "client_status_at": saved_at,
            "server_status": "saved",
            "server_status_at": saved_at,
            "user_id": user_credentials["user_id"],
        }
    )
    await db.create_email_source(
        item_id=item["id"],
        message_id="fresh-message",
        slug=slug,
        title="Fresh",
        resolved_url=None,
        html_content=None,
    )

    dummy_service = DummyNewsletterService([], html_lookup={"fresh-message": "<html><body>Fresh</body></html>"})
    async def fake_get_context(user_id: str, *, require_senders: bool = False):
        return ({"token": "test"}, ["sender@example.com"], "sender@example.com")

    def fake_create_newsletter_service(*, credentials_info, senders, user_id=None):
        return dummy_service

    monkeypatch.setattr(main, "_get_user_gmail_context", fake_get_context)
    monkeypatch.setattr(main, "_create_newsletter_service", fake_create_newsletter_service)

    response = await async_client.get(f"/items/view/{slug}")
    assert response.status_code == 200
    assert "Fresh" in response.text

    refreshed = await db.get_email_source_by_message_id("fresh-message")
    assert refreshed is not None
    assert "Fresh" in refreshed["html_content"]


@pytest.mark.asyncio
async def test_get_gmail_settings_reflects_stored_credentials(async_client, user_credentials):
    user_id = user_credentials["user_id"]
    existing_senders = await db.list_gmail_senders(user_id)
    for entry in existing_senders:
        await db.remove_gmail_sender(user_id=user_id, sender_id=entry["id"])
    await db.delete_gmail_credentials(user_id)
    await db.upsert_gmail_credentials(
        user_id=user_id,
        credentials={
            "token": "ya29.test-token",
            "refresh_token": "refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scopes": newsletters.GMAIL_SCOPES,
        },
        email_address="user@example.com",
        token_expiry=datetime.utcnow(),
    )
    await db.add_gmail_sender(user_id=user_id, email_address="updates@example.com")

    response = await async_client.get(
        "/gmail/settings",
        headers=user_credentials["auth_headers"],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["connected"] is True
    assert payload["email_address"] == "user@example.com"
    assert payload["has_senders"] is True
    assert any(sender["email_address"] == "updates@example.com" for sender in payload["senders"])
    await db.delete_gmail_credentials(user_id)
    remaining_senders = await db.list_gmail_senders(user_id)
    for entry in remaining_senders:
        await db.remove_gmail_sender(user_id=user_id, sender_id=entry["id"])


@pytest.mark.asyncio
async def test_create_and_delete_sender_endpoint(async_client, user_credentials):
    user_id = user_credentials["user_id"]
    existing = await db.list_gmail_senders(user_id)
    for entry in existing:
        await db.remove_gmail_sender(user_id=user_id, sender_id=entry["id"])
    create = await async_client.post(
        "/gmail/senders",
        json={"email_address": "Newsletter@Example.com"},
        headers=user_credentials["auth_headers"],
    )
    assert create.status_code == 200
    payload = create.json()
    sender = payload["sender"]
    assert sender["email_address"] == "newsletter@example.com"
    sender_id = sender["id"]
    assert sender_id

    # Sender exists in database
    senders = await db.list_gmail_senders(user_id)
    assert any(entry["id"] == sender_id for entry in senders)

    delete = await async_client.delete(
        f"/gmail/senders/{sender_id}",
        headers=user_credentials["auth_headers"],
    )
    assert delete.status_code == 200
    assert delete.json()["deleted"] is True

    remaining = await db.list_gmail_senders(user_id)
    assert all(entry["id"] != sender_id for entry in remaining)


@pytest.mark.asyncio
async def test_gmail_auth_start_requires_configuration(async_client, user_credentials, monkeypatch):
    monkeypatch.setattr(main, "GOOGLE_OAUTH_CLIENT_CONFIG", None)
    response = await async_client.post(
        "/gmail/auth/start",
        headers=user_credentials["auth_headers"],
    )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_gmail_auth_disconnect_removes_credentials(async_client, user_credentials):
    user_id = user_credentials["user_id"]
    await db.delete_gmail_credentials(user_id)
    await db.upsert_gmail_credentials(
        user_id=user_id,
        credentials={
            "token": "another-token",
            "refresh_token": "another-refresh",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scopes": newsletters.GMAIL_SCOPES,
        },
        email_address="user2@example.com",
        token_expiry=datetime.utcnow(),
    )

    disconnect = await async_client.post(
        "/gmail/auth/disconnect",
        headers=user_credentials["auth_headers"],
    )
    assert disconnect.status_code == 200
    assert disconnect.json()["disconnected"] is True

    record = await db.get_gmail_credentials(user_id)
    assert record is None
