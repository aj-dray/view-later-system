"""API endpoint tests for Later System backend using pytest-asyncio."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import uuid

import jwt
import pytest
from httpx import AsyncClient

from app import auth


@pytest.mark.asyncio
async def test_authentication_required(async_client: AsyncClient):
    """Test that authentication is required for protected endpoints."""
    response = await async_client.get("/items/select")
    assert response.status_code == 401, response.text


@pytest.mark.asyncio
async def test_user_registration_and_login(async_client: AsyncClient, database):
    """Test user registration and login flow."""
    # Generate unique username for test
    username = f"user_{uuid.uuid4().hex[:12]}"
    password = "testpass123"
    
    # Register user
    response = await async_client.post(
        "/user/add", json={"username": username, "password": password}
    )
    assert response.status_code == 201, response.text
    user_payload = response.json()
    user_id = user_payload.get("user_id")
    assert user_id, f"Missing user_id in response: {user_payload}"
    
    # Login with created user
    login_response = await async_client.post(
        "/auth/login", json={"username": username, "password": password}
    )
    assert login_response.status_code == 200, login_response.text
    token_payload = login_response.json()
    token = token_payload.get("access_token")
    assert token, f"Missing access_token in response: {token_payload}"


@pytest.mark.asyncio
async def test_user_access_token_endpoint(async_client: AsyncClient, database):
    """Issue access token through new endpoint."""
    username = f"user_{uuid.uuid4().hex[:12]}"
    password = "testpass123"

    reg_response = await async_client.post(
        "/user/add", json={"username": username, "password": password}
    )
    assert reg_response.status_code == 201, reg_response.text

    token_response = await async_client.post(
        "/user/access-token",
        json={"username": username, "password": password},
    )
    assert token_response.status_code == 200, token_response.text
    payload = token_response.json()
    assert payload.get("token_type") == "bearer"
    assert isinstance(payload.get("access_token"), str) and payload["access_token"]
    decoded = jwt.decode(
        payload["access_token"],
        auth._get_secret(),
        algorithms=[auth.JWT_ALGORITHM],
        options={"verify_exp": False},
    )
    assert "exp" not in decoded
    assert decoded.get("jti"), "Expected token to include a jti claim"


@pytest.mark.asyncio
async def test_user_access_token_endpoint_with_expiry(async_client: AsyncClient, database):
    """Issue access token with custom expiry."""
    username = f"user_{uuid.uuid4().hex[:12]}"
    password = "testpass123"

    reg_response = await async_client.post(
        "/user/add", json={"username": username, "password": password}
    )
    assert reg_response.status_code == 201, reg_response.text

    token_response = await async_client.post(
        "/user/access-token",
        json={"username": username, "password": password, "expires_in_hours": 2},
    )
    assert token_response.status_code == 200, token_response.text
    payload = token_response.json()
    decoded = jwt.decode(
        payload["access_token"],
        auth._get_secret(),
        algorithms=[auth.JWT_ALGORITHM],
        options={"verify_exp": False},
    )
    assert decoded.get("exp"), "Expected exp field when expires_in_hours specified"
    assert decoded.get("jti"), "Expected token to include a jti claim"


@pytest.mark.asyncio
async def test_list_and_delete_access_tokens(async_client: AsyncClient, database):
    """List issued tokens and revoke one."""
    username = f"user_{uuid.uuid4().hex[:12]}"
    password = "testpass123"

    reg_response = await async_client.post(
        "/user/add", json={"username": username, "password": password}
    )
    assert reg_response.status_code == 201, reg_response.text

    token_response = await async_client.post(
        "/user/access-token", json={"username": username, "password": password}
    )
    assert token_response.status_code == 200, token_response.text
    access_token = token_response.json()["access_token"]
    decoded = jwt.decode(
        access_token,
        auth._get_secret(),
        algorithms=[auth.JWT_ALGORITHM],
        options={"verify_exp": False},
    )
    token_id = decoded.get("jti")
    assert token_id, "Expected issued token to include jti"

    login_response = await async_client.post(
        "/auth/login", json={"username": username, "password": password}
    )
    assert login_response.status_code == 200, login_response.text
    session_token = login_response.json()["access_token"]
    session_headers = {"Authorization": f"Bearer {session_token}"}

    list_response = await async_client.get("/user/access-token", headers=session_headers)
    assert list_response.status_code == 200, list_response.text
    token_rows = list_response.json().get("tokens", [])
    assert any(row.get("token_id") == token_id for row in token_rows)

    api_headers = {"Authorization": f"Bearer {access_token}"}
    initial_access = await async_client.get("/items/select", headers=api_headers)
    assert initial_access.status_code != 401, initial_access.text

    delete_response = await async_client.delete(
        f"/user/access-token/{token_id}", headers=session_headers
    )
    assert delete_response.status_code == 204, delete_response.text

    post_delete_list = await async_client.get("/user/access-token", headers=session_headers)
    assert post_delete_list.status_code == 200, post_delete_list.text
    updated_rows = post_delete_list.json().get("tokens", [])
    revoked_row = next((row for row in updated_rows if row.get("token_id") == token_id), None)
    assert revoked_row is not None and revoked_row.get("revoked_at"), "Expected revoked token to be flagged"

    revoked_access = await async_client.get("/items/select", headers=api_headers)
    assert revoked_access.status_code == 401, revoked_access.text


@pytest.mark.asyncio
async def test_user_access_token_endpoint_invalid_password(async_client: AsyncClient, database):
    """Reject token issuance with invalid credentials."""
    username = f"user_{uuid.uuid4().hex[:12]}"
    password = "testpass123"

    reg_response = await async_client.post(
        "/user/add", json={"username": username, "password": password}
    )
    assert reg_response.status_code == 201, reg_response.text

    token_response = await async_client.post(
        "/user/access-token",
        json={"username": username, "password": "wrongpass"},
    )
    assert token_response.status_code == 401, token_response.text


@pytest.mark.asyncio
async def test_items_crud_flow(async_client: AsyncClient, database):
    """Test complete item CRUD operations flow."""
    # Setup user
    username = f"user_{uuid.uuid4().hex[:12]}"
    password = "testpass123"
    
    # Register user
    reg_response = await async_client.post(
        "/user/add", json={"username": username, "password": password}
    )
    assert reg_response.status_code == 201
    user_id = reg_response.json().get("user_id")
    
    # Login
    login_response = await async_client.post(
        "/auth/login", json={"username": username, "password": password}
    )
    assert login_response.status_code == 200
    token = login_response.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create item
    now = datetime.now(timezone.utc)
    item_url = f"https://example.com/{uuid.uuid4().hex}"
    item_payload = {
        "url": item_url,
        "client_status": "adding",
        "client_status_at": now.isoformat(),
        "server_status": "saved",
        "server_status_at": now.isoformat(),
        "user_id": user_id,
        "title": "Testing article",
        "content_text": "This content is used for verifying item flows.",
    }
    
    created_item = await database.create_item(item_payload)
    item_id = str(created_item.get("id"))
    assert item_id, f"Failed to create item with payload: {item_payload}"
    
    # Select items
    select_response = await async_client.get("/items/select", headers=headers)
    assert select_response.status_code == 200, select_response.text
    items = select_response.json()
    assert any(row.get("id") == item_id for row in items), f"Created item {item_id} not in items list"
    
    # Update item
    update_response = await async_client.post(
        "/items/update",
        headers=headers,
        json={
            "item_ids": [item_id],
            "updates": {"client_status": "completed"},
        },
    )
    assert update_response.status_code == 200, update_response.text
    update_payload = update_response.json()["results"][item_id]
    assert update_payload["updated"] is True, f"Update failed: {update_payload}"
    
    # Verify update with filtered select
    filtered_response = await async_client.get(
        "/items/select",
        headers=headers,
        params=[("filter", f"id:=:{item_id}"), ("columns", "id"), ("columns", "client_status")],
    )
    assert filtered_response.status_code == 200, filtered_response.text
    filtered_items = filtered_response.json()
    assert filtered_items and filtered_items[0]["client_status"] == "completed", \
           f"Item update not reflected: {filtered_items}"
    
    # Delete item
    delete_response = await async_client.post(
        "/items/delete",
        headers=headers,
        json={"item_ids": [item_id]},
    )
    assert delete_response.status_code == 200, delete_response.text
    delete_payload = delete_response.json()["results"]
    assert delete_payload[item_id] is True, f"Delete operation failed: {delete_payload}"
    
    # Verify deletion
    final_response = await async_client.get("/items/select", headers=headers)
    assert final_response.status_code == 200, final_response.text
    assert not any(item.get("id") == item_id for item in final_response.json()), \
           "Item still exists after deletion"


@pytest.mark.asyncio
async def test_items_search_lexical(async_client: AsyncClient, database):
    """Test lexical search functionality."""
    # Setup user
    username = f"user_{uuid.uuid4().hex[:12]}"
    password = "testpass123"
    
    # Register and login
    reg_response = await async_client.post(
        "/user/add", json={"username": username, "password": password}
    )
    user_id = reg_response.json().get("user_id")
    
    login_response = await async_client.post(
        "/auth/login", json={"username": username, "password": password}
    )
    token = login_response.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create test item
    now = datetime.now(timezone.utc)
    item_payload = {
        "url": f"https://example.com/{uuid.uuid4().hex}",
        "client_status": "adding",
        "client_status_at": now.isoformat(),
        "server_status": "saved",
        "server_status_at": now.isoformat(),
        "user_id": user_id,
        "title": "Python Testing",
        "content_text": "Python testing strategies and fixtures are useful.",
        "summary": "Notes on Python testing.",
    }
    
    created_item = await database.create_item(item_payload)
    item_id = str(created_item.get("id"))
    assert item_id
    
    # Perform lexical search
    search_response = await async_client.get(
        "/items/search",
        headers=headers,
        params={
            "query": "testing",  # lexical search should find the content text
            "mode": "lexical",
            "scope": "items",
            "limit": 5,
        },
    )
    assert search_response.status_code == 200, search_response.text
    results = search_response.json()["results"]
    assert results, "Expected at least one search result"
    assert any(row.get("id") == item_id for row in results), \
           f"Search didn't return expected item {item_id}"
