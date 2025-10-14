"""Simplified endpoint tests for Later System using pytest fixtures."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timezone
import uuid


def test_user_add_and_login(client: TestClient):
    """Test user creation and login flow."""
    # Create a user with unique username
    username = f"test_user_{uuid.uuid4().hex[:8]}"
    password = "test_password"
    
    # Register user
    create_resp = client.post(
        "/user/add", 
        json={"username": username, "password": password}
    )
    assert create_resp.status_code == 201, create_resp.text
    user_data = create_resp.json()
    assert "user_id" in user_data, f"Missing user_id in response: {user_data}"
    
    # Login with created user
    login_resp = client.post(
        "/auth/login", 
        json={"username": username, "password": password}
    )
    assert login_resp.status_code == 200, login_resp.text
    token_data = login_resp.json()
    assert "access_token" in token_data, f"Missing access_token in response: {token_data}"
    assert token_data["token_type"] == "bearer"


def test_items_crud_flow(client: TestClient, user_credentials, service_stubs):
    """Test complete item CRUD operations flow with authenticated user."""
    headers = user_credentials["auth_headers"]
    
    # 1. Create an item
    add_resp = client.post(
        "/items/add", 
        json={"url": "https://example.com/test-article"},
        headers=headers
    )
    assert add_resp.status_code == 201, add_resp.text
    item_data = add_resp.json()
    assert "item_id" in item_data, f"Missing item_id in response: {item_data}"
    item_id = item_data["item_id"]
    
    # 2. Select and verify item exists
    select_resp = client.get("/items/select", headers=headers)
    assert select_resp.status_code == 200, select_resp.text
    items = select_resp.json()
    assert any(item.get("id") == item_id for item in items), \
        f"Created item {item_id} not found in items list: {items}"
    
    # 3. Update item
    update_resp = client.post(
        "/items/update",
        headers=headers,
        json={
            "item_ids": [item_id],
            "updates": {"client_status": "completed"}
        }
    )
    assert update_resp.status_code == 200, update_resp.text
    update_result = update_resp.json()["results"][item_id]
    assert update_result["updated"] is True, f"Update failed: {update_result}"
    
    # 4. Verify update with filtered select
    filter_resp = client.get(
        "/items/select",
        headers=headers,
        params=[("filter", f"id:=:{item_id}"), ("columns", "client_status")]
    )
    assert filter_resp.status_code == 200, filter_resp.text
    filtered_items = filter_resp.json()
    assert filtered_items and filtered_items[0]["client_status"] == "completed", \
        f"Item update not reflected: {filtered_items}"
    
    # 5. Search items
    search_resp = client.get(
        "/items/search",
        headers=headers,
        params={
            "query": "test",
            "mode": "lexical",
            "scope": "items",
            "limit": 5
        }
    )
    assert search_resp.status_code == 200, search_resp.text
    
    # 6. Delete item
    delete_resp = client.post(
        "/items/delete",
        headers=headers,
        json={"item_ids": [item_id]}
    )
    assert delete_resp.status_code == 200, delete_resp.text
    delete_result = delete_resp.json()["results"]
    assert delete_result[item_id] is True, f"Delete operation failed: {delete_result}"
    
    # 7. Verify deletion
    final_resp = client.get("/items/select", headers=headers)
    assert final_resp.status_code == 200, final_resp.text
    final_items = final_resp.json()
    assert not any(item.get("id") == item_id for item in final_items), \
        f"Item still exists after deletion: {final_items}"


def test_add_item_rejects_empty_url(client: TestClient, user_credentials, service_stubs):
    headers = user_credentials["auth_headers"]

    response = client.post(
        "/items/add",
        json={"url": "   "},
        headers=headers,
    )

    assert response.status_code == 400, response.text
    assert response.json().get("detail") == "URL is required"


def test_add_item_rejects_invalid_url(client: TestClient, user_credentials, service_stubs):
    headers = user_credentials["auth_headers"]

    response = client.post(
        "/items/add",
        json={"url": "notaurl"},
        headers=headers,
    )

    assert response.status_code == 400, response.text
    assert response.json().get("detail") == "Please enter a valid URL with a proper domain"


def test_unauthorized_access(client: TestClient):
    """Test that unauthorized access is properly rejected."""
    # Try to access protected endpoint without authentication
    resp = client.get("/items/select")
    assert resp.status_code == 401, resp.text
    assert "detail" in resp.json(), "Error response missing detail"
    
    # Try with invalid token
    resp = client.get(
        "/items/select",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert resp.status_code == 401, resp.text


def test_user_settings(client: TestClient, user_credentials):
    """Test user settings endpoints."""
    headers = user_credentials["auth_headers"]
    
    setting_type = "preference"
    setting_key = "theme"
    
    # Set a user setting using PUT
    put_resp = client.put(
        f"/user/settings/{setting_type}/{setting_key}",
        json={"value": "dark"},
        headers=headers
    )
    assert put_resp.status_code in (200, 201), put_resp.text
    
    # Get the setting
    get_resp = client.get(
        f"/user/settings/{setting_type}/{setting_key}",
        headers=headers
    )
    assert get_resp.status_code == 200, get_resp.text
    setting = get_resp.json()
    assert setting.get("setting_value", {}).get("value") == "dark", f"Setting value mismatch: {setting}"
    
    # Get settings by type
    type_resp = client.get(
        f"/user/settings/{setting_type}",
        headers=headers
    )
    assert type_resp.status_code == 200, type_resp.text
    type_settings = type_resp.json()
    assert isinstance(type_settings, dict), "Expected dict of settings"
    assert "settings" in type_settings, f"Missing settings key in response: {type_settings}"
    assert setting_key in type_settings["settings"], \
        f"Setting key '{setting_key}' not found in response: {type_settings}"
    
    # Update setting with PATCH
    patch_resp = client.patch(
        f"/user/settings/{setting_type}/{setting_key}",
        json={"field_key": "value", "field_value": "light"},
        headers=headers
    )
    assert patch_resp.status_code == 200, patch_resp.text
    
    # Verify update
    verify_resp = client.get(
        f"/user/settings/{setting_type}/{setting_key}",
        headers=headers
    )
    updated_setting = verify_resp.json()
    assert updated_setting.get("setting_value", {}).get("value") == "light", \
        f"Setting not updated correctly: {updated_setting}"
