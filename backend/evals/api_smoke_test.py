import os
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests


# Configure the base URL for the FastAPI service.
# You can override via environment variable LATER_SERVICE_URL (e.g., http://localhost:8000).
BASE_URL = os.getenv("LATER_SERVICE_URL", "http://localhost:8000")


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def pretty_print_response(resp: requests.Response) -> None:
    print(f"Status: {resp.status_code}")
    try:
        data = resp.json()
        print(json.dumps(data, indent=2, sort_keys=True))
    except Exception:
        print(resp.text)


def post_json(path: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> requests.Response:
    url = f"{BASE_URL}{path}"
    return requests.post(url, json=payload, headers=headers or {}, timeout=30)


def get_query(path: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> requests.Response:
    url = f"{BASE_URL}{path}"
    return requests.get(url, params=params or {}, headers=headers or {}, timeout=30)


# -------------------------
# Auth & User helpers
# -------------------------

def try_create_user(username: str, password: str) -> requests.Response:
    return post_json(
        "/user/add",
        payload={"username": username, "password": password},
    )


def login_get_token(username: str, password: str) -> Tuple[Optional[str], requests.Response]:
    resp = post_json("/auth/login", payload={"username": username, "password": password})
    token = None
    try:
        token = resp.json().get("access_token")
    except Exception:
        token = None
    return token, resp


def auth_headers(token: Optional[str]) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


# -------------------------
# Item endpoints
# -------------------------

def add_item(url: str, token: str) -> requests.Response:
    return post_json("/items/add", payload={"url": url}, headers=auth_headers(token))


def select_items(
    token: str,
    columns: Optional[List[str]] = None,
    filters: Optional[List[str]] = None,
    limit: int = 50,
    offset: int = 0,
    order_by: Optional[str] = "created_at",
    order: str = "desc",
) -> requests.Response:
    params: Dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "order_by": order_by,
        "order": order,
    }
    # FastAPI expects repeated query params for lists
    if columns:
        params["columns"] = columns
    if filters:
        params["filter"] = filters  # alias defined in the API

    return get_query("/items/select", params=params, headers=auth_headers(token))


def update_items(item_ids: List[str], updates: Dict[str, Any], token: str) -> requests.Response:
    return post_json(
        "/items/update",
        payload={"item_ids": item_ids, "updates": updates},
        headers=auth_headers(token),
    )


def delete_items(item_ids: List[str], token: str) -> requests.Response:
    return post_json(
        "/items/delete",
        payload={"item_ids": item_ids},
        headers=auth_headers(token),
    )


# -------------------------
# Clustering endpoints
# -------------------------

def cluster_dimensional_reduction(
    item_ids: List[str],
    token: str,
    mode: str = "umap",
    extra: Optional[Dict[str, Any]] = None,
) -> requests.Response:
    params: Dict[str, Any] = {"item_ids": item_ids, "mode": mode}
    if extra:
        params.update(extra)
    return get_query("/clusters/dimensional-reduction", params=params, headers=auth_headers(token))


def cluster_generate(
    item_ids: List[str],
    token: str,
    mode: str = "kmeans",
    extra: Optional[Dict[str, Any]] = None,
) -> requests.Response:
    if mode == "kmeans":
        params: Dict[str, Any] = {"item_ids": item_ids, "mode": mode, "k": 2}
    elif mode == "hca":
        params: Dict[str, Any] = {"item_ids": item_ids, "mode": mode, "d_threshold": 0.25}
    else:
        params: Dict[str, Any] = {"item_ids": item_ids, "mode": mode}
    if extra:
        params.update(extra)
    return get_query("/clusters/generate", params=params, headers=auth_headers(token))


def cluster_label(
    item_ids: List[str],
    clusters: List[int],
    token: str,
) -> requests.Response:
    import json
    params: Dict[str, Any] = {"item_ids": item_ids, "clusters": json.dumps(clusters)}
    return get_query("/clusters/label", params=params, headers=auth_headers(token))


# -------------------------
# Search endpoint
# -------------------------

def search_items(
    token: str,
    query: str,
    mode: str = "lexical",
    scope: str = "items",
    limit: int = 10,
    columns: Optional[List[str]] = None,
) -> requests.Response:
    params: Dict[str, Any] = {
        "query": query,
        "mode": mode,
        "scope": scope,
        "limit": limit,
    }
    if columns:
        params["columns"] = columns
    return get_query("/items/search", params=params, headers=auth_headers(token))


def main() -> None:
    print_section("Setup test user (create if needed) and login")
    username = "test"
    password = "password"

    # === CREATE USER ===

    # resp_create = try_create_user(username, password)
    # print("POST /user/add")
    # pretty_print_response(resp_create)


    # === AUTH ===

    token, resp_login = login_get_token(username, password)
    print("POST /auth/login")
    pretty_print_response(resp_login)


    # === SEED DATABASE ===

    # print_section("Add a few items")
    # sample_urls = [
    #     "https://www.bbc.co.uk/news/articles/cn832y43ql5o",
    #     "https://www.ibm.com/think/topics/ai-agents",
    #     "https://www.anthropic.com/engineering/building-effective-agents",
    # ]
    # created_item_ids: List[str] = []
    # for url in sample_urls:
    #     resp_add = add_item(url, token)
    #     print(f"POST /items/add url={url}")
    #     pretty_print_response(resp_add)
    #     try:
    #         data = resp_add.json()
    #         if isinstance(data, dict) and "item_id" in data:
    #             created_item_ids.append(str(data["item_id"]))
    #     except Exception:
    #         pass


    # === SELECT ITEMS ===

    # print_section("Select items (default columns, ordered by created_at desc)")
    # resp_select = select_items(
    #     token=token,
    #     columns=["id", "title", "client_status"],
    #     filters=["client_status:IN:saved,queued", "server_status:=:embedded"],
    #     limit=50,
    #     offset=0,
    #     order_by="created_at",
    #     order="desc",
    # )
    # print("GET /items/select")
    # pretty_print_response(resp_select)


    # === UPDATE ITEMS ===

    # print_section("Update first item (if any)")
    # item_ids_for_update = ["9462f91e-1edb-4cc3-bd2a-c17a43f937e3"]
    # if item_ids_for_update:
    #     resp_update = update_items(
    #         item_ids=item_ids_for_update,
    #         updates={"client_status": "queued"},
    #         token=token,
    #     )
    #     print("POST /items/update")
    #     pretty_print_response(resp_update)
    # else:
    #     print("No item_ids collected from add responses; skipping update test.")


    # === SEARCH ===

    # print_section("Search items (lexical and semantic)")
    # resp_search_lex = search_items(token=token, query="government", mode="lexical", scope="chunks", limit=5)
    # print("GET /items/search mode=lexical")
    # pretty_print_response(resp_search_lex)

    # resp_search_sem = search_items(token=token, query="government", mode="semantic", scope="chunks", limit=5)
    # print("GET /items/search mode=semantic")
    # pretty_print_response(resp_search_sem)


    # === CLUSTERING ===

    print_section("Clustering: dimensional reduction, generate clusters, label clusters")
    # The API requires item_ids as query even though it aggregates embedded ones server-side.
    # Provide whatever we have (may be empty if not created).
    item_ids_for_cluster = ["9462f91e-1edb-4cc3-bd2a-c17a43f937e3", "6ff827ab-11f3-4237-bef7-9333d9d7935a", "c4825556-2d2f-49d7-80f0-fc4be4524eb2"]

    # Dimensionality reduction
    for mode in ["pca", "tsne", "umap"]:
        resp_dimred = cluster_dimensional_reduction(item_ids=item_ids_for_cluster, token=token, mode=mode, extra=None)
        print(f"GET /clusters/dimensional-reduction mode={mode}")
        pretty_print_response(resp_dimred)

    # Generate clusters
    for mode in ["kmeans", "hca", "dbscan"]:
        extra: Optional[Dict[str, Any]] = None
        if mode == "dbscan":
            # Example of passing additional parameters; the API uses kwargs.
            extra = {"eps": 0.5, "min_samples": 5}
        resp_cluster = cluster_generate(item_ids=item_ids_for_cluster, token=token, mode=mode, extra=extra)
        print(f"GET /clusters/generate mode={mode}")
        pretty_print_response(resp_cluster)

    # Label clusters (example cluster assignments for the given item_ids)
    # Here we just fabricate a label input vector for demonstration.
    example_clusters = list(range(len(item_ids_for_cluster)))
    resp_label = cluster_label(item_ids=item_ids_for_cluster, clusters=[0, 1, 2], token=token)
    print("GET /clusters/label")
    pretty_print_response(resp_label)

    # print_section("Delete created items (cleanup)")
    # if created_item_ids:
    #     resp_delete = delete_items(item_ids=created_item_ids, token=token)
    #     print("POST /items/delete")
    #     pretty_print_response(resp_delete)
    # else:
    #     print("No created items to delete; skipping delete test.")

    # print("\nAll requests completed (script does not validate outputs).\n")


if __name__ == "__main__":
    # Intentionally not executing anything by default per instructions.
    # You can run main() manually if needed.
    main()
    pass
