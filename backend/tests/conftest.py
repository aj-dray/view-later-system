from __future__ import annotations

"""
Pytest configuration for backend tests.

Provides fixtures for testing the Later System backend:
- Database connection handling
- FastAPI test client
- Authentication helpers
- Service stubs for isolated testing
"""

import os
import sys
import types
import asyncio
from pathlib import Path
from urllib.parse import urlparse, urlunparse
import uuid

import psycopg
from psycopg import sql
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient

# Ensure backend root (containing `app`) is importable
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _select_test_database_url() -> str:
    """Determine the database URL to use for tests.

    Precedence:
      1. TEST_DATABASE_URL env var
      2. fall back to a local `later_system_test` database
    """

    configured = os.environ.get("TEST_DATABASE_URL")
    if configured:
        return configured

    parsed = urlparse(os.environ.get("DATABASE_URL", ""))
    if parsed.scheme:
        # Reuse connection details but force a `_test` database suffix.
        base_db = parsed.path.lstrip("/") or "postgres"
        if not base_db.endswith("_test"):
            base_db = f"{base_db}_test"
        return urlunparse(parsed._replace(path=f"/{base_db}"))

    # Default to local developer credentials.
    return "postgresql://local:dev-password@localhost:5432/later_system_test"


def _ensure_isolated_database(url: str) -> None:
    """Create the test database if necessary and guarantee it's a test DB.

    Raises if the URL points to a non-test database to avoid destructive runs.
    """

    parsed = urlparse(url)
    dbname = parsed.path.lstrip("/") or "postgres"

    if not dbname.endswith("_test"):
        raise RuntimeError(
            "Refusing to run pytest against non-test database "
            f"'{dbname}'. Set TEST_DATABASE_URL to a safe database ending in '_test'."
        )

    admin_url = urlunparse(parsed._replace(path="/postgres"))

    try:
        with psycopg.connect(admin_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (dbname,),
                )
                exists = cur.fetchone() is not None
                if not exists:
                    cur.execute(
                        sql.SQL("CREATE DATABASE {}")
                        .format(sql.Identifier(dbname))
                    )
    except psycopg.OperationalError as exc:
        raise RuntimeError(
            "Unable to ensure test database exists. Check TEST_DATABASE_URL "
            "and database permissions."
        ) from exc


TEST_DATABASE_URL = _select_test_database_url()

# Force the application to use the isolated test database for the entire pytest run.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
_ensure_isolated_database(TEST_DATABASE_URL)

from app.main import app
from app import database as db
from app import services as app_services


# --- Database Fixtures ---

@pytest.fixture
async def database():
    """Initialize database pool for tests and reset data between tests."""
    await db.init_pool()
    
    # Reset database between tests
    async with db.get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "TRUNCATE email_sources, item_chunks, items, llm_usage_logs, users CASCADE"
            )
        await conn.commit()
    
    yield db
    
    await db.close_pool()


# --- Client Fixtures ---

@pytest.fixture
def client():
    """Return a TestClient instance for synchronous tests."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client():
    """Return an AsyncClient instance for async tests."""
    from starlette.testclient import TestClient
    from httpx import ASGITransport
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client


# --- Authentication Fixtures ---

@pytest.fixture
async def user_credentials(client):
    """Create a test user and return credentials."""
    username = f"test_user_{uuid.uuid4().hex[:8]}"
    password = "test_password"
    
    response = client.post(
        "/user/add", 
        json={"username": username, "password": password}
    )
    assert response.status_code == 201
    user_data = response.json()
    
    login_response = client.post(
        "/auth/login", 
        json={"username": username, "password": password}
    )
    assert login_response.status_code == 200
    token_data = login_response.json()
    
    return {
        "user_id": user_data["user_id"],
        "username": username,
        "password": password,
        "token": token_data["access_token"],
        "auth_headers": {"Authorization": f"Bearer {token_data['access_token']}"}
    }


# --- Service Stubs ---

@pytest.fixture
def service_stubs(monkeypatch):
    """Stub out service dependencies for isolated testing."""
    
    async def fake_extract(url: str):
        """Stub for extract_data service."""
        return {
            "canonical_url": url,
            "title": "Test Title",
            "content_text": "This is test content for automated testing.",
            "server_status": "extracted",
        }

    async def fake_generate(item: dict, *, user_id: str):
        """Stub for generate_data service."""
        return {
            "summary": "This is a test summary generated for testing.",
            "server_status": "summarised",
        }

    async def fake_index(item: dict, *, item_id: str, user_id: str):
        """Stub for index_item service."""
        # returns (embed_updates, item_chunks)
        embed_updates = {
            "server_status": "embedded",
            "mistral_embedding": [0.0] * 1024,
        }
        chunks = [
            {
                "content_text": "Test chunk content.", 
                "content_token_count": 4, 
                "mistral_embedding": [0.0] * 1024
            }
        ]
        return embed_updates, chunks

    # Apply stubs
    monkeypatch.setattr(app_services, "extract_data", fake_extract)
    monkeypatch.setattr(app_services, "generate_data", fake_generate)
    monkeypatch.setattr(app_services, "index_item", fake_index)


# --- Library Stubs ---

def _ensure_cohere_stub() -> None:
    """Provide a minimal stub for the `cohere` library if it is unavailable.
    Avoids import-time failures when the real package isn't installed.
    """
    try:
        import cohere
        return
    except Exception:
        pass

    cohere = types.ModuleType("cohere")

    class _RerankResult:
        def __init__(self, index: int, relevance_score: float) -> None:
            self.index = index
            self.relevance_score = float(relevance_score)

    class _RerankResponse:
        def __init__(self, n: int) -> None:
            # Deterministic scores in descending order
            self.results = [
                _RerankResult(i, (n - i) / max(n, 1)) for i in range(n)
            ]

    class Client:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        # Mirror the cohere.Client.rerank signature shape used in code
        def rerank(
            self,
            *,
            query: str,
            documents: list[str],
            model: str,
            top_n: int,
            return_documents: bool = False,
        ) -> _RerankResponse:
            n = min(top_n, len(documents))
            return _RerankResponse(n)

    cohere.Client = Client
    sys.modules["cohere"] = cohere


def _ensure_umap_stub() -> None:
    """Stub the `umap` module when missing to allow importing clustering code.
    Only minimal API is provided for import-time survival; tests don't execute
    UMAP paths, so correctness of results is not required here.
    """
    if "umap" in sys.modules:
        return
    try:
        import umap
        return
    except Exception:
        pass

    umap_mod = types.ModuleType("umap")

    class UMAP:
        def __init__(self, n_components: int = 2, **_: object) -> None:
            self.n_components = n_components

        def fit_transform(self, X: list | tuple) -> list[list[float]]:
            n = len(X) if hasattr(X, "__len__") else 0
            return [[0.0 for _ in range(getattr(self, "n_components", 2))] for _ in range(n)]

    umap_mod.UMAP = UMAP
    sys.modules["umap"] = umap_mod


def _ensure_sklearn_stub() -> None:
    """Provide a very small sklearn stub if it's not installed.
    The clustering module imports several sklearn subpackages. We expose
    placeholder classes with the expected methods so imports succeed.
    """
    try:
        import sklearn
        return
    except Exception:
        pass

    sklearn = types.ModuleType("sklearn")

    # decomposition
    decomposition = types.ModuleType("sklearn.decomposition")
    class PCA:
        def __init__(self, n_components: int = 2, **_: object) -> None:
            self.n_components = n_components
        def fit_transform(self, X):
            return X
    decomposition.PCA = PCA

    # manifold
    manifold = types.ModuleType("sklearn.manifold")
    class TSNE:
        def __init__(self, n_components: int = 2, **_: object) -> None:
            self.n_components = n_components
        def fit_transform(self, X):
            return X
    manifold.TSNE = TSNE

    # cluster
    cluster = types.ModuleType("sklearn.cluster")
    class KMeans:
        def __init__(self, **_: object) -> None:
            pass
        def fit_predict(self, X):
            n = len(X) if hasattr(X, "__len__") else 0
            return [0] * n
    class AgglomerativeClustering:
        def __init__(self, **_: object) -> None:
            pass
        def fit_predict(self, X):
            n = len(X) if hasattr(X, "__len__") else 0
            return [0] * n
    class DBSCAN:
        def __init__(self, **_: object) -> None:
            pass
        def fit_predict(self, X):
            n = len(X) if hasattr(X, "__len__") else 0
            return [-1] * n
    cluster.KMeans = KMeans
    cluster.AgglomerativeClustering = AgglomerativeClustering
    cluster.DBSCAN = DBSCAN

    # neighbors
    neighbors = types.ModuleType("sklearn.neighbors")
    class NearestNeighbors:
        def __init__(self, **_: object) -> None:
            pass
        def fit(self, X):
            return self
        def kneighbors(self, X):
            n = len(X) if hasattr(X, "__len__") else 0
            # return uniform distances and dummy indices
            return [[0.0] * 2 for _ in range(n)], [[0] * 2 for _ in range(n)]
    neighbors.NearestNeighbors = NearestNeighbors

    # Attach submodules
    sklearn.decomposition = decomposition
    sklearn.manifold = manifold
    sklearn.cluster = cluster
    sklearn.neighbors = neighbors

    # Register stubs
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.decomposition"] = decomposition
    sys.modules["sklearn.manifold"] = manifold
    sys.modules["sklearn.cluster"] = cluster
    sys.modules["sklearn.neighbors"] = neighbors


# Apply library stubs
_ensure_cohere_stub()
_ensure_umap_stub()
_ensure_sklearn_stub()

# Ensure an API key is available for tests
os.environ.setdefault("COHERE_API_KEY", "test-key")
