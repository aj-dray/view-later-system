"""Database connectivity and schema management for the Later service."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import (Any, Iterator, Sequence)
import uuid
from aglib import Response  # type: ignore[attr-defined]
from psycopg import errors, sql, OperationalError, InterfaceError
from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from datetime import datetime




from . import schemas
from . import utils as app_utils
from . import auth
from . import pricing


# === VARIABLES ===


# Embedding
EMBEDDING_DIM = 1024
# Chunking
CHUNK_INSERT_BATCH_SIZE = 16
CHUNK_INSERT_MAX_RETRIES = 1

ITEM_SEARCH_DEFAULT_COLUMNS: tuple[str, ...] = ("id", "title", "summary")

CHUNK_COLUMN_SOURCES: dict[str, str] = {
    "id": "c.id",
    "item_id": "c.item_id",
    "position": "c.position",
    "content_text": "c.content_text",
    "content_token_count": "c.content_token_count",
    "created_at": "c.created_at",
    "title": "i.title",
    "summary": "i.summary",
    "url": "i.url",
    "canonical_url": "i.canonical_url",
    "source_site": "i.source_site",
}

CHUNK_SEARCH_DEFAULT_COLUMNS: tuple[str, ...] = (
    "id",
    "item_id",
    "position",
    "content_text",
    "title",
)


# === CONFIGURATION ===


POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")
POSTGRES_USER = os.getenv("POSTGRES_USER", "local")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "dev-password")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = os.getenv(
    "DATABASE_URL", # if this set, use directly
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
pool: AsyncConnectionPool | None = None


async def init_pool():
    global pool
    if pool is None:
        pool = AsyncConnectionPool(
            conninfo=DATABASE_URL,
            min_size=1, max_size=20,
            kwargs={"row_factory": dict_row,
                "prepare_threshold": None},
            timeout=10, max_lifetime=1800, max_idle=300,
            open=False
        )
        await pool.open()


async def close_pool():
    global pool
    if pool:
        await pool.close()
        pool = None


@asynccontextmanager
async def get_connection():
    assert pool is not None
    async with pool.connection() as conn:
        yield conn


# === INITIALISATION ===


async def init_database() -> None:
    """Create required extensions, types, tables and indexes if they do not exist."""
    sql_statments: list[str] = schemas.get_create_sql()

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            for statement in sql_statments:
                await cur.execute(statement)  # type: ignore
        await conn.commit()


# === USERS ===


async def create_user(username: str, password: str) -> int:
    """Create a user row, returning the generated id."""
    password_hash = auth.hash_password(password)

    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            try:
                await cur.execute(
                    """
                    INSERT INTO users (username, password_hash)
                    VALUES (%(username)s, %(password_hash)s)
                    RETURNING id
                    """,
                    {"username": username, "password_hash": password_hash},
                )
                row = await cur.fetchone()
                if not row:
                    raise RuntimeError("No row inserted.")
            except Exception as e:
                await conn.rollback()
                if isinstance(e, errors.UniqueViolation):
                    raise ValueError("Username already exists") from e
                else:
                    raise RuntimeError(f"Failed to create user: {e}")
            else:
                await conn.commit()

    return row["id"]


async def authenticate_user(username: str, password: str) -> dict[str, str] | None:
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, username, password_hash
                FROM users
                WHERE username = %(username)s
                """,
                {"username": username},
            )
            row: dict[str, Any] | None = await cur.fetchone()

    if not row:
        return None
    ph = row.get("password_hash")
    if not isinstance(ph, str) or not auth.verify_password(password, ph):
        return None
    return {"user_id": str(row["id"]), "username": str(row["username"])}


async def get_user_by_username(username: str) -> dict[str, Any] | None:
    """Fetch a user row by username."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, username, password_hash, created_at
                FROM users
                WHERE username = %(username)s
                """,
                {"username": username},
            )
            row: dict[str, Any] | None = await cur.fetchone()
    return _normalise_row(row) if row else None


async def update_user_password(*, user_id: str, new_password: str) -> None:
    """Set a new password for the given user id."""
    password_hash = auth.hash_password(new_password)
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE users
                SET password_hash = %(password_hash)s
                WHERE id = %(user_id)s
                """,
                {"password_hash": password_hash, "user_id": user_id},
            )


async def create_user_access_token(
    *,
    user_id: str,
    token_id: str,
    expires_at: datetime | None,
    label: str | None = None,
) -> dict[str, Any]:
    """Persist metadata for an issued API access token and return the stored row."""
    token_uuid = uuid.UUID(token_id)
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO user_access_tokens (user_id, token_id, label, expires_at)
                VALUES (%(user_id)s, %(token_id)s, %(label)s, %(expires_at)s)
                RETURNING token_id, label, expires_at, created_at
                """,
                {
                    "user_id": user_id,
                    "token_id": token_uuid,
                    "label": label,
                    "expires_at": expires_at,
                },
            )
            row = await cur.fetchone()
            if row is None:
                raise RuntimeError("Failed to insert access token")
        await conn.commit()
    return _normalise_row(row)


async def list_user_access_tokens(user_id: str) -> list[dict[str, Any]]:
    """Return all API tokens for a user ordered by creation time."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT token_id, label, expires_at, created_at, revoked_at
                FROM user_access_tokens
                WHERE user_id = %(user_id)s
                ORDER BY created_at DESC
                """,
                {"user_id": user_id},
            )
            rows = await cur.fetchall()
    return [_normalise_row(row) for row in rows]


async def revoke_user_access_token(*, user_id: str, token_id: str) -> bool:
    """Revoke a previously issued API token."""
    token_uuid = uuid.UUID(token_id)
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE user_access_tokens
                SET revoked_at = NOW()
                WHERE user_id = %(user_id)s
                  AND token_id = %(token_id)s
                  AND revoked_at IS NULL
                """,
                {"user_id": user_id, "token_id": token_uuid},
            )
            updated = cur.rowcount
        await conn.commit()
    return bool(updated)


async def get_user_access_token(token_id: str) -> dict[str, Any] | None:
    """Fetch token metadata regardless of revocation state."""
    token_uuid = uuid.UUID(token_id)
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT user_id, token_id, label, expires_at, created_at, revoked_at
                FROM user_access_tokens
                WHERE token_id = %(token_id)s
                """,
                {"token_id": token_uuid},
            )
            row = await cur.fetchone()
    return _normalise_row(row) if row else None


async def clone_user_data(*, source_user_id: str, target_user_id: str) -> dict[str, int]:
    """
    Clone data from one user to another.

    Copies:
    - items
    - item_chunks (via URL match between source and target items)
    - user_settings (all types/keys)

    Returns a summary dict with counts of copied rows.
    """
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Copy items
            await cur.execute(
                """
                INSERT INTO items (
                    user_id,
                    url,
                    canonical_url,
                    title,
                    source_site,
                    format,
                    author,
                    type,
                    publication_date,
                    favicon_url,
                    content_markdown,
                    content_text,
                    duration,
                    content_token_count,
                    client_status,
                    server_status,
                    summary,
                    expiry_score,
                    mistral_embedding,
                    client_status_at,
                    server_status_at,
                    created_at
                )
                SELECT
                    %(target_user_id)s,
                    url,
                    canonical_url,
                    title,
                    source_site,
                    format,
                    author,
                    type,
                    publication_date,
                    favicon_url,
                    content_markdown,
                    content_text,
                    duration,
                    content_token_count,
                    client_status,
                    server_status,
                    summary,
                    expiry_score,
                    mistral_embedding,
                    client_status_at,
                    server_status_at,
                    created_at
                FROM items
                WHERE user_id = %(source_user_id)s
                """,
                {"source_user_id": source_user_id, "target_user_id": target_user_id},
            )
            inserted_count = cur.rowcount

            # Copy chunks
            await cur.execute(
                """
                INSERT INTO item_chunks (item_id, position, content_text, content_token_count, mistral_embedding, created_at)
                SELECT dest.id, c.position, c.content_text, c.content_token_count, c.mistral_embedding, c.created_at
                FROM item_chunks AS c
                JOIN items AS src ON src.id = c.item_id AND src.user_id = %(source_user_id)s
                JOIN items AS dest ON dest.user_id = %(target_user_id)s AND dest.url = src.url
                """,
                {"source_user_id": source_user_id, "target_user_id": target_user_id},
            )
            chunk_count = cur.rowcount

            # Copy user settings
            await cur.execute(
                """
                INSERT INTO user_settings (
                    user_id,
                    setting_type,
                    setting_key,
                    setting_value,
                    updated_at,
                    created_at
                )
                SELECT
                    %(target_user_id)s,
                    setting_type,
                    setting_key,
                    setting_value,
                    updated_at,
                    created_at
                FROM user_settings
                WHERE user_id = %(source_user_id)s
                ON CONFLICT (user_id, setting_type, setting_key) DO NOTHING
                """,
                {"source_user_id": source_user_id, "target_user_id": target_user_id},
            )
            settings_count = cur.rowcount
        await conn.commit()

    return {"items": inserted_count, "item_chunks": chunk_count, "user_settings": settings_count}


# === ITEMS ===


async def create_item(payload: dict[str, Any]) -> dict[str, Any]:
    """Persist an item, returning the created row."""

    if not payload.get("user_id"):
        raise ValueError("Item must belong to a user")

    columns = list(payload.keys())
    column_identifiers = [sql.Identifier(col) for col in columns]
    value_placeholders = [sql.Placeholder(col) for col in columns]

    query = sql.SQL("INSERT INTO items ({}) VALUES ({}) RETURNING id").format(
        sql.SQL(", ").join(column_identifiers),
        sql.SQL(", ").join(value_placeholders)
    )

    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            try:
                await cur.execute(query, payload)
            except errors.ForeignKeyViolation as exc:
                await conn.rollback()
                raise ValueError("User does not exist") from exc

            row = await cur.fetchone()
            if not row:
                await conn.rollback()
                raise RuntimeError("Failed to insert item")
        await conn.commit()
    return _normalise_row(row)


async def get_item_by_url(*, user_id: str, url: str) -> dict[str, Any] | None:
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, user_id, url, created_at
                FROM items
                WHERE user_id = %(user_id)s AND url = %(url)s
                """,
                {"user_id": user_id, "url": url},
            )
            row = await cur.fetchone()
    return _normalise_row(row) if row else None


async def get_item(item_id: str, cols: list[str], user_id: str) -> dict[str, Any] | None:
    """Return dict of cols for an item by id ensuring ownership."""
    safe_cols = [col for col in cols if col in schemas.ITEM_PUBLIC_COLS]

    if not safe_cols:
        raise ValueError("No valid columns specified")

    column_identifiers = [sql.Identifier(col) for col in safe_cols]
    query = sql.SQL("SELECT {} FROM items WHERE id = %(item_id)s AND user_id = %(user_id)s").format(
        sql.SQL(", ").join(column_identifiers)
    )

    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, {"item_id": item_id, "user_id": user_id})
            row = await cur.fetchone()

    if not row:
        return None
    return _normalise_row(row)


async def get_items(
    columns: list[str],
    filters: list[tuple[str, str, Any]],
    user_id: str,
    limit: int | None = None,
    offset: int | None = None,
    order_by: str | None = None,
    order_direction: str | None = None,
) -> list[dict[str, Any]]:
    """
    General purpose select for items with user ownership check.

    Args:
        columns: List of column names to select.
        filters: List of (column, operator, value) tuples for WHERE clause.
        user_id: User ID to ensure ownership.
        limit: Maximum number of rows to return.
        offset: Number of rows to skip.
        order_by: Column to order by.
        order_direction: "asc" or "desc".

    Returns:
        List of dicts mapping column names to values.
    """
    allowed_operators = ["=", "!=", "<", "<=", ">", ">=", "LIKE", "ILIKE", "IN"]

    safe_cols = [col for col in columns if col in schemas.ITEM_PUBLIC_COLS]
    if not safe_cols:
        raise ValueError("No valid columns specified")

    # Validate filters
    safe_filters = []
    params: dict[str, Any] = {"user_id": user_id}
    param_counter = 0

    for col, op, val in filters:
        if col not in schemas.ITEM_PUBLIC_COLS or op.upper() not in allowed_operators:
            continue
        param_key = f"filter_{param_counter}"
        if op.upper() == "IN":
            safe_filters.append((col, op.upper(), param_key))
            params[param_key] = list(val) if isinstance(val, (list, tuple)) else [val]
        else:
            safe_filters.append((col, op.upper(), param_key))
            params[param_key] = val
        param_counter += 1

    # Build query
    column_identifiers = [sql.Identifier(col) for col in safe_cols]
    base_query = sql.SQL("SELECT {} FROM items WHERE user_id = %(user_id)s").format(
        sql.SQL(", ").join(column_identifiers)
    )

    if safe_filters:
        filter_conditions = []
        for col, op, param_key in safe_filters:
            if op == "IN":
                condition = sql.SQL("{} = ANY(%({})s)").format(
                    sql.Identifier(col),
                    sql.SQL(param_key)
                )
            else:
                condition = sql.SQL("{} {} %({})s").format(
                    sql.Identifier(col),
                    sql.SQL(op),
                    sql.SQL(param_key)
                )
            filter_conditions.append(condition)

        filter_clause = sql.SQL(" AND ").join(filter_conditions)
        query = sql.SQL("{} AND {}").format(base_query, filter_clause)
    else:
        query = base_query

    if order_by and order_by in schemas.ITEM_PUBLIC_COLS:
        order_direction_sql = sql.SQL("DESC") if order_direction == "desc" else sql.SQL("ASC")
        query = sql.SQL("{} ORDER BY {} {}").format(
            query, sql.Identifier(order_by), order_direction_sql
        )

    if limit is not None:
        query = sql.SQL("{} LIMIT %(limit)s").format(query)
        params["limit"] = limit
    if offset is not None:
        query = sql.SQL("{} OFFSET %(offset)s").format(query)
        params["offset"] = offset

    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)
            rows = await cur.fetchall()

    return [_normalise_row(row) for row in rows]


async def update_item(updates: dict[str, Any], item_id: str, user_id: str) -> dict[str, Any] | None:
    allowed_columns = schemas.ITEM_PUBLIC_COLS
    cols = [c for c in updates if c in allowed_columns]
    if not cols:
        raise ValueError("No valid item fields supplied for update")

    payload: dict[str, Any] = {"item_id": item_id}
    payload.update({c: updates[c] for c in cols})
    # Ensure vector fields are serialised in pgvector text format
    if "mistral_embedding" in payload and isinstance(payload["mistral_embedding"], (list, tuple)):
        payload["mistral_embedding"] = app_utils.vector_to_pg(payload["mistral_embedding"])  # type: ignore[arg-type]
    if user_id is not None:
        payload["user_id"] = user_id

    set_parts = [
        sql.SQL("{} = {}").format(sql.Identifier(c), sql.Placeholder(c))
        for c in cols
    ]

    where_parts = [sql.SQL("id = {}").format(sql.Placeholder("item_id"))]
    if user_id is not None:
        where_parts.append(sql.SQL("AND user_id = {}").format(sql.Placeholder("user_id")))

    query = sql.SQL(
        "UPDATE items SET {set_clause} WHERE {where_clause} RETURNING {ret}"
    ).format(
        set_clause=sql.SQL(", ").join(set_parts),
        where_clause=sql.SQL(" ").join(where_parts),
        ret=sql.SQL(", ").join(sql.Identifier(c) for c in allowed_columns),
    )

    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, payload)
            row = await cur.fetchone()
        await conn.commit()

    if not row:
        return None
    return _normalise_row(row)


async def delete_item(item_id: str, user_id: str) -> bool:
    """Delete an item, optionally scoping to a user, returning success status."""

    params: dict[str, Any] = {"item_id": item_id}
    where_parts = [sql.SQL("id = {}").format(sql.Placeholder("item_id"))]

    if user_id:
        where_parts.append(sql.SQL("AND user_id = {}").format(sql.Placeholder("user_id")))
        params["user_id"] = user_id

    query = sql.SQL("DELETE FROM items WHERE {}").format(
        sql.SQL(" ").join(where_parts)
    )

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            deleted = cur.rowcount > 0
        await conn.commit()

    return deleted


async def lexical_search_items(*, user_id: str, query_text: str, columns: Sequence[str] | None = None, limit: int = 10) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("Limit must be positive")
    safe_columns = _ensure_columns(columns, schemas.ITEM_PUBLIC_COLS, ITEM_SEARCH_DEFAULT_COLUMNS)
    column_select = ", ".join(f"i.{col}" for col in safe_columns)
    if column_select:
        column_select = column_select + ", "
    params: dict[str, Any] = {"user_id": user_id, "limit": limit, "query": query_text.strip()}
    query = f"""
        SELECT {column_select}
               ts_rank(i.ts_embedding, plainto_tsquery('english', %(query)s)) AS score
        FROM items AS i
        WHERE i.user_id = %(user_id)s
          AND i.ts_embedding @@ plainto_tsquery('english', %(query)s)
        ORDER BY score DESC
        LIMIT %(limit)s
    """
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)  # type: ignore[arg-type]
            rows = await cur.fetchall()
    return [_normalise_row(row) for row in rows]


async def semantic_search_items(*, user_id: str, query_vector: Sequence[float], columns: Sequence[str] | None = None, limit: int = 10) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("Limit must be positive")
    safe_columns = _ensure_columns(columns, schemas.ITEM_PUBLIC_COLS, ITEM_SEARCH_DEFAULT_COLUMNS)
    column_select = ", ".join(f"i.{col}" for col in safe_columns)
    if column_select:
        column_select = column_select + ", "
    params: dict[str, Any] = {"user_id": user_id, "limit": limit, "query_vec": _vector_to_pg(query_vector)}
    distance_expr = "i.mistral_embedding <-> %(query_vec)s::vector"
    query = f"""
        SELECT {column_select}
               {distance_expr} AS distance,
               1.0 / (1.0 + ({distance_expr})::float) AS score
        FROM items AS i
        WHERE i.user_id = %(user_id)s
          AND i.mistral_embedding IS NOT NULL
        ORDER BY {distance_expr} ASC
        LIMIT %(limit)s
    """
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)  # type: ignore[arg-type]
            rows = await cur.fetchall()
    return [_normalise_row(row) for row in rows]


# === CHUNKS ===


async def lexical_search_chunks(*, user_id: str, query_text: str, columns: Sequence[str] | None = None, limit: int = 10) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("Limit must be positive")
    allowed_chunk_columns = tuple(CHUNK_COLUMN_SOURCES.keys())
    safe_columns = _ensure_columns(columns, allowed_chunk_columns, CHUNK_SEARCH_DEFAULT_COLUMNS)
    select_parts = [f"{CHUNK_COLUMN_SOURCES[col]} AS {col}" for col in safe_columns]
    column_select = ", ".join(select_parts)
    if column_select:
        column_select = column_select + ", "
    params: dict[str, Any] = {"user_id": user_id, "limit": limit, "query": query_text.strip()}
    query = f"""
        SELECT {column_select}
               ts_rank(c.ts_embedding, plainto_tsquery('english', %(query)s)) AS score
        FROM item_chunks AS c
        JOIN items AS i ON i.id = c.item_id
        WHERE i.user_id = %(user_id)s
          AND c.ts_embedding @@ plainto_tsquery('english', %(query)s)
        ORDER BY score DESC
        LIMIT %(limit)s
    """
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)  # type: ignore[arg-type]
            rows = await cur.fetchall()
    return [_normalise_row(row) for row in rows]


async def semantic_search_chunks(*, user_id: str, query_vector: Sequence[float], columns: Sequence[str] | None = None, limit: int = 10) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("Limit must be positive")
    allowed_chunk_columns = tuple(CHUNK_COLUMN_SOURCES.keys())
    safe_columns = _ensure_columns(columns, allowed_chunk_columns, CHUNK_SEARCH_DEFAULT_COLUMNS)
    select_parts = [f"{CHUNK_COLUMN_SOURCES[col]} AS {col}" for col in safe_columns]
    column_select = ", ".join(select_parts)
    if column_select:
        column_select = column_select + ", "
    params: dict[str, Any] = {"user_id": user_id, "limit": limit, "query_vec": _vector_to_pg(query_vector)}
    distance_expr = "c.mistral_embedding <-> %(query_vec)s::vector"
    query = f"""
        SELECT {column_select}
               {distance_expr} AS distance,
               1.0 / (1.0 + ({distance_expr})::float) AS score
        FROM item_chunks AS c
        JOIN items AS i ON i.id = c.item_id
        WHERE i.user_id = %(user_id)s
          AND c.mistral_embedding IS NOT NULL
        ORDER BY {distance_expr} ASC
        LIMIT %(limit)s
    """
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)  # type: ignore[arg-type]
            rows = await cur.fetchall()
    return [_normalise_row(row) for row in rows]


async def add_item_chunks(*, item_id: str, chunks: Sequence[dict[str, Any]]) -> None:
    """Persist chunk embeddings for an item."""

    if not chunks:
        return

    records: list[dict[str, Any]] = []
    for position, chunk in enumerate(chunks):
        embedding = chunk.get("mistral_embedding")
        if embedding is None:
            raise ValueError("Chunk embedding missing")
        records.append(
            {
                "item_id": item_id,
                "position": position,
                "content_text": chunk.get("content_text"),
                "content_token_count": chunk.get("content_token_count"),
                # Serialize to pgvector text format
                "mistral_embedding": app_utils.vector_to_pg(embedding),
            }
        )

    # Insert in batches
    stmt = (
        """
        INSERT INTO item_chunks (item_id, position, content_text, content_token_count, mistral_embedding)
        VALUES (%(item_id)s, %(position)s, %(content_text)s, %(content_token_count)s, %(mistral_embedding)s::vector)
        ON CONFLICT (item_id, position) DO UPDATE SET
            content_text = EXCLUDED.content_text,
            content_token_count = EXCLUDED.content_token_count,
            mistral_embedding = EXCLUDED.mistral_embedding
        """
    )

    async def _insert_batch(batch: Sequence[dict[str, Any]]) -> None:
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(stmt, batch)
            await conn.commit()

    # Process records in batches
    batch_size = max(1, CHUNK_INSERT_BATCH_SIZE)
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]

        try:
            await _insert_batch(batch)
        except Exception as exc:
            # Print error but don't raise - retry with half batch size
            print(f"Batch insert failed, retrying with smaller batch: {exc}")

            # Retry with half batch size
            half_size = max(1, len(batch) // 2)
            for j in range(0, len(batch), half_size):
                small_batch = batch[j : j + half_size]
                await _insert_batch(small_batch)


# === USAGE LOGS ===


async def create_usage_log(
    response: Response,
    operation: str,
    *,
    user_id: str | None,
    item_id: str | None = None,
) -> dict[str, Any]:
    """Create a usage log entry"""
    usage_details = pricing.prepare_usage_log(
        response.provider,
        response.model,
        getattr(response, "usage", {}),
    )

    log = {
        "operation": operation,
        "provider": response.provider,
        "model": response.model,
        "prompt_tokens": usage_details.get("prompt_tokens"),
        "completion_tokens": usage_details.get("completion_tokens"),
        "prompt_cost": usage_details.get("prompt_cost"),
        "completion_cost": usage_details.get("completion_cost"),
        "total_cost": usage_details.get("total_cost"),
        "currency": usage_details.get("currency") or "USD",
        "created_at": datetime.now(),
        "user_id": user_id,
        "item_id": item_id,
    }
    columns = list(log.keys())
    column_identifiers = [sql.Identifier(col) for col in columns]
    value_placeholders = [sql.Placeholder(col) for col in columns]

    query = sql.SQL("INSERT INTO llm_usage_logs ({}) VALUES ({}) RETURNING id").format(
        sql.SQL(", ").join(column_identifiers),
        sql.SQL(", ").join(value_placeholders)
    )

    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, log)
            row = await cur.fetchone()
            if not row:
                raise RuntimeError("Failed to insert item")
        await conn.commit()
    return _normalise_row(row)


# === EMAIL SOURCES ===


async def create_email_source(
    *,
    item_id: str,
    message_id: str,
    slug: str,
    title: str | None = None,
    resolved_url: str | None = None,
    html_content: str | None = None,
) -> dict[str, Any]:
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO email_sources (item_id, message_id, resolved_url, slug, title, html_content)
                VALUES (%(item_id)s, %(message_id)s, %(resolved_url)s, %(slug)s, %(title)s, %(html_content)s)
                RETURNING id, item_id, message_id, resolved_url, slug, title, html_content, created_at, updated_at
                """,
                {
                    "item_id": item_id,
                    "message_id": message_id,
                    "resolved_url": resolved_url,
                    "slug": slug,
                    "title": title,
                    "html_content": html_content,
                },
            )
            row = await cur.fetchone()
        await conn.commit()
    if not row:
        raise RuntimeError("Failed to insert email source")
    return _normalise_row(row)


async def get_email_source_by_message_id(message_id: str) -> dict[str, Any] | None:
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT es.id, es.item_id, es.message_id, es.resolved_url, es.slug, es.title,
                       es.html_content, es.created_at, es.updated_at, i.user_id
                FROM email_sources es
                JOIN items i ON es.item_id = i.id
                WHERE es.message_id = %(message_id)s
                """,
                {"message_id": message_id},
            )
            row = await cur.fetchone()
    return _normalise_row(row) if row else None


async def get_email_source_by_slug(slug: str) -> dict[str, Any] | None:
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT es.id, es.item_id, es.message_id, es.resolved_url, es.slug, es.title,
                       es.html_content, es.created_at, es.updated_at, i.user_id
                FROM email_sources es
                JOIN items i ON es.item_id = i.id
                WHERE es.slug = %(slug)s
                """,
                {"slug": slug},
            )
            row = await cur.fetchone()
    return _normalise_row(row) if row else None


async def email_slug_exists(slug: str) -> bool:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM email_sources WHERE slug = %s",
                (slug,),
            )
            row = await cur.fetchone()
    return bool(row)


async def update_email_source_html(*, email_source_id: str, html_content: str | None) -> dict[str, Any] | None:
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                UPDATE email_sources
                SET html_content = %(html_content)s,
                    updated_at = NOW()
                WHERE id = %(email_source_id)s
                RETURNING id, item_id, message_id, resolved_url, slug, title, html_content, created_at, updated_at
                """,
                {"email_source_id": email_source_id, "html_content": html_content},
            )
            row = await cur.fetchone()
        await conn.commit()
    return _normalise_row(row) if row else None


# === GMAIL CONFIGURATION ===


async def upsert_gmail_credentials(
    *,
    user_id: str,
    credentials: dict[str, Any],
    email_address: str | None = None,
    token_expiry: datetime | None = None,
) -> dict[str, Any]:
    """Insert or update Gmail OAuth credentials for a user."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO gmail_credentials (user_id, credentials, email_address, token_expiry, created_at, updated_at)
                VALUES (%(user_id)s, %(credentials)s::jsonb, %(email_address)s, %(token_expiry)s, NOW(), NOW())
                ON CONFLICT (user_id)
                DO UPDATE SET
                    credentials = EXCLUDED.credentials,
                    email_address = EXCLUDED.email_address,
                    token_expiry = EXCLUDED.token_expiry,
                    updated_at = NOW()
                RETURNING id, user_id, credentials, email_address, token_expiry, created_at, updated_at
                """,
                {
                    "user_id": user_id,
                    "credentials": json.dumps(credentials),
                    "email_address": email_address,
                    "token_expiry": token_expiry,
                },
            )
            row = await cur.fetchone()
        await conn.commit()
    if not row:
        raise RuntimeError("Failed to store Gmail credentials")
    return _normalise_row(row)


async def get_gmail_credentials(user_id: str) -> dict[str, Any] | None:
    """Retrieve stored Gmail credentials for a user, if present."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, user_id, credentials, email_address, token_expiry, created_at, updated_at
                FROM gmail_credentials
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = await cur.fetchone()
    return _normalise_row(row) if row else None


async def delete_gmail_credentials(user_id: str) -> bool:
    """Delete stored Gmail credentials for a user."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM gmail_credentials WHERE user_id = %s",
                (user_id,),
            )
            deleted = cur.rowcount > 0
        await conn.commit()
    return deleted


async def list_gmail_senders(user_id: str) -> list[dict[str, Any]]:
    """List configured Gmail newsletter senders for a user."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, user_id, email_address, label, created_at
                FROM gmail_senders
                WHERE user_id = %s
                ORDER BY created_at ASC
                """,
                (user_id,),
            )
            rows = await cur.fetchall()
    return [_normalise_row(row) for row in rows]


async def add_gmail_sender(
    *,
    user_id: str,
    email_address: str,
    label: str | None = None,
) -> dict[str, Any]:
    """Add a sender to the Gmail newsletter configuration."""
    if not email_address or not email_address.strip():
        raise ValueError("Email address is required")

    normalised = email_address.strip().lower()
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO gmail_senders (user_id, email_address, label)
                VALUES (%(user_id)s, %(email_address)s, %(label)s)
                ON CONFLICT (user_id, email_address)
                DO UPDATE SET
                    label = COALESCE(EXCLUDED.label, gmail_senders.label),
                    created_at = gmail_senders.created_at
                RETURNING id, user_id, email_address, label, created_at
                """,
                {
                    "user_id": user_id,
                    "email_address": normalised,
                    "label": label,
                },
            )
            row = await cur.fetchone()
        await conn.commit()
    if not row:
        raise RuntimeError("Failed to add Gmail sender")
    return _normalise_row(row)


async def remove_gmail_sender(*, user_id: str, sender_id: str) -> bool:
    """Remove a sender from the Gmail newsletter configuration."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                DELETE FROM gmail_senders
                WHERE user_id = %s AND id = %s
                """,
                (user_id, sender_id),
            )
            deleted = cur.rowcount > 0
        await conn.commit()
    return deleted


# === UTILITIES ===


def _normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    """Transform database row values into JSON-serialisable primitives."""

    result: dict[str, Any] = {}
    for column in row.keys():
        value = row.get(column)
        if isinstance(value, uuid.UUID):
            result[column] = str(value)
        elif isinstance(value, Decimal):
            result[column] = str(value) # to keep precision
        else:
            result[column] = value
    return result


# === USER SETTINGS ===


async def get_user_setting(user_id: str, setting_type: str, setting_key: str) -> dict[str, Any] | None:
    """Get a specific user setting."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT setting_value FROM user_settings WHERE user_id = %s AND setting_type = %s AND setting_key = %s",
                (user_id, setting_type, setting_key)
            )
            row = await cur.fetchone()
            if row:
                return row["setting_value"]  # type: ignore[return-value]
            return None


async def set_user_setting(user_id: str, setting_type: str, setting_key: str, setting_value: dict[str, Any]) -> None:
    """Set a user setting."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO user_settings (user_id, setting_type, setting_key, setting_value, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (user_id, setting_type, setting_key)
                DO UPDATE SET
                    setting_value = EXCLUDED.setting_value,
                    updated_at = NOW()
                """,
                (user_id, setting_type, setting_key, json.dumps(setting_value))
            )
        await conn.commit()


async def update_user_setting_field(user_id: str, setting_type: str, setting_key: str, field_key: str, field_value: Any) -> None:
    """Update a single field within a user setting."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # First try to update existing record
            await cur.execute(
                """
                UPDATE user_settings
                SET setting_value = jsonb_set(setting_value, %s, %s, true),
                    updated_at = NOW()
                WHERE user_id = %s AND setting_type = %s AND setting_key = %s
                """,
                (f'{{{field_key}}}', json.dumps(field_value), user_id, setting_type, setting_key)
            )

            # If no rows were updated, insert a new record
            if cur.rowcount == 0:
                setting_value = {field_key: field_value}
                await cur.execute(
                    """
                    INSERT INTO user_settings (user_id, setting_type, setting_key, setting_value)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (user_id, setting_type, setting_key, json.dumps(setting_value))
                )
        await conn.commit()


async def get_user_settings_by_type(user_id: str, setting_type: str) -> dict[str, dict[str, Any]]:
    """Get all user settings of a specific type."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT setting_key, setting_value FROM user_settings WHERE user_id = %s AND setting_type = %s",
                (user_id, setting_type)
            )
            rows = await cur.fetchall()
            return {row["setting_key"]: row["setting_value"] for row in rows}  # type: ignore[misc]


# ===== UTILITIES =====


def _ensure_columns(
    requested: Sequence[str] | None,
    allowed: Sequence[str],
    default: Sequence[str],
) -> list[str]:
    """Return a validated list of columns limited to an allow-list."""

    if requested:
        safe = [col for col in requested if col in allowed]
    else:
        safe = list(default)
    if not safe:
        raise ValueError("No valid columns specified")
    return safe


def _vector_to_pg(vec: Sequence[float]) -> str:
    """Deprecated: use app.utils.vector_to_pg. Kept for backward compatibility."""
    return app_utils.vector_to_pg(vec)


# ===== EXPORTS =====


__all__ = [
    "get_connection",
    "close_pool",
    "init_database",
    "create_user",
    "authenticate_user",
    "create_user_access_token",
    "list_user_access_tokens",
    "revoke_user_access_token",
    "get_user_access_token",
    "create_item",
    "get_item_by_url",
    "get_item",
    "get_items",
    "lexical_search_items",
    "semantic_search_items",
    "lexical_search_chunks",
    "semantic_search_chunks",
    "update_item",
    "delete_item",
    "add_item_chunks",
    "create_usage_log",
    "create_email_source",
    "get_email_source_by_message_id",
    "get_email_source_by_slug",
    "email_slug_exists",
    "update_email_source_html",
    "upsert_gmail_credentials",
    "get_gmail_credentials",
    "delete_gmail_credentials",
    "list_gmail_senders",
    "add_gmail_sender",
    "remove_gmail_sender",
    "get_user_setting",
    "set_user_setting",
    "update_user_setting_field",
    "get_user_settings_by_type",
]
