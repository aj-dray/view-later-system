"""
Database schema definitions.
"""


# === VARIABLES ===


NN_EMBEDDING_SIZE = 1024


# === EXTENSIONS ===


EXTENSIONS = [
    "CREATE EXTENSION IF NOT EXISTS pgcrypto",
    "CREATE EXTENSION IF NOT EXISTS vector",
]


# === ENUMS ===


CLIENT_STATUSES = [
    'adding',
    'queued',
    'paused',
    'completed',
    'bookmark',
    'error',
]
SERVER_STATUSES = ['saved', 'extracted', 'summarised', 'embedded', 'classified']
ENUMS = [
    f"""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_type WHERE typname = 'item_client_status'
        ) THEN
            CREATE TYPE item_client_status AS ENUM (
                {', '.join(f"'{s}'" for s in CLIENT_STATUSES)}
            );
        END IF;
    END$$
    """,
    *[
        f"""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_type WHERE typname = 'item_client_status'
            ) AND NOT EXISTS (
                SELECT 1
                FROM pg_enum
                WHERE enumlabel = '{status}'
                  AND enumtypid = 'item_client_status'::regtype
            ) THEN
                ALTER TYPE item_client_status ADD VALUE '{status}';
            END IF;
        END$$
        """
        for status in CLIENT_STATUSES
    ],
    f"""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_type WHERE typname = 'item_server_status'
        ) THEN
            CREATE TYPE item_server_status AS ENUM (
                {', '.join(f"'{s}'" for s in SERVER_STATUSES)}
            );
        END IF;
    END$$
    """,
]


# === TABLES ===


def get_users_table() -> str:
    """Users table schema."""
    return """
    CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """

def get_items_table() -> str:
    """Items table schema with configurable embedding column type."""
    return f"""
    CREATE TABLE IF NOT EXISTS items (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id) ON DELETE SET NULL,
        url TEXT NOT NULL,
        canonical_url TEXT,
        title TEXT,
        source_site TEXT,
        format TEXT,
        author TEXT,
        type TEXT,
        publication_date TIMESTAMPTZ,
        favicon_url TEXT,
        content_markdown TEXT,
        content_text TEXT,
        duration TEXT,
        content_token_count INTEGER,
        client_status item_client_status,
        server_status item_server_status NOT NULL DEFAULT 'saved',
        summary TEXT,
        expiry_score DOUBLE PRECISION,
        ts_embedding TSVECTOR GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(content_text, ''))
        ) STORED,
        mistral_embedding VECTOR({NN_EMBEDDING_SIZE}),
        client_status_at TIMESTAMPTZ,
        server_status_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """


def get_email_sources_table() -> str:
    """Store newsletter imports sourced from Gmail."""
    return """
    CREATE TABLE IF NOT EXISTS email_sources (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        item_id UUID NOT NULL REFERENCES items(id) ON DELETE CASCADE,
        message_id TEXT NOT NULL UNIQUE,
        resolved_url TEXT,
        slug TEXT NOT NULL UNIQUE,
        title TEXT,
        html_content TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """

def get_gmail_credentials_table() -> str:
    """Persist Gmail OAuth credentials per user."""
    return """
    CREATE TABLE IF NOT EXISTS gmail_credentials (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        credentials JSONB NOT NULL,
        email_address TEXT,
        token_expiry TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (user_id)
    )
    """


def get_gmail_senders_table() -> str:
    """Track user-configured Gmail newsletter senders."""
    return """
    CREATE TABLE IF NOT EXISTS gmail_senders (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        email_address TEXT NOT NULL,
        label TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (user_id, email_address)
    )
    """

ITEM_PUBLIC_COLS = [
        "id",
        "user_id",
        "url",
        "canonical_url",
        "title",
        "format",
        "type",
        "author",
        "source_site",
        "publication_date",
        "favicon_url",
        "content_markdown",
        "content_text",
        "duration",
        "content_token_count",
        "client_status",
        "server_status",
        "summary",
        "expiry_score",
        "ts_embedding",
        "mistral_embedding",
        "client_status_at",
        "server_status_at",
        "created_at",
]


def get_item_chunks_table(embedding_column_type: str = "BYTEA") -> str:
    """Item chunks table schema with configurable embedding column type."""
    return f"""
    CREATE TABLE IF NOT EXISTS item_chunks (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        item_id UUID NOT NULL REFERENCES items(id) ON DELETE CASCADE,
        position INTEGER NOT NULL,
        content_text TEXT,
        content_token_count INTEGER,
        ts_embedding TSVECTOR GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(content_text, ''))
        ) STORED,
        mistral_embedding VECTOR({NN_EMBEDDING_SIZE}),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (item_id, position)
    )
    """

def get_usage_logs_table() -> str:
    """LLM usage log table schema."""
    return """
    CREATE TABLE IF NOT EXISTS llm_usage_logs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id) ON DELETE SET NULL,
        item_id UUID REFERENCES items(id) ON DELETE SET NULL,
        operation TEXT NOT NULL,
        provider TEXT,
        model TEXT,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        prompt_cost NUMERIC,
        completion_cost NUMERIC,
        total_cost NUMERIC,
        currency TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """

def get_user_access_tokens_table() -> str:
    """Store issued API access tokens for revocation and audit."""
    return """
    CREATE TABLE IF NOT EXISTS user_access_tokens (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token_id UUID NOT NULL UNIQUE,
        label TEXT,
        expires_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        revoked_at TIMESTAMPTZ
    )
    """

def get_user_settings_table() -> str:
    """User settings table schema for storing user preferences including UI controls."""
    return """
    CREATE TABLE IF NOT EXISTS user_settings (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        setting_type TEXT NOT NULL,
        setting_key TEXT NOT NULL,
        setting_value JSONB NOT NULL DEFAULT '{}',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (user_id, setting_type, setting_key)
    )
    """


# === INDEXES ===


INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_items_user_client_status ON items(user_id, client_status)",
    "CREATE INDEX IF NOT EXISTS idx_items_user_server_status ON items(user_id, server_status)",
    "CREATE INDEX IF NOT EXISTS idx_items_ts_embedding ON items USING GIN (ts_embedding)",
    "CREATE INDEX IF NOT EXISTS idx_items_mistral_embedding_ivfflat ON items USING ivfflat (mistral_embedding vector_cosine_ops) WITH (lists = 100)",
    "CREATE INDEX IF NOT EXISTS idx_item_chunks_ts_embedding ON item_chunks USING GIN (ts_embedding)",
    "CREATE INDEX IF NOT EXISTS idx_item_chunks_mistral_embedding_ivfflat ON item_chunks USING ivfflat (mistral_embedding vector_l2_ops) WITH (lists = 100)",
    "CREATE INDEX IF NOT EXISTS idx_llm_usage_logs_user_created_at ON llm_usage_logs(user_id, created_at DESC)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_items_user_url_unique ON items(user_id, url)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_items_user_canonical_url ON items(user_id, canonical_url) WHERE canonical_url IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_user_settings_lookup ON user_settings(user_id, setting_type, setting_key)",
    "CREATE INDEX IF NOT EXISTS idx_user_settings_type ON user_settings(user_id, setting_type)",
    "CREATE INDEX IF NOT EXISTS idx_email_sources_item ON email_sources(item_id)",
    "CREATE INDEX IF NOT EXISTS idx_user_access_tokens_user ON user_access_tokens(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_gmail_credentials_user ON gmail_credentials(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_gmail_senders_user ON gmail_senders(user_id)",
]


# === MIGRATIONS ===


COLUMN_ADDITIONS = [
    "ALTER TABLE items ADD COLUMN IF NOT EXISTS format TEXT",
    "ALTER TABLE items ADD COLUMN IF NOT EXISTS author TEXT",
    "ALTER TABLE items ADD COLUMN IF NOT EXISTS type TEXT",
    "ALTER TABLE items ADD COLUMN IF NOT EXISTS duration TEXT",
]


# === FULL SCHEMA ===


def get_create_sql() -> list[str]:
    """
    Get all SQL statements needed to create the complete database schema.

    Args:
        embedding_column_type: Column type for embedding fields (e.g., 'BYTEA', 'VECTOR(1536)')

    Returns:
        List of SQL statements to execute in order
    """
    statements = []

    # Extensions first
    statements.extend(EXTENSIONS)

    # Custom types
    statements.extend(ENUMS)

    # Tables
    statements.append(get_users_table())
    statements.append(get_items_table())
    statements.append(get_email_sources_table())
    statements.append(get_gmail_credentials_table())
    statements.append(get_gmail_senders_table())
    statements.append(get_item_chunks_table())
    statements.append(get_usage_logs_table())
    statements.append(get_user_access_tokens_table())
    statements.append(get_user_settings_table())

    # Column additions (for migrations)
    if COLUMN_ADDITIONS:
        statements.extend(COLUMN_ADDITIONS)

    # Indexes last
    statements.extend(INDEXES)

    return statements

__all__ = [
    "ITEM_PUBLIC_COLS",
]
