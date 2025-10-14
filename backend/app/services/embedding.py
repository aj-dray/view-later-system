from __future__ import annotations

import asyncio
import html
import re
import unicodedata
from collections.abc import Sequence
from datetime import datetime
from itertools import accumulate
from typing import Any
from aglib import Client

from .. import database as db


# === VARIABLES ===


_EMBEDDING_PROVIDER = "mistral"
_EMBEDDING_MODEL = "mistral-embed"
_EMBEDDING_MAX_TOKENS = 8_192
_EMBEDDING_SAFETY_MARGIN = 256  # reduce risk of tokenizer undercount vs provider
_WORDS_PER_CHUNK = 400
_OVERLAP_WORDS = 80
_EMBED_BATCH_SIZE = 16


# === UTILITIES ===


def _clean_text(s: str) -> str:
    """Clean for embedding"""
    s = html.unescape(s)                     # decode entities
    s = unicodedata.normalize("NFC", s)      # normalize unicode
    s = s.replace("\u00A0", " ")             # non-breaking spaces -> space
    s = re.sub(r"[ \t]+", " ", s)            # collapse spaces
    s = re.sub(r"\s*\n\s*", "\n", s)         # trim around newlines
    s = re.sub(r"\n{3,}", "\n\n", s)         # max 2 consecutive newlines
    return s.strip()


_SENT_PAT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"\'])")
def _sentences(s: str) -> list[str]:
    # Fallback: if no clear sentences, return the whole thing
    parts = _SENT_PAT.split(s)
    return [p.strip() for p in parts if p.strip()]


def _mean_pool(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Cannot mean-pool an empty list of vectors")
    length = len(vectors[0])
    totals = [0.0] * length
    for vector in vectors:
        if len(vector) != length:
            raise ValueError("All embeddings must share the same dimensionality")
        for idx, value in enumerate(vector):
            totals[idx] += float(value)
    scale = 1.0 / len(vectors)
    return [value * scale for value in totals]


def _weighted_mean_pool(vectors: Sequence[Sequence[float]], weights: Sequence[float]) -> list[float]:
    if len(vectors) != len(weights):
        raise ValueError("Vectors and weights must share the same length")
    if not vectors:
        raise ValueError("Cannot pool an empty list of vectors")

    length = len(vectors[0])
    totals = [0.0] * length
    total_weight = 0.0
    for vector, weight in zip(vectors, weights):
        if len(vector) != length:
            raise ValueError("All embeddings must share the same dimensionality")
        if weight <= 0:
            continue
        total_weight += float(weight)
        for idx, value in enumerate(vector):
            totals[idx] += float(value) * float(weight)

    if total_weight <= 0:
        raise ValueError("Total weight for pooling must be positive")

    scale = 1.0 / total_weight
    return [value * scale for value in totals]


def _chunk_text(
    s: str,
    words_per_chunk: int = _WORDS_PER_CHUNK,
    overlap_words: int = _OVERLAP_WORDS,
) -> list[str]:
    if not s:
        return []
    sents = _sentences(s) or [s]
    # Precompute sentence word counts
    wc = [len(si.split()) for si in sents]
    cum = [0] + list(accumulate(wc))

    chunks = []
    i = 0
    target = words_per_chunk
    ov = overlap_words

    # Walk by word budget while keeping sentence boundaries
    while i < len(sents):
        # find j such that cum[j]-cum[i] <= target
        j = i
        while j < len(sents) and (cum[j+1] - cum[i]) <= target:
            j += 1
        if j == i:  # very long sentence; hard split by words
            words = sents[i].split()
            chunks.append(" ".join(words[:target]))
            # overlap on hard split
            back = max(0, target - ov)
            sents[i] = " ".join(words[back:])  # remainder with overlap
            continue
        chunk = " ".join(sents[i:j]).strip()
        chunks.append(chunk)
        # move start forward with overlap
        # compute how many sentences to step back to keep ~ov words overlap
        k = j
        back_words = 0
        while k > i and back_words < ov:
            k -= 1
            back_words += wc[k]
        i = max(k, i + 1)  # ensure progress
    return chunks


def _create_embedding_text(raw: str) -> tuple[str, list[str]]:
    clean_text = _clean_text(raw)
    chunked_text = _chunk_text(clean_text)
    return clean_text, chunked_text


def _overlap_token_count(prev_chunk: str, chunk: str, token_counter) -> int:
    prev_tokens = prev_chunk.split()
    chunk_tokens = chunk.split()
    max_overlap = min(len(prev_tokens), len(chunk_tokens))
    overlap_words = 0
    for size in range(max_overlap, 0, -1):
        if prev_tokens[-size:] == chunk_tokens[:size]:
            overlap_words = size
            break
    if overlap_words == 0:
        return 0
    overlap_text = " ".join(chunk_tokens[:overlap_words])
    return token_counter(overlap_text)


def calc_effective_token_counts(chunks: Sequence[str], token_counts: Sequence[int], token_counter) -> list[int]:
    """Accounts for overlap"""
    effective: list[int] = []
    for idx, (chunk, token_count) in enumerate(zip(chunks, token_counts)):
        if idx == 0:
            effective.append(max(token_count, 0))
            continue
        overlap = _overlap_token_count(chunks[idx - 1], chunk, token_counter)
        effective.append(max(token_count - overlap, 0))
    return effective


# === EMBEDDING ===


async def index_item(
    item: dict[str, Any],
    *,
    item_id: str,
    user_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    text_source = item.get("content_text") or item.get("content_markdown")
    if not isinstance(text_source, str) or not text_source.strip():
        raise ValueError("Cannot embed item without extracted content")

    clean_text, chunks = _create_embedding_text(text_source)
    if not chunks:
        raise ValueError("Failed to split content into chunks for embedding")

    embedding_client = Client.embedding(
        provider=_EMBEDDING_PROVIDER, model=_EMBEDDING_MODEL
    )

    chunk_embeddings: list[list[float]] = []
    chunk_token_counts: list[int] = []

    for start in range(0, len(chunks), _EMBED_BATCH_SIZE):
        batch: Sequence[str] = chunks[start : start + _EMBED_BATCH_SIZE]
        batch_tokens = [embedding_client.token_counter(text) for text in batch]
        try:
            response = await asyncio.to_thread(embedding_client.request, input=batch)
        except Exception as exc:
            raise RuntimeError(f"Embedding provider failed for chunk batch: {exc}") from exc

        if not response.embeddings:
            raise RuntimeError("Embedding provider returned no embeddings for chunk batch")

        chunk_embeddings.extend(response.embeddings)
        chunk_token_counts.extend(batch_tokens)

        await db.create_usage_log(
            response,
            "embedding.item_chunk_batch",
            user_id=user_id,
            item_id=item_id,
        )

    item_chunks = [
        {
            "content_text": chunk_text,
            "mistral_embedding": embedding,
            "content_token_count": token_count,
        }
        for chunk_text, embedding, token_count in zip(
            chunks, chunk_embeddings, chunk_token_counts
        )
    ]

    embedding_vector: list[float] | None = None
    content_token_count = embedding_client.token_counter(clean_text)
    if content_token_count <= (_EMBEDDING_MAX_TOKENS - _EMBEDDING_SAFETY_MARGIN):
        try:
            response = await asyncio.to_thread(embedding_client.request, input=clean_text)
            if response.embeddings:
                embedding_vector = response.embeddings[0]
            await db.create_usage_log(
                response,
                "embedding.full_item",
                user_id=user_id,
                item_id=item_id,
            )
        except Exception as exc:
            # If provider rejects due to token limit despite our pre-check,
            # silently fall back to pooled chunk embeddings rather than failing pipeline.
            message = str(exc).lower()
            token_limit_signals = [
                "exceeding max",
                "too many tokens",
                "max tokens",
                "invalid_request_prompt",
            ]
            if any(sig in message for sig in token_limit_signals):
                embedding_vector = None  # will trigger pooling fallback below
            else:
                raise RuntimeError(f"Embedding provider failed for full text: {exc}") from exc

    if embedding_vector is None:
        if not chunk_embeddings:
            raise RuntimeError("No embeddings available for fallback pooling")
        token_counter = embedding_client.token_counter
        effective_counts = calc_effective_token_counts(
            chunks, chunk_token_counts, token_counter
        )
        total_effective = sum(effective_counts)
        if total_effective > 0:
            embedding_vector = _weighted_mean_pool(chunk_embeddings, effective_counts)
        else:
            embedding_vector = _mean_pool(chunk_embeddings)

    item_updates = {
        "content_text": clean_text,
        "content_token_count": content_token_count,
        "mistral_embedding": embedding_vector,
        "server_status": "embedded",
        "server_status_at": datetime.now()
    }

    return item_updates, item_chunks


async def embed_query(text: str) -> list[float]:
    """Generate a single embedding vector for ad-hoc semantic search queries."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Query text must not be empty")

    embedding_client = Client.embedding(
        provider=_EMBEDDING_PROVIDER, model=_EMBEDDING_MODEL
    )

    try:
        response = await asyncio.to_thread(embedding_client.request, input=[text])
    except Exception as exc:
        raise RuntimeError(f"Embedding provider failed for query: {exc}") from exc

    if not response.embeddings:
        raise RuntimeError("Embedding provider returned no query embeddings")

    vector = response.embeddings[0]
    return [float(value) for value in vector]
