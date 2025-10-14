from __future__ import annotations

from typing import Any, Sequence
import os
from functools import lru_cache

import cohere

from .embedding import embed_query
from .. import database as db
from .search_agents.base import SearchItem


"""Simple retrieval pipeline with clear steps and tunables.

Steps:
1) Semantic chunk search to gather candidate items and preview chunks
2) Optional cross-encode scoring with Cohere to assess relevance
3) Rerank by CE score (fallback to semantic when CE disabled) and return top_k

Tunable via env variables:
- RETRIEVAL_FETCH_FACTOR (default 8)
- SEMANTIC_MIN_SCORE (default 0.0)
- CE_THRESHOLD (default 0.65)
- PREVIEW_CHUNK_COUNT (default 3)
- COHERE_RERANK_MODEL (default 'rerank-english-v3.0')
"""


# === Tunables ===

RETRIEVAL_FETCH_FACTOR = int(os.getenv("RETRIEVAL_FETCH_FACTOR", "8"))
SEMANTIC_MIN_SCORE = float(os.getenv("SEMANTIC_MIN_SCORE", "0.0"))
PREVIEW_CHUNK_COUNT = int(os.getenv("PREVIEW_CHUNK_COUNT", "3"))
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")


@lru_cache(maxsize=1)
def _get_cohere_client() -> cohere.Client | None:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return None
    return cohere.Client(api_key)


def _prepare_documents(candidates: Sequence[dict[str, Any]]) -> list[str]:
    docs: list[str] = []
    for c in candidates:
        parts: list[str] = []
        t = c.get("title")
        if isinstance(t, str) and t.strip():
            parts.append(t.strip())
        s = c.get("summary")
        if isinstance(s, str) and s.strip():
            parts.append(s.strip())
        chunks = c.get("chunks") or []
        if chunks:
            parts.append(" ".join(chunks)[:1000])
        txt = " ".join(parts).strip()
        docs.append(txt if txt else " ")
    return docs


async def retrieve_candidates(*, user_id: str, query: str, top_k: int = 10, ce_threshold: float | None = 0.65) -> list[SearchItem]:
    """Semantic → (optional) CE → rerank, minimal logic.

    ce_threshold: when provided and CE is enabled, drop items with CE score below this.
                  when None, do not drop on CE score (still rerank).
    """
    if not query:
        return []

    # Enforce max 10 results from retrieval
    k = max(1, min(int(top_k or 10), 10))

    # 1) Embed + search many chunks
    vec = await embed_query(query)
    columns = ["item_id", "content_text", "title", "summary", "score", "distance"]
    raw = await db.semantic_search_chunks(
        user_id=user_id,
        query_vector=vec,
        columns=columns,
        # Fetch extra to ensure enough unique items before merge/rerank
        limit=max(100, k * max(1, RETRIEVAL_FETCH_FACTOR)),
    )
    if not raw:
        return []
    if SEMANTIC_MIN_SCORE > 0:
        raw = [r for r in raw if float(r.get("score") or 0.0) >= SEMANTIC_MIN_SCORE]

    # 2) Group by item, keep top few chunks for preview
    by_item: dict[str, dict[str, Any]] = {}
    for row in raw:
        iid = str(row.get("item_id") or "").strip()
        if not iid:
            continue
        rec = by_item.get(iid)
        if rec is None:
            rec = {
                "id": iid,
                "title": row.get("title"),
                "summary": row.get("summary"),
                "chunks": [],
                "score": float(row.get("score") or 0.0),
            }
            by_item[iid] = rec
        # Accumulate up to PREVIEW_CHUNK_COUNT representative chunks
        if isinstance(row.get("content_text"), str) and len(rec["chunks"]) < PREVIEW_CHUNK_COUNT:
            rec["chunks"].append(row["content_text"])
        # Track best semantic score per item
        if (row.get("score") or 0.0) > (rec.get("score") or 0.0):
            rec["score"] = float(row.get("score") or 0.0)

    items = list(by_item.values())

    # 3) Cross-encode: score relevance via Cohere if configured
    client = _get_cohere_client()
    ce_scores: dict[str, float] = {}
    if client is not None and items:
        docs = _prepare_documents(items)
        try:
            resp = await __import__("asyncio").to_thread(
                client.rerank,
                query=query,
                documents=docs,
                model=COHERE_RERANK_MODEL,
                top_n=len(docs),
                return_documents=False,
            )
            for r in resp.results:
                idx = int(r.index)
                if 0 <= idx < len(items):
                    ce_scores[items[idx]["id"]] = float(r.relevance_score)
        except Exception:
            ce_scores = {}

    # 4) Build outputs, rerank and crop (no CE threshold filtering)
    out: list[SearchItem] = []
    for it in items:
        iid = str(it.get("id"))
        sem = float(it.get("score") or 0.0)
        ce = ce_scores.get(iid)

        out.append(
            SearchItem(
                id=iid,
                preview=list(it.get("chunks") or []),
                title=it.get("title"),
                summary=it.get("summary"),
                score=(float(ce) if ce is not None else sem),
                semantic_score=sem,
                cross_encoder_score=(float(ce) if ce is not None else None),
                combined_score=(float(ce) if ce is not None else sem),
            )
        )

    # Rerank by CE score when present; otherwise by semantic score
    out.sort(key=lambda s: (s.cross_encoder_score if s.cross_encoder_score is not None else (s.semantic_score or 0.0)), reverse=True)
    return out[:k]


def filter_retrieval(
    items: list[SearchItem],
    *,
    limit: int,
    alpha: float,
    tau: float,
    min_keep: int | None = None,
) -> list[SearchItem]:
    """Filter retrieval candidates using a relative/absolute cutoff.

    - Compute s_max across CE scores
    - cutoff = max(alpha * s_max, tau)
    - Keep all with CE score >= cutoff, preserving descending order
    - If fewer than min_keep items meet cutoff, include top items until min_keep is reached
    """
    if not items:
        return []

    # Sort in descending order by CE score when available; fallback to semantic
    sorted_items = sorted(
        items,
        key=lambda s: (s.cross_encoder_score if s.cross_encoder_score is not None else (s.semantic_score or 0.0)),
        reverse=True,
    )

    ce_scores = [(it.cross_encoder_score or 0.0) for it in sorted_items]
    s_max = max(ce_scores) if ce_scores else 0.0
    cutoff = max(alpha * s_max, tau)

    kept: list[SearchItem] = [it for it in sorted_items if (it.cross_encoder_score or 0.0) >= cutoff]

    if min_keep is not None and min_keep > 0 and len(kept) < min_keep:
        target = min(min_keep, len(sorted_items))
        seen = {id(it) for it in kept}
        for it in sorted_items:
            if len(kept) >= target:
                break
            if id(it) not in seen:
                kept.append(it)
                seen.add(id(it))

    # Enforce limit
    max_keep = max(1, int(limit or 1))
    return kept[:max_keep]
