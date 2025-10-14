from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence
import json
import asyncio
import traceback


from .. import database as db
from .retrieval import retrieve_candidates
from .search_agents import StructuredAgent, UnstructuedAgent
from .search_agents.base import SearchItem

logger = logging.getLogger(__name__)


# === SEARCH HELPERS ===


def _coerce_float(value: Any) -> float | None:
    """Convert arbitrary values to float, returning None on failure."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _clean_str(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


@dataclass
class _AccumItem:
    item_id: str
    title: str | None = None
    summary: str | None = None
    score: float | None = None
    chunks: list[str] = field(default_factory=list)
    distance: float | None = None


def _merge_chunk_rows(rows: Iterable[dict[str, Any]]) -> list[_AccumItem]:
    by_id: dict[str, _AccumItem] = {}
    for row in rows:
        iid_raw = row.get("item_id")
        if iid_raw is None:
            continue
        iid = str(iid_raw)
        if not iid:
            continue
        item = by_id.get(iid)
        if item is None:
            item = _AccumItem(item_id=iid)
            by_id[iid] = item
        # title/summary best effort
        t = _clean_str(row.get("title"))
        if t and not item.title:
            item.title = t
        s = _clean_str(row.get("summary"))
        if s and not item.summary:
            item.summary = s
        # score: keep max per item
        sc = _coerce_float(row.get("score"))
        if sc is not None and (item.score is None or sc > item.score):
            item.score = sc
        # distance: keep min per item if present
        dist = _coerce_float(row.get("distance"))
        if dist is not None and (item.distance is None or dist < item.distance):
            item.distance = dist
        # chunks: collect up to 3
        ct = _clean_str(row.get("content_text"))
        if ct and len(item.chunks) < 3:
            item.chunks.append(ct)
    return list(by_id.values())


def parse_search_results(items: Sequence[SearchItem], *, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Build client payload from SearchItems and requested columns.

    - Always includes: id, score, preview (list[str])
    - Adds requested columns if present on the item (e.g., title, summary)
    - Drops internal fields (llm_score, combined_score, etc.)
    """
    cols = set(columns or [])
    payloads: list[dict[str, Any]] = []
    for it in items:
        entry: dict[str, Any] = {
            "id": it.id,
            "score": it.score if it.score is not None else (
                it.cross_encoder_score or it.semantic_score or 0.0
            ),
            "preview": list(it.preview or []),
        }
        if "title" in cols and getattr(it, "title", None) is not None:
            entry["title"] = it.title  # type: ignore[attr-defined]
        if "summary" in cols and it.summary is not None:
            entry["summary"] = it.summary
        payloads.append(entry)
    return payloads


async def _rank_items_from_chunks(rows: Sequence[dict[str, Any]], limit: int) -> list[SearchItem]:
    """Pick best chunks per item and convert to SearchItems."""
    accum = _merge_chunk_rows(rows)
    # Sort by score desc
    accum.sort(key=lambda e: e.score or 0.0, reverse=True)
    results: list[SearchItem] = []
    for a in accum[:limit]:
        results.append(
            SearchItem(
                id=a.item_id,
                preview=a.chunks,
                title=a.title,
                summary=a.summary,
                justification=None,
                is_just=True,
                semantic_score=a.score,
                cross_encoder_score=None,
                llm_score=None,
                combined_score=a.score,
                score=a.score,
                distance=a.distance,
            )
        )
    return results


async def lexical(
    *,
    user_id: str,
    query: str,
    limit: int,
    columns: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Lexical search using chunk-level retrieval, unified output.

    - Always retrieves chunks and aggregates per item
    - score: ts_rank-based max per item
    - preview: list of top relevant chunks
    """
    # Fetch extra chunks to ensure enough unique items
    chunk_rows = await db.lexical_search_chunks(
        user_id=user_id,
        query_text=query,
        columns=["item_id", "content_text", "title", "summary", "score"],
        limit=max(50, limit * 6),
    )
    items = await _rank_items_from_chunks(chunk_rows, limit)
    return parse_search_results(items, columns=columns)


async def semantic(
    *,
    user_id: str,
    query: str,
    limit: int,
    columns: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Semantic retrieval + rerank using shared retrieval function.

    - score: cross-encoder score (fallback to semantic score)
    - preview: list of top relevant chunks
    """
    top_k = max(1, min(int(limit or 10), 10))
    # Get up to top_k candidates (already reranked by CE when present)
    candidates = await retrieve_candidates(
        user_id=user_id,
        query=query,
        top_k=top_k,
    )

    # Apply the same cutoff policy as agent round 1 (no minimum keep)
    from .retrieval import filter_retrieval
    filtered = filter_retrieval(
        candidates,
        limit=top_k,
        alpha=0.85,
        tau=0.20,
        min_keep=None,
    )

    return parse_search_results(filtered, columns=columns)


async def agentic(
    *,
    user_id: str,
    query: str,
    limit: int,
    columns: Sequence[str] | None,
):
    """Streaming agentic search (SSE) with normalized final results.

    - Forwards status events as-is from LangGraphSearchAgent
    - Rewrites the final event's data to parsed payload using SearchItems stored on the agent
    - Yields text/event-stream lines: `data: {json}\n\n`
    """
    try:
        lg_agent = StructuredAgent(user_id=user_id, limit=limit)
        print(f"[searching.agentic] Starting astream for query: {query}")
        async for ev in lg_agent.astream(query=query):
            print(f"[searching.agentic] Got event: {ev.get('kind')}")
            if ev.get("kind") == "final":
                try:
                    items = getattr(lg_agent, "_final_results", [])
                    ev = {
                        **ev,
                        "data": parse_search_results(items[: max(1, int(limit or 10))], columns=columns),
                    }
                except Exception:
                    # If parsing fails, fall back to raw
                    pass
            payload = json.dumps(ev, ensure_ascii=False)
            yield f"data: {payload}\n\n"
    except asyncio.CancelledError:
        # Client disconnected / request aborted; stop streaming quietly
        return
    except Exception as exc:
        traceback.print_exc()
        err = {
            "kind": "final",
            "message": "no-results",
            "reason": f"agent-error:{exc.__class__.__name__}:{str(exc)}",
            "data": [],
        }
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
