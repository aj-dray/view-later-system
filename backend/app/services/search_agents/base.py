from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence


# === Shared data structures ===


@dataclass
class SearchItem:
    """Unified search entity.

    Minimal init: id and preview (list of relevant chunk texts).
    Additional properties can be attached as available per search method.
    """

    id: str
    preview: list[str]

    # Common optional fields
    score: float | None = None
    title: str | None = None
    summary: str | None = None
    justification: str | None = None

    # Scoring signals
    semantic_score: float | None = None
    cross_encoder_score: float | None = None
    llm_score: float | None = None
    combined_score: float | None = None
    distance: float | None = None

    # Internal-only flag for agent loops (not sent to client)
    is_just: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise for streaming/debugging. Excludes internal-only fields."""
        d = asdict(self)
        d.pop("is_just", None)
        return d


# === Shared helpers ===


def compute_combined_score(
    *,
    semantic_score: float | None,
    cross_encoder_score: float | None,
    llm_score: float | None,
) -> float:
    sem = float(semantic_score or 0.0)
    ce = float(cross_encoder_score or 0.0)
    llm = float(llm_score or 0.0)
    return 0.6 * ce + 0.2 * sem + 0.2 * llm


def build_justify_prompt(
    user_query: str,
    current_query: str,
    candidates: Sequence[dict[str, Any]],
    history: Sequence[dict[str, Any]]
) -> str:
    """Build prompt for justifying individual candidates (Phase 1)."""
    lines: list[str] = []
    lines.append("TASK: Classify each retrieved item for relevance to the user's query.\n")

    lines.append(f"USER'S ORIGINAL QUERY: {user_query}")
    if current_query != user_query:
        lines.append(f"CURRENT SEARCH QUERY: {current_query}")

    # Add context from previous rounds
    if history:
        lines.append("\nPREVIOUS SEARCH ATTEMPTS:")
        for h in history[-2:]:  # Show last 2 rounds for context
            lines.append(f"  Round {h.get('round', '?')}: query='{h.get('query', '')}' → {h.get('num_justified', 0)}/{h.get('num_candidates', 0)} relevant")

    lines.append("\nRETRIEVED CANDIDATES:")
    for idx, c in enumerate(candidates, 1):
        cid = c.get("id", "")
        summary = c.get("summary") or "No summary"
        chunks = c.get("chunks") or []
        chunk_preview = str(chunks[0])[:150] if chunks else ""
        lines.append(f"{idx}. ID: {cid}")
        lines.append(f"   Summary: {summary}")
        if chunk_preview:
            lines.append(f"   Content: {chunk_preview}...")

    lines.append("\nOUTPUT FORMAT (strict JSON array):")
    lines.append('[')
    lines.append('  {"id": "...", "is_just": true/false, "justification": "10-15 word explanation", "score": 0.0-1.0},')
    lines.append('  ...')
    lines.append(']')
    lines.append("\nRULES:")
    lines.append("- is_just=true ONLY if item directly addresses user's query")
    lines.append("- score: 1.0=perfect match, 0.8=strong, 0.6=moderate, 0.4=weak, 0.0=irrelevant")
    lines.append("- Respond with JSON array only, no other text")

    return "\n".join(lines)


def build_evaluate_prompt(
    user_query: str,
    current_query: str,
    justifications: Sequence[dict[str, Any]],
    history: Sequence[dict[str, Any]]
) -> str:
    """Build prompt for evaluating overall results quality."""
    lines: list[str] = []

    lines.append("Evaluate whether these search results adequately answer the user's question.\n")
    lines.append(f"USER'S QUESTION: {user_query}")
    lines.append(f"CURRENT SEARCH QUERY: {current_query}\n")

    # Show search history if exists
    if history:
        lines.append("PREVIOUS ATTEMPTS:")
        for h in history:
            lines.append(f"  Round {h.get('round')}: '{h.get('query')}' found {h.get('num_justified', 0)} relevant items")
        lines.append("")

    # Show current results
    justified = [j for j in justifications if j.get("is_just")]

    lines.append(f"CURRENT RESULTS: {len(justified)} relevant items found\n")

    if justified:
        lines.append("Relevant items:")
        for j in justified:  # Show all items
            score = j.get('score', 0)
            just = j.get('justification', '')
            lines.append(f"  [{score:.2f}] {just}")

    lines.append("\nAre these results good enough to answer the user's question?")
    lines.append("Consider: relevance, coverage, and quality of results.")
    lines.append("\nRespond with JSON:")
    lines.append('{')
    lines.append('  "decision": "accept" or "refine",')
    lines.append('  "reasoning": "Why you made this decision",')
    lines.append('  "new_query": "Better search query if refining, else null"')
    lines.append('}')

    return "\n".join(lines)
