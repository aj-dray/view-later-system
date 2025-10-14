from __future__ import annotations

import json
import traceback
from typing import Any, AsyncGenerator, Annotated, TypedDict
from operator import add

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from .base import (
    SearchItem,
    build_justify_prompt,
    compute_combined_score,
)
from ..retrieval import retrieve_candidates, filter_retrieval
from ... import database as db


class MockResponse:
    """Wrapper to make langchain responses compatible with create_usage_log."""

    def __init__(self, langchain_response: Any, provider: str = "openai", model: str = "gpt-4o"):
        self.provider = provider
        self.model = model
        self.content = getattr(langchain_response, "content", str(langchain_response))

        # Extract usage information from langchain response
        usage_metadata = {}
        if hasattr(langchain_response, "usage_metadata"):
            usage_metadata = langchain_response.usage_metadata or {}
        elif hasattr(langchain_response, "response_metadata"):
            # Fallback for older langchain versions
            response_metadata = langchain_response.response_metadata or {}
            usage_metadata = response_metadata.get("token_usage", {})

        # Map langchain usage keys to expected format
        self.usage = {
            "prompt_tokens": usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens", 0),
            "completion_tokens": usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens", 0),
        }


# filter_retrieval moved to ..retrieval for shared use


class AgentState(TypedDict):
    """State object passed between nodes."""
    # Input
    user_query: str
    user_id: str
    limit: int
    intent: str

    # Loop state
    current_query: str
    round_idx: int
    max_rounds: int

    # Per-round data
    candidates: list[SearchItem]
    justifications: list[dict[str, Any]]

    # Accumulated data
    justified: dict[str, SearchItem]
    history: list[dict[str, Any]]

    # Decision state (produced by evaluate)
    should_refine: bool
    reasoning: str
    new_query: str | None

    # Output streaming
    events: Annotated[list[dict[str, Any]], add]
    error: str | None


def build_intent_prompt(user_query: str) -> str:
    """Build prompt for calm intent extraction and an embedding-friendly query."""
    return f"""Extract a high-level search intent and generate a neutral embedding query for dense retrieval (not a natural-language question).

USER'S INPUT (from a search box): {user_query}

Intent guidelines:
- Phrase intent as an information need, e.g., "Find items covering …" or "Find items about …".
- Do not assume details the user did not state; keep breadth and neutrality.
- If multiple aspects are present, include all at a high level without narrowing.
- Avoid prescriptive tasks like definitions, tutorials, comparisons, or step-by-steps unless explicitly asked.

Embedding query guidelines:
- Produce a short keyword-style phrase (not a question) for an embedding model.
- 3–8 words; focus on core nouns/entities/terms present or obviously synonymous.
- No extra constraints (dates, versions, locations, names) unless explicitly provided.
- Avoid stopwords, punctuation, quotes, boolean operators, or site/domain operators.
- Keep neutral and general; do not over-specialize.

Respond with strict JSON only:
{{
  "intent": "Find items covering <neutral, high-level topic>",
  "search_query": "embedding-friendly keyword phrase"
}}"""


def build_evaluate_prompt_v2(
    user_query: str,
    current_query: str,
    justifications: list[dict[str, Any]],
    history: list[dict[str, Any]],
    round_idx: int,
    max_rounds: int
) -> str:
    """Build prompt for evaluating results and deciding next action."""
    lines: list[str] = []

    lines.append("Evaluate search results and decide whether to accept or refine the search.\n")
    lines.append(f"USER'S QUESTION: {user_query}")
    lines.append(f"CURRENT SEARCH QUERY: {current_query}")
    lines.append(f"ROUND: {round_idx + 1}/{max_rounds}\n")

    # Show search history
    if history:
        lines.append("SEARCH HISTORY:")
        for h in history:
            lines.append(
                f"  Round {h['round']}: '{h['query']}' → {h['num_justified']} relevant items (avg score: {h.get('avg_score', 0):.2f})"
            )
        lines.append("")

    # Show current results
    justified = [j for j in justifications if j.get("is_just")]
    lines.append(f"CURRENT RESULTS: {len(justified)} relevant items\n")

    if justified:
        lines.append("Top results:")
        for j in sorted(justified, key=lambda x: x.get('score', 0), reverse=True)[:5]:
            score = j.get('score', 0)
            just = j.get('justification', '')
            lines.append(f"  [{score:.2f}] {just}")
        lines.append("")

    # Decision guidance (relaxed)
    lines.append("DECISION GUIDANCE:")
    lines.append("Accept if results look clearly useful based on your judgment.")
    lines.append('For example: if you see several strong, on-topic items with high confidence, accept.')
    lines.append("Refine if results look weak, off-topic, or sparse and there are rounds remaining.")
    lines.append("If nothing is relevant, accept with an empty result set.")
    lines.append("")

    lines.append("If refining, suggest a NEW query that:")
    lines.append("- Uses different keywords or synonyms")
    lines.append("- Attempts to approaches the question from a different angle")
    lines.append("- Is NOT identical to any previous query")
    lines.append("")

    lines.append("Respond with JSON:")
    lines.append('{')
    lines.append('  "should_refine": true/false,')
    lines.append('  "reasoning": "Why this choice is appropriate",')
    lines.append('  "new_query": "Only if refining; otherwise null"')
    lines.append('}')

    return "\n".join(lines)


def build_refine_query_prompt(
    *,
    user_query: str,
    current_query: str,
    justifications: list[dict[str, Any]],
    history_queries: list[str],
    intent: str | None = None,
) -> str:
    """Propose one alternative embedding query, general and not specialized to prior results."""
    lines: list[str] = []
    lines.append("Propose ONE alternative embedding query that broadly rephrases or slightly broadens the current query for dense retrieval.\n")
    lines.append(f"USER'S ORIGINAL QUERY: {user_query}")
    if current_query != user_query:
        lines.append(f"CURRENT SEARCH QUERY: {current_query}")
    if intent:
        lines.append(f"INTENT (high-level): {intent}")
    if history_queries:
        lines.append("\nPREVIOUS QUERIES:")
        for q in history_queries:
            if q:
                lines.append(f"- {q}")

    # Do NOT include or rely on successful-result justifications to avoid specialization
    lines.append("\nConstraints for the new embedding query:")
    lines.append("- This is for an embedding model; return a keyword-style phrase, not a question.")
    lines.append("- Do not specialize based on previously seen results; ignore specific sources, names, or entities.")
    lines.append("- Keep general and neutral; do not add constraints not present in the user's question/intent.")
    lines.append("- Prefer synonyms, re-ordering, or slight broadening/narrowing of core terms.")
    lines.append("- 3–8 words; avoid stopwords, punctuation, quotes, boolean/site operators.")
    lines.append("- Must be different from all previous queries.")

    lines.append("\nRespond with strict JSON only:")
    lines.append('{"new_query": "alternative query"}')
    return "\n".join(lines)

class EvaluateDecision(BaseModel):
    """Structured decision output for evaluation."""
    should_refine: bool = Field(description="Whether to refine with a new query")
    reasoning: str = Field(description="Brief reason for the decision")
    new_query: str | None = Field(default=None, description="New query text when refining; else null")


class StructuredAgent:
    """Two-phase agentic search with intent extraction and iterative refinement.

    Architecture:
      intent → retrieve → justify → evaluate
                 ↑__________________|
                      (if refine)

    The agent extracts search intent first, then iteratively refines queries
    until satisfactory results are found or max rounds reached.
    """

    def __init__(self, *, user_id: str, limit: int = 10) -> None:
        self.user_id = user_id
        self.limit = limit
        self._final_results: list[SearchItem] = []

        # Two models for cost optimization
        self.llm_mini = ChatOpenAI(model="gpt-5-mini", temperature=0)
        self.llm = ChatOpenAI(model="gpt-5", temperature=0)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct the search graph with intent extraction."""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("intent", self._intent_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("justify", self._justify_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("finalize", self._finalize_node)

        # Linear flow with conditional loop
        workflow.set_entry_point("intent")
        workflow.add_edge("intent", "retrieve")
        workflow.add_edge("retrieve", "justify")
        workflow.add_edge("justify", "evaluate")

        # Conditional routing: evaluate decides to refine or finalize
        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {
                "retrieve": "retrieve",
                "finalize": "finalize",
            }
        )

        workflow.add_edge("finalize", END)

        return workflow.compile()

    async def _intent_node(self, state: AgentState) -> dict[str, Any]:
        """Node 0: Extract search intent and generate initial query."""
        # Keep node minimal; astream emits initial status. Capture intent/query for later reasoning.
        events: list[dict[str, Any]] = []

        try:
            prompt = build_intent_prompt(state["user_query"])
            resp = await self.llm_mini.ainvoke(prompt)
            text = getattr(resp, "content", str(resp))

            # Track LLM usage
            try:
                mock_resp = MockResponse(resp, provider="openai", model="gpt-4o-mini")
                await db.create_usage_log(
                    mock_resp,
                    "search.intent_extraction",
                    user_id=state["user_id"],
                    item_id=None,
                )
            except Exception as log_exc:
                # Don't fail the search if logging fails
                print(f"Failed to log intent extraction usage: {log_exc}")

            intent_data = json.loads(text.strip())
            search_query = intent_data.get("search_query", state["user_query"])
            intent = intent_data.get("intent", "")

            return {
                "current_query": search_query,
                "intent": intent,
                "events": events
            }

        except Exception as e:
            # Fallback to original query without exposing internal details
            return {
                "current_query": state["user_query"],
                "intent": "",
                "events": events
            }

    async def _retrieve_node(self, state: AgentState) -> dict[str, Any]:
        """Node 1: Retrieve candidates from database."""
        # Compose user-facing reasoning with intent and round context
        if state["round_idx"] == 0:
            reasoning = (
                f"Intent: '{(state.get('intent') or '').strip() or state['user_query']}'.\n "
                f"Query: '{state['current_query']}'."
            )
            message = "retrieving items..."
        else:
            reasoning = (
                f"Using refined query '{state['current_query']}' to broaden coverage."
            )
            message = "retrieving more items..."
        events = [{
            "kind": "status",
            "message": message,
            "reasoning": reasoning
        }]

        try:
            candidates = await retrieve_candidates(
                user_id=state["user_id"],
                query=state["current_query"],
                top_k=state["limit"],
            )

            if not candidates:
                return {
                    "candidates": [],
                    "events": events,
                }

            # Configure filtering per round
            round_idx = state["round_idx"]
            if round_idx == 0:
                alpha, tau, min_keep = 0.85, 0.20, None
            else:
                alpha, tau, min_keep = 0.60, 0.10, 3

            # Compute metrics for logging
            s_max = max((c.cross_encoder_score or 0.0) for c in candidates) if candidates else 0.0
            cutoff = max(alpha * s_max, tau)
            avg_ce = sum((c.cross_encoder_score or 0.0) for c in candidates) / len(candidates)

            # Apply filtering
            candidates = filter_retrieval(
                candidates,
                limit=state["limit"],
                alpha=alpha,
                tau=tau,
                min_keep=min_keep,
            )

            # On second pass, do not re-consider items already accepted
            if state["round_idx"] > 0 and state.get("justified"):
                seen_ids = set(state["justified"].keys())
                candidates = [c for c in candidates if c.id not in seen_ids]
            # Keep UI concise; skip extra filtering messages

            return {
                "candidates": candidates,
                "events": events
            }

        except Exception as e:
            return {
                "candidates": [],
                "events": events,
                "error": str(e)
            }

    async def _justify_node(self, state: AgentState) -> dict[str, Any]:
        """Node 2: Justify candidates with mini model."""
        if not state["candidates"]:
            return {
                "justifications": [],
                "events": [],
            }

        # Let the user know we're checking candidates against their intent
        if state.get("round_idx", 0) > 0:
            msg = "evaluating more items..."
            rsn = f"Reviewing {len(state['candidates'])} new candidates from the refined query."
        else:
            msg = "evaluating items..."
            rsn = f"Reviewing {len(state['candidates'])} candidates for relevance to '{state['current_query']}'."
        events = [{
            "kind": "status",
            "message": msg,
            "reasoning": rsn
        }]

        try:
            prompt_candidates = [
                {"id": c.id, "summary": c.summary, "chunks": c.preview}
                for c in state["candidates"]
            ]
            prompt = build_justify_prompt(
                state["user_query"],
                state["current_query"],
                prompt_candidates,
                state["history"]
            )

            resp = await self.llm_mini.ainvoke(prompt)
            text = getattr(resp, "content", str(resp))

            # Track LLM usage
            try:
                mock_resp = MockResponse(resp, provider="openai", model="gpt-4o-mini")
                await db.create_usage_log(
                    mock_resp,
                    "search.candidate_justification",
                    user_id=state["user_id"],
                    item_id=None,
                )
            except Exception as log_exc:
                # Don't fail the search if logging fails
                print(f"Failed to log justification usage: {log_exc}")

            justifications = json.loads(text.strip())

            if not isinstance(justifications, list):
                justifications = []

            num_justified = len([j for j in justifications if j.get("is_just")])
            avg_score = (
                sum(j.get("score", 0) for j in justifications if j.get("is_just"))
                / max(1, num_justified)
            )

            # Summarize outcome using LLM-provided justifications where possible

            return {
                "justifications": justifications,
                "events": events
            }

        except Exception as e:
            return {
                "justifications": [],
                "events": events
            }

    async def _evaluate_node(self, state: AgentState) -> dict[str, Any]:
        """Node 3: Evaluate, decide, and do bookkeeping in one step."""
        # If nothing to judge, finalize empty (no extra status here).
        if not state["justifications"]:
            # If we're in the second pass and nothing to justify, surface the nuance to UI
            events: list[dict[str, Any]] = []
            if state["round_idx"] >= 1:
                events.append({
                    "kind": "status",
                    "message": "evaluating more items...",
                    "reasoning": "Found 0 additional items in the second pass."
                })
            return {
                "should_refine": False,
                "reasoning": "No justifications to evaluate",
                "new_query": None,
                "events": events,
                "history": state["history"] + [{
                    "round": state["round_idx"] + 1,
                    "query": state["current_query"],
                    "num_candidates": len(state["candidates"]),
                    "num_justified": 0,
                    "avg_score": 0.0,
                    "decision": "accept",
                    "reasoning": "No justifications"
                }],
                "justified": dict(state["justified"]),
                "round_idx": state["round_idx"] + 1,
                "current_query": state["current_query"],
            }

        # Stats for decision making (avoid surfacing internal counts)
        num_justified = sum(1 for j in state["justifications"] if j.get("is_just"))
        events: list[dict[str, Any]] = []

        # Decision policy: Only ever at most 2 rounds.
        # Refine only after round 1 when <3 justified; otherwise accept.
        should_refine = False
        reasoning = ""
        new_query: str | None = None
        if state["round_idx"] == 0 and num_justified < 3:
            refine_prompt = build_refine_query_prompt(
                user_query=state["user_query"],
                current_query=state["current_query"],
                justifications=state["justifications"],
                history_queries=[h.get("query", "") for h in state["history"]],
                intent=state.get("intent") or None,
            )
            try:
                resp = await self.llm_mini.ainvoke(refine_prompt)
                text = getattr(resp, "content", str(resp))

                # Track LLM usage
                try:
                    mock_resp = MockResponse(resp, provider="openai", model="gpt-4o-mini")
                    await db.create_usage_log(
                        mock_resp,
                        "search.query_refinement",
                        user_id=state["user_id"],
                        item_id=None,
                    )
                except Exception as log_exc:
                    # Don't fail the search if logging fails
                    print(f"Failed to log query refinement usage: {log_exc}")

                parsed = json.loads(text.strip())
                if isinstance(parsed, dict):
                    cand = str(parsed.get("new_query") or "").strip()
                    if cand:
                        should_refine = True
                        new_query = cand
                        reasoning = "<3 relevant items; attempting refined query"
            except Exception:
                should_refine = False
                reasoning = "Refine generation failed; accepting current results"
        else:
            should_refine = False
            reasoning = "Sufficient results or max rounds reached"

        # Prevent duplicate or empty refinements
        previous_queries = {state["current_query"], *[h["query"] for h in state["history"]]}
        if should_refine and (not new_query or new_query in previous_queries):
            should_refine = False
            reasoning = "New query missing or duplicate; accepting"
            new_query = None

        # Build user-facing evaluation reasoning summarizing top LLM justifications
        rel_just = [j for j in state["justifications"] if j.get("is_just")]
        top_reasons = []
        for j in sorted(rel_just, key=lambda x: x.get('score', 0), reverse=True)[:2]:
            r = str(j.get("justification") or "").strip()
            if r:
                top_reasons.append(f"“{r[:120]}”")

        if should_refine and new_query:
            eval_reasoning = (
                f"Only {num_justified} strong matches; exploring a different phrasing to better capture your intent. "
                f"New query: '{new_query}'."
            )
            events.append({
                "kind": "status",
                "message": "trying another query...",
                "reasoning": eval_reasoning
            })
        else:
            eval_reasoning = (
                f"Found {num_justified} relevant items aligned with intent "
                f"'{(state.get('intent') or '').strip() or state['user_query']}'."
            )
            if top_reasons:
                eval_reasoning += " Top signals: " + "; ".join(top_reasons)
            # No extra status; finalize node will emit 'finished'

        # Accumulate best items this round
        justified_acc = dict(state["justified"])
        rel = [j for j in state["justifications"] if j.get("is_just")]
        avg_score = sum(j.get("score", 0.0) for j in rel) / max(1, len(rel))

        new_items_stream: list[dict[str, Any]] = []
        for just in rel:
            iid = str(just.get("id", "")).strip()
            if not iid:
                continue

            support = next((c for c in state["candidates"] if c.id == iid), None)
            if not support:
                continue

            sem = float(support.semantic_score or 0.0)
            ce = float(support.cross_encoder_score or 0.0)
            llm_score = float(just.get("score", 0.7))
            combined = compute_combined_score(
                semantic_score=sem,
                cross_encoder_score=ce,
                llm_score=llm_score
            )

            item = SearchItem(
                id=iid,
                preview=list(support.preview or []),
                title=support.title,
                summary=support.summary,
                justification=str(just.get("justification") or "")[:200],
                is_just=True,
                semantic_score=sem,
                cross_encoder_score=ce,
                llm_score=llm_score,
                combined_score=combined,
                score=combined,
            )

            existing = justified_acc.get(iid)
            if existing is None or combined > float(existing.combined_score or 0.0):
                justified_acc[iid] = item
                new_items_stream.append(item.to_dict())

        # On second pass, report how many additional items were found (concise)
        if state["round_idx"] >= 1:
            addl = len(new_items_stream)
            events.append({
                "kind": "status",
                "message": "evaluating more items...",
                "reasoning": f"Found {addl} additional item{'s' if addl != 1 else ''} in the second pass."
            })

        # History entry
        action = "refine" if should_refine else "accept"
        new_history_entry = {
            "round": state["round_idx"] + 1,
            "query": state["current_query"],
            "num_candidates": len(state["candidates"]),
            "num_justified": len(rel),
            "avg_score": avg_score,
            "decision": action,
            "reasoning": reasoning
        }

        # Stream newly justified items for this round immediately
        if new_items_stream:
            events.append({
                "kind": "items",
                "message": "streaming justified items",
                "round": state["round_idx"] + 1,
                "data": new_items_stream,
            })

        # Prepare next state
        return {
            "should_refine": should_refine,
            "reasoning": reasoning,
            "new_query": new_query if should_refine else None,
            "events": events,
            "justified": justified_acc,
            "history": state["history"] + [new_history_entry],
            "round_idx": state["round_idx"] + 1,
            "current_query": new_query if should_refine else state["current_query"],
        }

    def _should_continue(self, state: AgentState) -> str:
        """Routing logic: refine → retrieve, otherwise → finalize."""
        if state.get("error") or not state["candidates"]:
            return "finalize"
        if state["should_refine"] and state["new_query"]:
            return "retrieve"
        return "finalize"

    async def _finalize_node(self, state: AgentState) -> dict[str, Any]:
        """Node 4: Sort and prepare final results."""
        # Provide a clear final message with a concise summary
        intent_text = (state.get("intent") or "").strip() or state["user_query"]
        events = [{
            "kind": "status",
            "message": "finished",
            "reasoning": f"Search results are complete. Intent: '{intent_text}'."
        }]

        results = list(state["justified"].values())

        if not results:
            self._final_results = []
            events.append({
                "kind": "final",
                "message": "no-results",
                "reason": "no-justified-items",
                "data": []
            })
            return {"events": events}

        # Sort by combined score
        results.sort(key=lambda i: (i.combined_score or 0.0), reverse=True)
        self._final_results = results[: state["limit"]]
        final_results = [r.to_dict() for r in self._final_results]

        events.append({
            "kind": "final",
            "message": "success",
            "data": final_results
        })

        return {"events": events}

    async def astream(self, *, query: str) -> AsyncGenerator[dict[str, Any], None]:
        """Stream search results through LangGraph execution."""
        user_query = (query or "").strip()

        if not user_query:
            yield {"kind": "final", "message": "no-results", "reason": "empty-query", "data": []}
            return

        try:
            # Emit an immediate, user-friendly status for responsiveness
            yield {
                "kind": "status",
                "message": "refining query...",
                "reasoning": "Understanding your intent and optimizing the query."
            }

            # Initialize state
            initial_state: AgentState = {
                "user_query": user_query,
                "user_id": self.user_id,
                "limit": self.limit,
                "intent": "",
                "current_query": user_query,  # Will be replaced by intent node
                "round_idx": 0,
                "max_rounds": 2,
                "candidates": [],
                "justifications": [],
                "justified": {},
                "history": [],
                "should_refine": False,
                "reasoning": "",
                "new_query": None,
                "events": [],
                "error": None
            }

            # Execute graph and stream events
            async for chunk in self.graph.astream(initial_state):
                for node_name, node_output in chunk.items():
                    if "events" in node_output and node_output["events"]:
                        for event in node_output["events"]:
                            yield event

        except Exception as exc:
            traceback.print_exc()
            yield {
                "kind": "final",
                "message": "error",
                "reason": f"{exc.__class__.__name__}: {str(exc)}",
                "data": []
            }

    def get_results(self) -> list[SearchItem]:
        """Get final results after streaming completes."""
        return self._final_results
