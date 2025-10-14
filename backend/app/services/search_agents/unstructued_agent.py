from __future__ import annotations

import asyncio
import json
import logging
from operator import add
import re
from typing import Any, Annotated, AsyncGenerator, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from ..retrieval import retrieve_candidates, filter_retrieval
from .base import SearchItem
from ... import database as db

logger = logging.getLogger(__name__)


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


class FreeformState(TypedDict):
    """State for the freeform LangGraph agent."""

    # Core context
    user_id: str
    user_query: str
    limit: int
    default_threshold: float
    loop_count: int

    # Conversation (for tool calling)
    messages: list

    # Streaming output
    events: Annotated[list[dict[str, Any]], add]

    # Results tracking: {item_id: {"item": {...}, "sent": bool}}
    results: dict[str, dict[str, Any]]


def _extract_reasoning(message: AIMessage) -> str:
    """Pull freeform reasoning from the model response."""
    content = message.content
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, str) and chunk.strip():
                parts.append(chunk.strip())
            elif isinstance(chunk, dict):
                text = str(chunk.get("text", "")).strip()
                if text:
                    parts.append(text)
        if parts:
            return " ".join(parts)
    tool_calls = getattr(message, "tool_calls", None) or []
    summaries: list[str] = []
    for call in tool_calls:
        name = str(call.get("name") or "").strip() or "tool"
        args = call.get("args")
        try:
            arg_text = json.dumps(args, ensure_ascii=False)
        except Exception:
            arg_text = str(args)
        summaries.append(f"{name}: {arg_text}")
    return "; ".join(summaries)


def _build_system_instructions(limit: int, default_threshold: float) -> str:
    """System prompt guiding a freeform retrieval + send workflow via tools.

    The model should:
    - Choose one or both search tools (semantic vs. lexical) as needed
    - Decide whether to use quality filtering or get top results regardless
    - After searching, reason about each item's relevance and approve/reject ids
    - Merge and deduplicate results, then send approved ids to the client
    - Stream partial batches if helpful; send a final batch when done
    - Only respond by calling tools; do not produce natural language
    """
    return (
        "You are a retrieval agent for a personal knowledge system. "
        "Use the provided tools to surface the most relevant saved items and stream them to the client. "
        f"Return at most {limit} items overall. "
        "\n\n"
        "SEARCH TOOLS:\n"
        "- semantic_search(query, use_threshold): Embedding-based search. Query should be concise nouns/noun phrases.\n"
        "- lexical_search(query, use_threshold): Keyword-based search. Query should be literal keywords/phrases.\n"
        "- use_threshold=true: Apply quality filtering (may return 0 results if nothing is relevant)\n"
        "- use_threshold=false: Return top 5 items regardless of quality (use this if filtering returns nothing)\n"
        "\n"
        "WORKFLOW:\n"
        "1. Start with use_threshold=true to get high-quality results\n"
        "2. If you get 0 results, try use_threshold=false to get the best available items\n"
        "3. You may try different query formulations or both search types\n"
        "4. Each turn: max 3 tool calls, so prioritize strategically\n"
        "5. After receiving results, evaluate each item's preview/summary\n"
        "6. Call send_results with approved ids (only from current search results)\n"
        "7. When done, call send_results with final=true\n"
        "\n"
        "Only call tools; do not write natural language responses."
    )


class UnstructuedAgent:
    """Flexible LangGraph-based agent using GPT-5 and explicit tools.

    Tools available to the model:
    - semantic_search(query: str, threshold: float) → items
    - lexical_search(query: str, threshold: float) → items
    - send_results(ids: list[str], final: bool = False, message: str | None = None)

    This is a freeform variant of LangGraphSearchAgent that delegates when and
    how to search and send to the model via tools.
    """

    def __init__(self, *, user_id: str, limit: int = 10, default_threshold: float = 0.20) -> None:
        self.user_id = user_id
        self.limit = max(1, min(int(limit or 10), 50))
        self.default_threshold = float(default_threshold)

        # Strong model; tool decisions + reasoning handled internally by the model
        self.llm = ChatOpenAI(model="gpt-5", temperature=0)
        self.llm_summary = ChatOpenAI(model="gpt-5-mini", temperature=0)

        # Tool registry (name -> callable) and tool specs for model binding
        self._tool_fns: dict[str, Any]
        self._tools: list[Any]
        self._tool_fns, self._tools = self._build_tools()

        # Build graph
        self.graph = self._build_graph()

        # Final results cache (optional for external callers)
        self._final_results: list[dict[str, Any]] = []

    # --- Tools ---
    def _build_tools(self) -> tuple[dict[str, Any], list[Any]]:
        user_id = self.user_id
        limit = self.limit

        class SemanticSearchInput(BaseModel):
            """Input schema for semantic search."""
            query: str = Field(description="Embedding-friendly search text (concise nouns / concepts)")
            use_threshold: bool = Field(
                default=True,
                description="If true, apply quality filtering (may return 0 results). If false, return top 5 items regardless of quality."
            )

        @tool("semantic_search", args_schema=SemanticSearchInput)
        async def semantic_search_tool(query: str, use_threshold: bool = True) -> str:
            """Retrieve items via semantic + reranking. Returns a JSON string with an array of items, each like: {"id": str, "score": float, "preview": list[str], "title"?: str, "summary"?: str}"""
            top_k = max(1, min(limit, 10))
            candidates: list[SearchItem] = await retrieve_candidates(
                user_id=user_id,
                query=query,
                top_k=top_k,
            )

            if use_threshold:
                # Apply strict filtering: alpha=0.85, tau=0.20, may return 0 items
                filtered = filter_retrieval(
                    candidates,
                    limit=top_k,
                    alpha=0.85,
                    tau=0.20,
                    min_keep=None,
                )
                print(f"    [semantic_search] use_threshold=True: {len(candidates)} → {len(filtered)} items")
            else:
                # No filtering: return top 5 items regardless of quality
                filtered = filter_retrieval(
                    candidates,
                    limit=min(5, top_k),
                    alpha=0.0,
                    tau=0.0,
                    min_keep=min(5, len(candidates)),
                )
                print(f"    [semantic_search] use_threshold=False: returning top {len(filtered)} items (no filtering)")

            from ..searching import parse_search_results  # local import to avoid cycle

            payload = parse_search_results(filtered, columns=("title", "summary"))
            return json.dumps(payload, ensure_ascii=False)

        class LexicalSearchInput(BaseModel):
            """Input schema for lexical search."""
            query: str = Field(description="Literal keywords / phrases to match in documents")
            use_threshold: bool = Field(
                default=True,
                description="If true, apply quality filtering (may return 0 results). If false, return top 5 items regardless of quality."
            )

        @tool("lexical_search", args_schema=LexicalSearchInput)
        async def lexical_search_tool(query: str, use_threshold: bool = True) -> str:
            """Retrieve items via keyword/lexical matching using chunk-level ranking. Returns a JSON string with an array of items, each like: {"id": str, "score": float, "preview": list[str], "title"?: str, "summary"?: str}"""
            from ..searching import lexical as lexical_search  # local import to avoid cycle

            items = await lexical_search(
                user_id=user_id,
                query=query,
                limit=limit,
                columns=("title", "summary"),
            )

            if use_threshold:
                # Apply filtering with threshold 0.10
                threshold = 0.10
                kept = [it for it in items if float(it.get("score", 0.0)) >= threshold]
                print(f"    [lexical_search] use_threshold=True: {len(items)} → {len(kept)} items (threshold={threshold})")
            else:
                # No filtering: return top 5 items
                kept = items[:min(5, len(items))]
                print(f"    [lexical_search] use_threshold=False: returning top {len(kept)} items (no filtering)")

            return json.dumps(kept[: limit], ensure_ascii=False)

        class SendResultsInput(BaseModel):
            """Input schema for sending results."""
            ids: list[str] = Field(description="List of item ids to send (must be from prior search tool results)")
            final: bool = Field(default=False, description="Set true when you are done sending results")
            message: str | None = Field(default=None, description="Optional short status message to show the user")

        @tool("send_results", args_schema=SendResultsInput)
        def send_results_tool(ids: list[str], final: bool = False, message: str | None = None) -> str:
            """Send a batch of relevant result ids to the client UI. Returns 'sent'."""
            # Actual sending is handled by the executor; this is a schema shim.
            return "sent"

        tool_fns: dict[str, Any] = {
            "semantic_search": semantic_search_tool,
            "lexical_search": lexical_search_tool,
            "send_results": send_results_tool,
        }
        tools = [semantic_search_tool, lexical_search_tool, send_results_tool]
        return tool_fns, tools

    async def _summarize_status(self, message: AIMessage) -> str:
        """Create a concise user-facing summary of the model's latest reasoning."""
        raw_reasoning = _extract_reasoning(message)
        if not raw_reasoning:
            return ""

        prompt = (
            "You are narrating progress for a search assistant. "
            "Write ONE short present-tense sentence (≤120 characters) describing what the agent will do next for the user. "
            "Be concrete (e.g. 'Refining query with synonyms'), avoid tool jargon, and never mention JSON. "
            "Reasoning:\n"
            f"{raw_reasoning}\n\nSummary:"
        )

        try:
            resp = await self.llm_summary.ainvoke(prompt)
            text = getattr(resp, "content", str(resp))

            # Track LLM usage
            try:
                mock_resp = MockResponse(resp, provider="openai", model="gpt-4o-mini")
                # Note: We don't have user_id in this context, so we'll skip tracking for this summarization call
                # This is a minor internal call and less critical to track
            except Exception:
                pass

            if isinstance(text, list):
                text = " ".join(str(part).strip() for part in text if str(part).strip())
            summary = str(text).strip()
            if len(summary) > 120:
                summary = summary[:117].rstrip() + "..."
            return summary
        except Exception:
            # Fall back to truncated raw reasoning
            fallback = raw_reasoning.strip()
            if len(fallback) > 120:
                fallback = fallback[:117].rstrip() + "..."
            return fallback

    # --- Graph construction ---
    def _build_graph(self):
        workflow = StateGraph(FreeformState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("finalize", self._finalize_node)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._route_from_agent, {"act": "act", "finalize": "finalize"})
        workflow.add_edge("act", "agent")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    # --- Nodes ---
    async def _agent_node(self, state: FreeformState) -> dict[str, Any]:
        """Let the LLM decide which tools to call."""
        messages = state["messages"]
        loop_count = state.get("loop_count", 0)
        results = state.get("results", {})

        logger.info(f"[UnstructuredAgent] _agent_node: loop_count={loop_count}, results_count={len(results)}")
        print(f"[Node: agent] loop_count={loop_count}, results={len(results)}")

        # On final turn, don't bind tools (force completion)
        # Allow 6 loops total (0,1,2,3,4,5) so agent has enough turns even with conservative tool calling
        is_final_turn = loop_count >= 5
        llm = self.llm if is_final_turn else self.llm.bind_tools(self._tools)

        print(f"[Node: agent] Calling LLM (is_final_turn={is_final_turn})...")
        ai = await llm.ainvoke(messages)
        print(f"[Node: agent] LLM returned")

        # Track LLM usage
        try:
            mock_resp = MockResponse(ai, provider="openai", model="gpt-4o" if is_final_turn else "gpt-4o")
            await db.create_usage_log(
                mock_resp,
                "search.unstructured_agent",
                user_id=state["user_id"],
                item_id=None,
            )
        except Exception as log_exc:
            # Don't fail the search if logging fails
            print(f"Failed to log unstructured agent usage: {log_exc}")

        tool_calls = getattr(ai, "tool_calls", [])
        logger.info(f"[UnstructuredAgent] _agent_node: got {len(tool_calls)} tool calls")

        # Generate status message
        has_results = any(r["sent"] for r in results.values())
        if is_final_turn:
            message = "finishing up"
        elif has_results:
            message = "finding more items"
        else:
            message = "finding items"

        status_event: dict[str, Any] = {
            "kind": "status",
            "message": message,
        }

        reasoning = await self._summarize_status(ai)
        if reasoning:
            status_event["reasoning"] = reasoning

        return {
            "messages": messages + [ai],
            "events": [status_event],
            "loop_count": loop_count + 1,
        }

    def _route_from_agent(self, state: FreeformState) -> str:
        """Route to act if there are tool calls and we haven't hit the limit."""
        # Max 6 loops (0,1,2,3,4,5)
        if state.get("loop_count", 0) >= 6:
            return "finalize"

        msgs = state.get("messages", [])
        if not msgs:
            return "finalize"

        last = msgs[-1]
        if not isinstance(last, AIMessage):
            return "finalize"

        tool_calls = getattr(last, "tool_calls", None)
        if not tool_calls:
            return "finalize"

        # Check if send_results with final=true was called
        for call in tool_calls:
            if call.get("name") == "send_results" and call.get("args", {}).get("final"):
                return "finalize"

        return "act"

    async def _execute_search_tool(
        self,
        call: dict[str, Any],
        results: dict[str, dict[str, Any]],
    ) -> tuple[ToolMessage, list[dict[str, Any]]]:
        """Execute a search tool call and return new items."""
        name = call.get("name", "")
        args = call.get("args", {})
        fn = self._tool_fns[name]

        print(f"    [Tool Call] {name}(query='{args.get('query', '')}', use_threshold={args.get('use_threshold', True)})")

        # Execute tool and parse response
        res_json = await fn.ainvoke(args)
        try:
            items = json.loads(res_json)
            if not isinstance(items, list):
                items = []
        except Exception:
            items = []

        # Track new items (haven't seen before)
        new_items: list[dict[str, Any]] = []
        for it in items:
            iid = str(it.get("id", "")).strip()
            if not iid:
                continue

            # Only add if we haven't seen this item yet
            if iid not in results:
                results[iid] = {"item": it, "sent": False}
                new_items.append(it)
            else:
                # Update if score is higher
                existing_score = float(results[iid]["item"].get("score", 0.0))
                new_score = float(it.get("score", 0.0))
                if new_score > existing_score:
                    results[iid]["item"] = it

        # Return message for the model
        response = json.dumps(new_items, ensure_ascii=False)
        tool_msg = ToolMessage(
            tool_call_id=call.get("id"),
            content=f"Found {len(new_items)} new items (total: {len(results)}):\n{response}",
        )
        return tool_msg, new_items

    async def _execute_send_tool(
        self,
        call: dict[str, Any],
        results: dict[str, dict[str, Any]],
    ) -> tuple[ToolMessage, list[dict[str, Any]], bool]:
        """Execute send_results tool call. Returns (tool_msg, sent_items, is_final)."""
        args = call.get("args", {})
        ids = args.get("ids", [])
        final = bool(args.get("final", False))

        print(f"    [Tool Call] send_results(ids={len(ids) if isinstance(ids, list) else 0}, final={final})")

        # Validate and collect items to send
        sent_items: list[dict[str, Any]] = []
        for iid in ids if isinstance(ids, list) else []:
            iid = str(iid).strip()
            if not iid or iid not in results:
                continue

            # Skip if already sent
            if results[iid]["sent"]:
                continue

            # Mark as sent and add to batch
            results[iid]["sent"] = True
            sent_items.append(results[iid]["item"])

        print(f"    [send_results] Sending {len(sent_items)} items to client")

        tool_msg = ToolMessage(
            tool_call_id=call.get("id"),
            content=f"Sent {len(sent_items)} items; final={final}",
        )
        return tool_msg, sent_items, final

    async def _act_node(self, state: FreeformState) -> dict[str, Any]:
        """Execute tool calls from the agent."""
        msgs = list(state.get("messages", []))
        if not msgs:
            return {"messages": msgs, "events": []}

        last = msgs[-1]
        if not isinstance(last, AIMessage):
            return {"messages": msgs, "events": []}

        tool_calls = list(getattr(last, "tool_calls", []) or [])
        if not tool_calls:
            return {"messages": msgs, "events": []}

        logger.info(f"[UnstructuredAgent] _act_node: executing {len(tool_calls)} tool calls")

        results = dict(state.get("results", {}))
        events: list[dict[str, Any]] = []

        # Process up to 3 tool calls per turn
        for idx, call in enumerate(tool_calls[:3]):
            name = call.get("name", "")
            logger.info(f"[UnstructuredAgent] _act_node: tool {idx+1}: {name}")

            if name in ("semantic_search", "lexical_search"):
                tool_msg, new_items = await self._execute_search_tool(call, results)
                msgs.append(tool_msg)

            elif name == "send_results":
                tool_msg, sent_items, is_final = await self._execute_send_tool(call, results)
                msgs.append(tool_msg)

                # Stream sent items to client
                if sent_items:
                    events.append({
                        "kind": "items",
                        "message": "items",
                        "data": sent_items[: self.limit],
                    })

                # Handle final flag
                if is_final:
                    all_sent = [r["item"] for r in results.values() if r["sent"]]
                    all_sent.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
                    final_items = all_sent[: self.limit]

                    events.append({
                        "kind": "status",
                        "message": "finished",
                        "reasoning": f"Found {len(final_items)} relevant items." if final_items else "No relevant items found.",
                    })
                    events.append({
                        "kind": "final",
                        "message": "success" if final_items else "no-results",
                        "data": final_items,
                    })
                    return {"messages": msgs, "events": events, "results": results}

            else:
                # Unknown tool
                msgs.append(ToolMessage(tool_call_id=call.get("id"), content="Unknown tool"))

        return {"messages": msgs, "events": events, "results": results}

    async def _finalize_node(self, state: FreeformState) -> dict[str, Any]:
        """Finalize and send results if not already done."""
        results = state.get("results", {})

        # Collect all sent items
        sent_items = [r["item"] for r in results.values() if r["sent"]]
        sent_items.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        final_items = sent_items[: state["limit"]]

        events = [
            {
                "kind": "status",
                "message": "finished",
                "reasoning": f"Found {len(final_items)} relevant items." if final_items else "No relevant items found.",
            },
            {
                "kind": "final",
                "message": "success" if final_items else "no-results",
                "data": final_items,
            },
        ]
        return {"events": events}

    # --- Public API ---
    async def astream(self, *, query: str) -> AsyncGenerator[dict[str, Any], None]:
        """Run the freeform agent and stream UI events."""
        q = (query or "").strip()
        print(f"\n{'='*60}")
        print(f"[UnstructuredAgent] Starting search: '{q}'")
        print(f"{'='*60}\n")
        logger.info(f"[UnstructuredAgent] astream: starting with query='{q}'")

        if not q:
            yield {"kind": "final", "message": "no-results", "reason": "empty-query", "data": []}
            return

        initial_messages = [
            SystemMessage(content=_build_system_instructions(self.limit, self.default_threshold)),
            HumanMessage(content=q),
        ]

        state: FreeformState = {
            "user_id": self.user_id,
            "user_query": q,
            "limit": self.limit,
            "default_threshold": self.default_threshold,
            "messages": initial_messages,
            "events": [],
            "results": {},
            "loop_count": 0,
        }

        # The function must yield at least once to be recognized as an async generator
        # We'll yield events from the graph directly
        print("[astream] About to start graph.astream...")
        try:
            async for chunk in self.graph.astream(state):
                print(f"[astream] Got chunk: {list(chunk.keys())}")
                for node_name, out in chunk.items():
                    print(f"[Node: {node_name}] Executed")

                    # Store final results if we see them in the state
                    if "results" in out:
                        results_count = len(out["results"])
                        sent_count = sum(1 for r in out["results"].values() if r["sent"])
                        print(f"  → Results: {results_count} total, {sent_count} sent")
                        self._store_final_results(out["results"], self.limit)

                    # Yield each event from the graph
                    events = out.get("events", []) or []
                    for ev in events:
                        kind = ev.get("kind")
                        msg = ev.get("message")
                        print(f"  → Event: {kind}={msg}")
                        if kind == "items":
                            print(f"     Items: {len(ev.get('data', []))}")
                        yield ev
        except Exception as exc:  # pragma: no cover - defensive stream handling
            print(f"[ERROR] {exc.__class__.__name__}: {exc}")
            import traceback
            traceback.print_exc()
            yield {
                "kind": "final",
                "message": "error",
                "reason": f"{exc.__class__.__name__}: {str(exc)}",
                "data": [],
            }

    def get_results(self) -> list[dict[str, Any]]:
        """Best-effort final results after astream completes (already parsed)."""
        return self._final_results

    def _store_final_results(self, results: dict[str, dict[str, Any]], limit: int) -> None:
        """Store final results for external retrieval."""
        sent_items = [r["item"] for r in results.values() if r["sent"]]
        sent_items.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        self._final_results = sent_items[:limit]
