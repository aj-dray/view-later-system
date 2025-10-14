"""Utilities for generating derived data for items."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from enum import Enum
from aglib import Client
from pydantic import BaseModel

from .. import database as db


class ItemType(Enum):
    ARTICLE = "article"
    VIDEO = "video"
    PAPER = "paper"
    PODCAST = "podcast"
    POST = "post"
    NEWSLETTER = "newsletter"
    OTHER = "other"


class SummaryData(BaseModel):
    type: ItemType
    summary: str
    expiry_score: float


def _build_generation_context(item: dict[str, Any]) -> str:
    parts: list[str] = []
    if item.get("title"):
        parts.append(f"Title: {item['title']}")
    if item.get("source_site"):
        parts.append(f"Source: {item['source_site']}")
    if item.get("publication_date"):
        parts.append(f"Published: {item['publication_date']}")
    parts.append(f"URL: {item.get('canonical_url') or item.get('url', '')}")
    if item.get("content_markdown"):
        parts.append(f"\nArticle Content:\n{item['content_markdown']}")
    return "\n".join(parts)


def _request_summary(item: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    prompt = (
        """
        You are an assistant helping to index items to be saved in a database.
        Using the metadata and raw text content provided, your task is to fill in other properties for that article.
        These properties you must generate values for are:
            - Type: What type of media best describes it from: "article" (news or educational article), "video", "paper" (scientific paper), "podcast" (from podcast platform), "post" (social media or blog), "newsletter" (email newsletter), "other" (not from other categories).
            - Summary: 1-2 sentence summary of the article content.
            - Expiry Score: a float between 0 and 1 to capture how quickly this content will decay in relevance. 1 indicates very fast decay in relevance (i.e. a news article).
        """
    )

    llm = Client.completion(provider="mistral", model="mistral-medium-latest")
    response = llm.request(
        system_prompt=prompt,
        messages=[{"role": "user", "content": _build_generation_context(item)}],
        response_format=SummaryData,
    )

    summary_data = SummaryData.model_validate(json.loads(response.content))
    payload = {
        "type": summary_data.type.value,
        "summary": summary_data.summary,
        "expiry_score": summary_data.expiry_score,
    }
    return response, payload


async def generate_data(item: dict[str, Any], user_id: str) -> dict[str, Any]:
    try:
        response, payload = await asyncio.to_thread(_request_summary, item)
    except Exception as exc:
        raise ValueError(f"Failed to generate summary: {exc}") from exc

    await db.create_usage_log(
        response,
        "completion.item_summary",
        user_id=user_id,
        item_id=item.get("id"),
    )

    payload.update(
        {
            "server_status": "summarised",
            "server_status_at": datetime.now(),
        }
    )
    return payload
