"use server";

import { fetchItems, authedFetch } from "@/app/_lib/items";
import type { ItemSummary } from "@/app/_lib/items";
import { MAX_SEARCH_LIMIT } from "@/app/_lib/search-config";

export type SearchMode = "lexical" | "semantic" | "agentic";
export type SearchResult = {
  item: ItemSummary;
  preview: string | null;
  score: number | null;
  distance: number | null;
};

export type AgentTraceStepKind = "thought" | "action" | "observation" | "final";

export type AgentTraceStep = {
  kind: AgentTraceStepKind;
  message: string;
  data?: Record<string, unknown> | null;
};

export type AgentSearchMetadata = {
  summary: string | null;
  backend: "langchain";
  steps: AgentTraceStep[];
};

export type SearchResponse = {
  results: SearchResult[];
  agent: AgentSearchMetadata | null;
};

type PerformSearchOptions = {
  query: string;
  mode?: SearchMode;
  limit?: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function toStringOrNull(value: unknown): string | null {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  return null;
}

function toNumberOrNull(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    const parsed = Number(trimmed);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

type ParsedRawResult = {
  id: string;
  preview: string | null;
  score: number | null;
  distance: number | null;
};

function parseRawResult(value: unknown): ParsedRawResult | null {
  if (!isRecord(value)) {
    return null;
  }

  const id = toStringOrNull(value.id);
  if (!id) {
    return null;
  }

  // preview can be string or string[] (preferred). If array, use first element.
  let preview: string | null = null;
  const rawPreview = value.preview as unknown;
  if (Array.isArray(rawPreview)) {
    const first = rawPreview.find((entry) => typeof entry === "string" && entry.trim().length > 0);
    preview = first ? String(first) : null;
  } else {
    preview = toStringOrNull(rawPreview) ?? toStringOrNull(value.content_text) ?? null;
  }
  const score = toNumberOrNull(value.score);
  const distance = toNumberOrNull(value.distance);

  return { id, preview, score, distance };
}

function sanitiseLimit(limit: number | undefined): number {
  if (!Number.isFinite(limit ?? NaN)) {
    return MAX_SEARCH_LIMIT;
  }
  const value = Math.trunc(limit as number);
  if (!Number.isFinite(value)) {
    return MAX_SEARCH_LIMIT;
  }
  return Math.min(Math.max(value, 1), MAX_SEARCH_LIMIT);
}

function parseAgentStep(value: unknown): AgentTraceStep | null {
  if (!isRecord(value)) {
    return null;
  }

  const kindValue = toStringOrNull(value.kind);
  if (kindValue !== "thought" && kindValue !== "action" && kindValue !== "observation" && kindValue !== "final") {
    return null;
  }

  const message = toStringOrNull(value.message);
  if (!message) {
    return null;
  }

  const dataValue = value.data;
  const data = isRecord(dataValue) ? (dataValue as Record<string, unknown>) : null;

  return {
    kind: kindValue,
    message,
    data,
  };
}

function parseAgentMetadata(value: unknown): AgentSearchMetadata | null {
  if (!isRecord(value)) {
    return null;
  }

  const summary = toStringOrNull(value.summary);

  const stepsRaw = Array.isArray(value.steps) ? value.steps : [];
  const steps = stepsRaw
    .map((entry) => parseAgentStep(entry))
    .filter((entry): entry is AgentTraceStep => Boolean(entry));

  if (summary === null && steps.length === 0) {
    return null;
  }

  return {
    summary,
    backend: "langchain",
    steps,
  };
}

export async function searchItems({
  query,
  mode = "lexical",
  limit = MAX_SEARCH_LIMIT,
}: PerformSearchOptions): Promise<SearchResponse> {
  const trimmedQuery = query.trim();
  if (!trimmedQuery) {
    return { results: [], agent: null };
  }

  if (mode === "agentic") {
    throw new Error("Agentic search must be streamed; use the agentic API route.");
  }

  const safeLimit = sanitiseLimit(limit);
  const params = new URLSearchParams();
  params.set("query", trimmedQuery);
  params.set("mode", mode === "semantic" ? "semantic" : "lexical");
  params.set("limit", String(safeLimit));

  const response = await authedFetch(`/items/search?${params.toString()}`, {
    cache: "no-store",
    next: { revalidate: 0 },
  });

  if (response.status === 401) {
    throw new Error("Search request failed: Unauthorized. Please log in again.");
  }

  if (!response.ok) {
    let errorDetail = `Search request failed with status ${response.status}`;
    try {
      const errorBody = await response.json();
      if (errorBody && typeof errorBody.detail === "string") {
        errorDetail = errorBody.detail;
      }
    } catch {
      // Ignore JSON parse errors, use default message
    }
    throw new Error(errorDetail);
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(`Search failed: Invalid response format from server. ${error instanceof Error ? error.message : ""}`);
  }
  if (!isRecord(payload)) {
    throw new Error("Unexpected payload when performing search");
  }

  const agentMetadata = parseAgentMetadata(payload.agent);
  const rawResults = Array.isArray(payload.results)
    ? payload.results
    : [];

  const parsed = rawResults
    .map((entry) => parseRawResult(entry))
    .filter((entry): entry is ParsedRawResult => Boolean(entry));

  if (parsed.length === 0) {
    return { results: [], agent: agentMetadata };
  }

  const uniqueIds: string[] = [];
  const seenIds = new Set<string>();
  parsed.forEach((entry) => {
    if (!seenIds.has(entry.id)) {
      seenIds.add(entry.id);
      uniqueIds.push(entry.id);
    }
  });

  if (uniqueIds.length === 0) {
    return { results: [], agent: agentMetadata };
  }

  const items = await fetchItems({
    filters: [
      {
        column: "id",
        operator: "IN",
        value: uniqueIds,
      },
    ],
    limit: Math.max(uniqueIds.length, 1),
  });

  const itemById = new Map<string, ItemSummary>(
    items.map((item) => [item.id, item]),
  );

  const results: SearchResult[] = [];
  const usedIds = new Set<string>();
  parsed.forEach((entry) => {
    const item = itemById.get(entry.id);
    if (!item || usedIds.has(entry.id)) {
      return;
    }
    usedIds.add(entry.id);
    results.push({
      item,
      preview: entry.preview,
      score: entry.score,
      distance: entry.distance,
    });
  });

  return { results, agent: agentMetadata };
}
