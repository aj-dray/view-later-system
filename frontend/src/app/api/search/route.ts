export const dynamic = "force-dynamic";
export const revalidate = 0;
export const fetchCache = "force-no-store";

import { NextResponse } from "next/server";

import {
  searchItems,
  type SearchMode,
  type SearchResponse,
} from "@/app/_lib/search";
import { MAX_SEARCH_LIMIT } from "@/app/_lib/search-config";

function sanitiseMode(value: string | null): SearchMode {
  if (value === "semantic" || value === "agentic") {
    return value;
  }
  return "lexical";
}

function sanitiseLimit(value: string | null): number {
  if (!value) {
    return MAX_SEARCH_LIMIT;
  }
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return MAX_SEARCH_LIMIT;
  }
  return Math.min(Math.max(parsed, 1), MAX_SEARCH_LIMIT);
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get("query")?.trim() ?? "";
  const modeParam = searchParams.get("mode");
  const limitParam = searchParams.get("limit");

  if (!query) {
    return NextResponse.json({ results: [] satisfies SearchResponse["results"], agent: null });
  }

  const mode = sanitiseMode(modeParam);
  const limit = sanitiseLimit(limitParam);

  if (mode === "agentic") {
    return NextResponse.json(
      { error: "Agentic search requires streaming endpoint" },
      { status: 400 },
    );
  }

  try {
    const data = await searchItems({
      query,
      mode,
      limit,
    });

    return NextResponse.json(data);
  } catch (error) {
    console.error("Failed to perform search", { error, mode });

    if (error instanceof Error && error.message.includes("401")) {
      return NextResponse.json(
        { error: "Authentication required" },
        { status: 401 },
      );
    }

    return NextResponse.json(
      { error: "Unable to complete search" },
      { status: 500 },
    );
  }
}
