export const dynamic = "force-dynamic";
export const revalidate = 0;
export const fetchCache = "force-no-store";

import { NextResponse } from "next/server";

import { authedFetch } from "@/app/_lib/items";
import { MAX_SEARCH_LIMIT } from "@/app/_lib/search-config";

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
  const query = searchParams.get("query")?.trim();
  const limitParam = searchParams.get("limit");

  if (!query) {
    return NextResponse.json(
      { error: "Query parameter is required" },
      { status: 400 },
    );
  }

  const limit = sanitiseLimit(limitParam);

  const backendParams = new URLSearchParams();
  backendParams.set("query", query);
  backendParams.set("mode", "agentic");
  backendParams.set("limit", String(limit));
  backendParams.append("columns", "title");
  backendParams.append("columns", "summary");

  // Allow aborting the upstream backend request when the client cancels.
  const upstreamAbort = new AbortController();
  const backendResponse = await authedFetch(
    `/items/search?${backendParams.toString()}`,
    {
      headers: {
        Accept: "text/event-stream",
      },
      signal: upstreamAbort.signal,
    },
  );

  if (!backendResponse.ok || !backendResponse.body) {
    const text = await backendResponse.text().catch(() => null);
    return NextResponse.json(
      {
        error:
          text && text.trim().length > 0
            ? text
            : `Backend search returned ${backendResponse.status}`,
      },
      { status: backendResponse.status || 500 },
    );
  }

  const reader = backendResponse.body.getReader();

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      async function pump() {
        try {
          while (true) {
            const { value, done } = await reader.read();
            if (done) {
              controller.close();
              break;
            }
            if (value) {
              controller.enqueue(value);
            }
          }
        } catch (error) {
          controller.error(error);
        } finally {
          reader.releaseLock?.();
        }
      }

      pump();
    },
    cancel() {
      try {
        // Abort the upstream fetch to the backend so it can stop the agent.
        upstreamAbort.abort();
      } catch {}
      reader.cancel().catch(() => {
        // Swallow cancellation errors
      });
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
