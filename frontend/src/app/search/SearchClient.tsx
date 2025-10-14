"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

import ItemCard from "@/app/_components/ItemCard";
import type {
  SearchMode,
  SearchResult,
  SearchResponse,
} from "@/app/_lib/search";
import { MAX_SEARCH_LIMIT } from "@/app/_lib/search-config";
import type { ItemSummary } from "@/app/_lib/items";

type SearchClientProps = {
  initialQuery: string;
  initialMode: SearchMode;
  initialData: SearchResponse;
};

type SearchApiResponse = SearchResponse;

type AgentStreamEvent = {
  kind?: string;
  message?: string;
  reasoning?: string;
  data?: unknown;
  reason?: string;
};

type AgentStreamItem = {
  id: string;
  preview: string | null;
  score: number | null;
  distance: number | null;
};

const STREAM_RESULT_LIMIT = MAX_SEARCH_LIMIT;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function toStringOrNull(value: unknown): string | null {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
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

function parsePreview(value: unknown): string | null {
  if (Array.isArray(value)) {
    const candidate = value.find(
      (entry) => typeof entry === "string" && entry.trim().length > 0,
    );
    return candidate ? String(candidate) : null;
  }
  return toStringOrNull(value);
}

function parseItemsPayload(value: unknown): AgentStreamItem[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const items: AgentStreamItem[] = [];
  for (const entry of value) {
    if (!isRecord(entry)) {
      continue;
    }
    const id = toStringOrNull(entry.id);
    if (!id) {
      continue;
    }
    items.push({
      id,
      preview: parsePreview(entry.preview ?? entry.content_text),
      score: toNumberOrNull(entry.score),
      distance: toNumberOrNull(entry.distance),
    });
  }
  return items;
}

function scoreValue(result: SearchResult): number {
  const score = result.score;
  return typeof score === "number" && Number.isFinite(score)
    ? score
    : Number.NEGATIVE_INFINITY;
}

function sortResultsByScore(results: SearchResult[]): SearchResult[] {
  return [...results].sort((a, b) => scoreValue(b) - scoreValue(a));
}

async function fetchItemSummaries(
  ids: string[],
  signal: AbortSignal,
): Promise<Map<string, ItemSummary>> {
  const uniqueIds = Array.from(new Set(ids)).filter(
    (id) => id.trim().length > 0,
  );
  if (uniqueIds.length === 0) {
    return new Map();
  }

  const query = uniqueIds
    .map((id) => `ids=${encodeURIComponent(id)}`)
    .join("&");
  const response = await fetch(`/api/items?${query}`, { signal });
  if (!response.ok) {
    throw new Error(`Failed to fetch items (${response.status})`);
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(
      `Failed to parse item lookup response: ${
        error instanceof Error ? error.message : "unknown error"
      }`,
    );
  }

  if (!isRecord(payload) || !Array.isArray(payload.items)) {
    return new Map();
  }

  const map = new Map<string, ItemSummary>();
  for (const entry of payload.items) {
    if (!isRecord(entry)) {
      continue;
    }
    const id = toStringOrNull(entry.id);
    if (!id) {
      continue;
    }
    map.set(id, entry as ItemSummary);
  }
  return map;
}

export default function SearchClient({
  initialQuery,
  initialMode,
  initialData,
}: SearchClientProps) {
  const formRef = useRef<HTMLFormElement | null>(null);
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const searchParamsString = useMemo(
    () => searchParams.toString(),
    [searchParams],
  );

  const [query, setQuery] = useState(initialQuery);
  const [searchQuery, setSearchQuery] = useState(initialQuery);
  const [mode, setMode] = useState<SearchMode>(initialMode);
  const [results, setResults] = useState<SearchResult[]>(
    sortResultsByScore(initialData.results),
  );
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [statusReasoning, setStatusReasoning] = useState<string | null>(null);
  const [statusQueue, setStatusQueue] = useState<
    Array<{ message: string | null; reasoning: string | null }>
  >([]);
  const [isStatusBusy, setIsStatusBusy] = useState(false);
  const isStatusBusyRef = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTrigger, setSearchTrigger] = useState(0);
  const [isButtonPressed, setIsButtonPressed] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const lastSyncedSearchRef = useRef(searchParamsString);
  const searchRunIdRef = useRef(0);
  const itemCacheRef = useRef<Map<string, ItemSummary>>(new Map());
  const isInitialMount = useRef(true);
  const lastStatusRef = useRef<{ message: string | null; reasoning: string | null }>({ message: null, reasoning: null });

  // Sync local state with the URL when the user navigates with history controls
  useEffect(() => {
    const params = new URLSearchParams(searchParamsString);
    const nextQuery = params.get("q") ?? "";
    const nextModeParam = params.get("mode");
    const desiredMode: SearchMode =
      nextModeParam === "semantic"
        ? "semantic"
        : nextModeParam === "agentic"
          ? "agentic"
          : "lexical";

    setQuery((current) => (current === nextQuery ? current : nextQuery));
    setSearchQuery((current) => (current === nextQuery ? current : nextQuery));
    setMode((current) => (current === desiredMode ? current : desiredMode));
    lastSyncedSearchRef.current = searchParamsString;
  }, [searchParamsString]);

  useEffect(() => {
    // keep ref in sync to avoid stale closures inside stream handler
    isStatusBusyRef.current = isStatusBusy;
  }, [isStatusBusy]);

  useEffect(() => {
    if (!isStatusBusy) {
      if (statusQueue.length > 0) {
        const [nextStatus, ...rest] = statusQueue;
        setStatusQueue(rest);
        setStatusMessage(nextStatus.message);
        setStatusReasoning(nextStatus.reasoning);
        setIsStatusBusy(true);
      }
    } else { // isStatusBusy is true
      const timer = setTimeout(() => {
        setIsStatusBusy(false);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [isStatusBusy, statusQueue]);

  // Keep the URL in sync with the current search configuration
  useEffect(() => {
    const params = new URLSearchParams();
    const trimmedQuery = searchQuery.trim();
    if (trimmedQuery) {
      params.set("q", trimmedQuery);
    }
    if (mode !== "lexical") {
      params.set("mode", mode);
    }

    const nextSearch = params.toString();
    if (nextSearch === lastSyncedSearchRef.current) {
      return;
    }

    lastSyncedSearchRef.current = nextSearch;
    router.replace(nextSearch ? `${pathname}?${nextSearch}` : pathname, {
      scroll: false,
    });
  }, [searchQuery, mode, router, pathname]);

  const resetSearchState = useCallback(() => {
    setResults([]);
    setError(null);
    setIsLoading(false);
    setIsStreaming(false);
    setStatusMessage(null);
    setStatusReasoning(null);
    setStatusQueue([]);
    setIsStatusBusy(false);
    isStatusBusyRef.current = false;
    itemCacheRef.current = new Map();
    lastStatusRef.current = { message: null, reasoning: null };
  }, []);

  const upsertResults = useCallback(
    (entries: SearchResult[], options: { replace?: boolean } = {}) => {
      const { replace = false } = options;
      if (entries.length === 0 && !replace) {
        return;
      }

      setResults((prev) => {
        if (replace) {
          return sortResultsByScore(entries);
        }
        const existing = new Map(prev.map((entry) => [entry.item.id, entry]));
        let changed = false;
        const next = [...prev];

        entries.forEach((entry) => {
          const current = existing.get(entry.item.id);
          if (current) {
            const index = next.findIndex((r) => r.item.id === entry.item.id);
            if (index !== -1) {
              next[index] = entry;
              changed = true;
            }
          } else {
            existing.set(entry.item.id, entry);
            next.push(entry);
            changed = true;
          }
        });

        return changed ? sortResultsByScore(next) : prev;
      });
    },
    [],
  );

  const processStreamItems = useCallback(
    async (
      items: AgentStreamItem[],
      controller: AbortController,
      { replace = false }: { replace?: boolean } = {},
    ) => {
      if (items.length === 0) {
        // Only clear results on explicit replace when nothing has been streamed at all
        if (replace && itemCacheRef.current.size === 0) {
          setResults([]);
        }
        return;
      }

      const cache = itemCacheRef.current;
      const missingIds = items
        .filter((item) => !cache.has(item.id))
        .map((item) => item.id);

      if (missingIds.length > 0) {
        try {
          const summaries = await fetchItemSummaries(
            missingIds,
            controller.signal,
          );
          summaries.forEach((value, key) => {
            cache.set(key, value);
          });
        } catch (err) {
          if ((err as Error).name === "AbortError") {
            return;
          }
          console.error("Failed to fetch streamed items", err);
          setError("Unable to load some streamed results. Try again?");
        }
      }

      const entries: SearchResult[] = [];
      for (const item of items) {
        const summary = cache.get(item.id);
        if (!summary) {
          continue;
        }
        entries.push({
          item: summary,
          preview: item.preview,
          score: item.score,
          distance: item.distance,
        });
      }

      if (entries.length > 0) {
        upsertResults(entries, { replace });
      }
    },
    [upsertResults],
  );

  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }
    const trimmedQuery = searchQuery.trim();

    // Cancel any in-flight work when dependencies change
    abortControllerRef.current?.abort();

    if (!trimmedQuery) {
      resetSearchState();
      return;
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;
    const searchId = (searchRunIdRef.current += 1);
    itemCacheRef.current = new Map();

    // Timer to clear the final status message after a short delay
    let clearStatusTimeout: ReturnType<typeof setTimeout> | null = null;

    const beginSearch = () => {
      setResults([]);
      setError(null);
      setStatusMessage(null);
      setStatusReasoning(null);
      lastStatusRef.current = { message: null, reasoning: null };
    };

    const completeSearch = (cleanupOnly = false) => {
      if (searchRunIdRef.current !== searchId) {
        return;
      }
      abortControllerRef.current = null;
      setIsLoading(false);
      setIsStreaming(false);
      if (!cleanupOnly) {
        if (clearStatusTimeout) {
          clearTimeout(clearStatusTimeout);
        }
        // Keep the final status visible for 3 seconds, then clear
        clearStatusTimeout = setTimeout(() => {
          if (searchRunIdRef.current === searchId) {
            setStatusMessage(null);
            setStatusReasoning(null);
          }
        }, 3000);
      }
    };

    if (mode !== "agentic") {
      beginSearch();
      setIsLoading(true);
      setIsStreaming(false);

      const params = new URLSearchParams();
      params.set("query", trimmedQuery);
      params.set("mode", mode);
      params.set("limit", String(MAX_SEARCH_LIMIT));

      fetch(`/api/search?${params.toString()}`, {
        signal: controller.signal,
      })
        .then(async (response) => {
          if (!response.ok) {
            throw new Error(`Search failed (${response.status})`);
          }
          const payload = (await response.json()) as SearchApiResponse;
          if (searchRunIdRef.current !== searchId) {
            return;
          }
          const nextResults = Array.isArray(payload.results)
            ? payload.results
            : [];
          setResults(sortResultsByScore(nextResults));
          setError(null);
        })
        .catch((err) => {
          if ((err as Error).name === "AbortError") {
            return;
          }
          if (searchRunIdRef.current !== searchId) {
            return;
          }
          console.error("Search request failed", err);
          setError("Something went wrong while searching. Please try again.");
        })
        .finally(() => {
          completeSearch(true);
        });

      return () => {
        controller.abort();
        if (clearStatusTimeout) {
          clearTimeout(clearStatusTimeout);
        }
      };
    }

    // Agentic streaming search
    beginSearch();
    setIsLoading(false);
    setIsStreaming(true);

    const params = new URLSearchParams();
    params.set("query", trimmedQuery);
    params.set("limit", String(STREAM_RESULT_LIMIT));

    const runStream = async () => {
      try {
        const response = await fetch(
          `/api/search/agentic?${params.toString()}`,
          { signal: controller.signal },
        );
        if (!response.ok || !response.body) {
          const text = await response.text().catch(() => "");
          throw new Error(
            text && text.trim()
              ? text.trim()
              : `Agentic search failed (${response.status})`,
          );
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        const handleEvent = async (event: AgentStreamEvent) => {
          if (searchRunIdRef.current !== searchId) {
            return;
          }

          const kind = event.kind ?? "";
          if (kind === "status") {
            const newStatus = { message: event.message ?? null, reasoning: event.reasoning ?? null };
            // Deduplicate consecutive identical status updates
            if (
              newStatus.message === lastStatusRef.current.message &&
              newStatus.reasoning === lastStatusRef.current.reasoning
            ) {
              return;
            }
            lastStatusRef.current = newStatus;
            if (isStatusBusyRef.current) {
              setStatusQueue((q) => [...q, newStatus]);
            } else {
              setStatusMessage(newStatus.message);
              setStatusReasoning(newStatus.reasoning);
              setIsStatusBusy(true);
            }
            return;
          }

          if (kind === "items") {
            const items = parseItemsPayload(event.data);
            await processStreamItems(items, controller);
            return;
          }

          if (kind === "final") {
            const items = parseItemsPayload(event.data);
            const shouldReplace =
              items.length > 0 || (event.message ?? "") === "no-results";
            await processStreamItems(items, controller, {
              replace: shouldReplace,
            });

            if (event.message === "error") {
              const reason =
                toStringOrNull(event.reason) ?? toStringOrNull(event.reasoning);
              setError(reason ?? "Agent search failed.");
            } else if (event.message === "no-results") {
              setError(null);
            } else {
              setError(null);
            }

            completeSearch();
            return;
          }
        };

        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            break;
          }
          if (!value) {
            continue;
          }

          buffer += decoder.decode(value, { stream: true });

          let boundary = buffer.indexOf("\n\n");
          while (boundary !== -1) {
            const rawEvent = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);

            const dataLines = rawEvent
              .split("\n")
              .filter((line) => line.startsWith("data:"))
              .map((line) => line.slice(5).trimStart());

            if (dataLines.length === 0) {
              boundary = buffer.indexOf("\n\n");
              continue;
            }

            const jsonPayload = dataLines.join("\n");
            try {
              const parsed = JSON.parse(jsonPayload) as AgentStreamEvent;
              await handleEvent(parsed);
            } catch (error) {
              console.error("Failed to parse agent event", error);
            }
            boundary = buffer.indexOf("\n\n");
          }
        }

        completeSearch();
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          return;
        }
        if (searchRunIdRef.current !== searchId) {
          return;
        }
        console.error("Agentic search failed", err);
        setError(
          err instanceof Error
            ? err.message
            : "Agentic search failed unexpectedly.",
        );
        completeSearch();
      }
    };

    runStream();

    return () => {
      controller.abort();
      if (clearStatusTimeout) {
        clearTimeout(clearStatusTimeout);
      }
    };
  }, [processStreamItems, resetSearchState, searchTrigger]);

  const handleSubmit: React.FormEventHandler<HTMLFormElement> = (event) => {
    event.preventDefault();
    // Ensure mode is synced from URL at submit time (covers timing issues)
    const urlMode = searchParams.get("mode");
    const submitMode: SearchMode =
      urlMode === "semantic" ? "semantic" : urlMode === "agentic" ? "agentic" : "lexical";
    setMode((current) => (current === submitMode ? current : submitMode));

    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      setQuery("");
      setSearchQuery("");
      resetSearchState();
      return;
    }
    setSearchQuery(trimmedQuery);
    setSearchTrigger((prev) => prev + 1);

    // Trigger button press effect
    setIsButtonPressed(true);
    setTimeout(() => setIsButtonPressed(false), 150);
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newQuery = event.target.value;
    setQuery(newQuery);

    if (newQuery.trim() === "" && searchQuery.trim() !== "") {
      setSearchQuery("");
      resetSearchState();
    }
  };

  const handleInputFocus = (event: React.FocusEvent<HTMLInputElement>) => {
    event.target.select();
  };

  const handleDelete = (itemId: string) => {
    setResults((prev) => prev.filter((r) => r.item.id !== itemId));
    const cache = itemCacheRef.current;
    cache.delete(itemId);
  };

  const handleCancelAgent = () => {
    searchRunIdRef.current += 1;
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsStreaming(false);
    setIsLoading(false);
    setStatusMessage(null);
    setStatusReasoning(null);
    setStatusQueue([]);
    setIsStatusBusy(false);
  };

  const showLoadingSkeletons =
    searchQuery.trim().length > 0 &&
    ((mode !== "agentic" && isLoading) || (mode === "agentic" && isStreaming));

  const showEmptyState =
    searchQuery.trim().length > 0 &&
    !isLoading &&
    !isStreaming &&
    !error &&
    results.length === 0;

  return (
    <div className="flex flex-col gap-[15px] p-[25px]">
      <form ref={formRef} className="flex flex-col gap-[10px]" onSubmit={handleSubmit}>
        <motion.div
          layout
          initial={false}
          animate={{ scale: isStreaming ? 1.02 : 1, boxShadow: isStreaming ? "0 0 0 0 rgba(0,0,0,0)" : undefined }}
          transition={{ type: "spring", stiffness: 300, damping: 24 }}
          className={`flex w-[750px] flex-col justify-end rounded-[25px] transition-all ${
            isStreaming
              ? "bg-white shadow-inner animate-search-input-pulse cursor-wait"
              : "bg-white hover:drop-shadow-md hover:scale-101"
          }`}
        >
          <div className="flex h-[50px] items-center">
            <input
              type="text"
              value={query}
              onChange={handleInputChange}
              onFocus={handleInputFocus}
              placeholder="Search for items"
              disabled={isStreaming}
              className={`flex-1 h-full bg-transparent text-[15px] font-medium text-slate-600 focus:outline-none ${
                isStreaming ? "cursor-not-allowed opacity-80" : ""
              }`}
              style={{ paddingLeft: "25px", paddingRight: "25px" }}
            />
            <div className="flex items-center justify-end pr-1">
              {statusMessage && (
                <span className="text-sm font-medium text-slate-500">
                  {statusMessage}
                </span>
              )}
              <button
                type="button"
                onClick={() => {
                  if (isStreaming) {
                    handleCancelAgent();
                  } else {
                    // Programmatically submit the form to avoid type toggling issues
                    formRef.current?.requestSubmit?.();
                  }
                }}
                style={{ marginRight: "5px", marginLeft: "8px" }}
                className={`flex h-[40px] min-w-[70px] items-center justify-center rounded-full text-[15px] font-bold text-slate-900 transition ${
                  isStreaming
                    ? "bg-black text-white animate-none"
                    : `hover:scale-105 hover:bg-blue-400 hover:drop-shadow-md ${
                        isButtonPressed
                          ? "scale-105 bg-blue-400 drop-shadow-md"
                          : "bg-blue-400/50"
                      }`
                }`}>
                {isStreaming ? "Cancel" : "⏎"}
              </button>
            </div>
          </div>
          <div
            style={{ paddingLeft: "25px", paddingRight: "25px" }}
            className={`text-sm text-gray-500 transition-all duration-200 ease-out ${
              statusReasoning
                ? "max-h-32 py-3 opacity-100"
                : "max-h-0 py-0 opacity-0"
            } overflow-hidden`}
          >
            {statusReasoning}
          </div>
        </motion.div>
      </form>

      <div className="flex flex-col gap-[15px]">
        {results.map((result, index) => (
          <Link
            key={result.item.id}
            href={result.item.url}
            target="_blank"
            rel="noopener noreferrer"
            className="block rounded-lg outline-none transition-all duration-300 ease-out transform focus-visible:-translate-y-1 focus-visible:drop-shadow-lg focus-visible:outline focus-visible:outline-offset-2 focus-visible:outline-blue-500 animate-search-result-enter"
            style={{
              animationDelay: `${index * 50}ms`,
            }}
          >
            <ItemCard item={result.item} onDelete={handleDelete} />
          </Link>
        ))}

        {showLoadingSkeletons && (
          <>
            {[...Array(3)].map((_, index) => (
              <div
                key={`loading-${index}`}
                className="flex rounded-2xl overflow-hidden border border-slate-200 bg-white animate-loading-pulse"
                style={{
                  animationDelay: `${index * 200}ms`,
                  animationDuration: "1000ms",
                  animationIterationCount: "infinite",
                  animationFillMode: "both",
                }}
              >
                <div className="flex justify-center bg-white p-[15px]">
                  <div className="flex justify-center items-center w-[45px] rounded-xl h-[45px] bg-[#D8D8D8]">
                    <div className="bg-gray-200 h-[35px] w-[35px] rounded-[8px] flex items-center justify-center border border-gray-200" />
                  </div>
                </div>

                <div className="flex flex-col px-[15px] py-[12px] flex-1 gap-[6px]">
                  <div className="h-[20px] bg-slate-200 rounded w-3/4" />
                  <div className="flex flex-col gap-[4px] min-h-[48px]">
                    <div className="h-[12px] bg-slate-200 rounded w-full max-w-[220px]" />
                    <div className="h-[12px] bg-slate-200 rounded w-3/4 max-w-[180px]" />
                    <div className="h-[12px] bg-slate-200 rounded w-4/5 max-w-[200px]" />
                  </div>
                </div>

                <div className="flex flex-col items-end justify-start gap-[6px] p-[15px] ml-auto">
                  <div className="h-[20px] bg-slate-200 rounded w-[40px]" />
                  <div className="flex items-center gap-[5px] px-[10px] rounded-xl bg-transparent h-[20px]">
                    <div className="h-[12px] bg-slate-200 rounded w-[35px]" />
                    <div className="h-[15px] bg-slate-200 rounded w-[15px]" />
                  </div>
                  <div className="flex items-center gap-[5px] px-[10px] rounded-xl bg-transparent h-[20px]">
                    <div className="h-[12px] bg-slate-200 rounded w-[30px]" />
                    <div className="h-[15px] bg-slate-200 rounded w-[15px]" />
                  </div>
                </div>
              </div>
            ))}
          </>
        )}
      </div>
      {showEmptyState && (
        <div className="text-center text-slate-500 text-sm py-6">No results</div>
      )}
    </div>
  );
}
