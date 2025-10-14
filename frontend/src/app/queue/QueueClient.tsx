"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import Link from "next/link";

import ItemCard from "@/app/_components/ItemCard";
import type { AddedItemDetail } from "@/app/_components/AddPanel";
import type { ItemSummary } from "@/app/_lib/items";
import {
  applyQueueFilter,
  applyTypeFilter,
  deriveLoadingStage,
  sortQueueItems,
  type ItemLoadingStage,
  type QueueFilter,
  type QueueOrder,
  type QueueTypeFilter,
} from "@/app/_lib/queue";
import { useSettings } from "@/app/_contexts/SettingsContext";

const POLL_INTERVAL = 1500; // Single interval for all adding items

const QUEUE_ITEM_SELECTOR = "[data-queue-item-id]";

function useReorderAnimation(itemIds: readonly string[]) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const previousPositions = useRef(new Map<string, DOMRect>());
  const signature = useMemo(() => itemIds.join("|"), [itemIds]);

  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) {
      previousPositions.current.clear();
      return;
    }

    const elements = Array.from(
      container.querySelectorAll<HTMLElement>(QUEUE_ITEM_SELECTOR),
    );

    const nextPositions = new Map<string, DOMRect>();
    const cleanups: Array<() => void> = [];

    elements.forEach((element) => {
      const key = element.dataset.queueItemId;
      if (!key) {
        return;
      }

      const rect = element.getBoundingClientRect();
      nextPositions.set(key, rect);

      const previousRect = previousPositions.current.get(key);
      if (previousRect) {
        const deltaX = previousRect.left - rect.left;
        const deltaY = previousRect.top - rect.top;

        if (deltaX !== 0 || deltaY !== 0) {
          element.style.willChange = "transform";
          element.style.transition = "transform 0s";
          element.style.transform = `translate(${deltaX}px, ${deltaY}px)`;

          const finish = () => {
            element.style.transition = "";
            element.style.transform = "";
            element.style.willChange = "";
            element.removeEventListener("transitionend", finish);
          };

          requestAnimationFrame(() => {
            element.addEventListener("transitionend", finish, { once: true });
            element.style.transition =
              "transform 320ms cubic-bezier(0.22, 1, 0.36, 1)";
            element.style.transform = "";
          });

          cleanups.push(() => finish());
        }
      }
    });

    previousPositions.current = nextPositions;

    return () => {
      cleanups.forEach((cleanup) => cleanup());
    };
  }, [signature]);

  useEffect(() => {
    return () => {
      previousPositions.current.clear();
    };
  }, []);

  return containerRef;
}

type DisplayItem = {
  item: ItemSummary;
  stage: ItemLoadingStage;
  isOptimistic: boolean;
  addingError: string | null;
};

type QueueClientProps = {
  initialItems: ItemSummary[];
};

function createPlaceholderItem(detail: AddedItemDetail): ItemSummary {
  const now = new Date().toISOString();
  return {
    id: detail.itemId,
    user_id: null,
    url: detail.url,
    canonical_url: null,
    title: null,
    source_site: null,
    format: null,
    author: null,
    type: null,
    publication_date: null,
    favicon_url: null,
    content_markdown: null,
    content_text: null,
    content_token_count: null,
    client_status: "adding",
    server_status: "saved",
    summary: null,
    expiry_score: null,
    ts_embedding: null,
    mistral_embedding: null,
    client_status_at: now,
    server_status_at: now,
    created_at: now,
  } satisfies ItemSummary;
}

export default function QueueClient({ initialItems }: QueueClientProps) {
  const { getSetting } = useSettings();

  // Get current settings from global context
  const order = (getSetting("order") as QueueOrder) || "date";
  const filter = (getSetting("filter") as QueueFilter) || "queued";
  const typeFilter = (getSetting("typeFilter") as QueueTypeFilter) || "all";
  const [items, setItems] = useState<DisplayItem[]>(() =>
    initialItems.map((item) => ({
      item,
      stage: deriveLoadingStage(item),
      isOptimistic: false,
      addingError:
        item.client_status === "error" ? "Failed to process URL" : null,
    })),
  );

  const pollTimers = useRef(new Map<string, number>());
  const animatedOrder = useMemo(
    () => items.map((entry) => entry.item.id),
    [items],
  );
  const listRef = useReorderAnimation(animatedOrder);

  const sortDisplayItems = useCallback(
    (entries: DisplayItem[]): DisplayItem[] => {
      const processedEntries = entries.map((entry) => {
        const stage = deriveLoadingStage(entry.item);
        return {
          ...entry,
          stage,
          isOptimistic: entry.isOptimistic && stage !== "complete",
          addingError: entry.addingError,
        };
      });

      // Separate error/adding items first - they always appear regardless of filter
      const errorItems = processedEntries.filter((entry) => entry.addingError);
      const addingItems = processedEntries.filter(
        (entry) => !entry.addingError && entry.item.client_status === "adding",
      );
      const otherEntries = processedEntries.filter(
        (entry) => !entry.addingError && entry.item.client_status !== "adding",
      );

      // Apply filter only to non-error, non-adding items
      const otherItems = otherEntries.map((entry) => entry.item);
      const filteredByStatus = applyQueueFilter(otherItems, filter);
      const filteredByType = applyTypeFilter(filteredByStatus, typeFilter);
      const filteredOtherEntries = otherEntries.filter((entry) =>
        filteredByType.some((item) => item.id === entry.item.id),
      );

      const sorted = sortQueueItems(
        filteredOtherEntries.map((entry) => entry.item),
        order,
      );
      const entryById = new Map(
        filteredOtherEntries.map((entry) => [entry.item.id, entry]),
      );
      const sortedOtherEntries = sorted.map((item) => {
        const existing = entryById.get(item.id);
        const stage = deriveLoadingStage(item);
        const optimistic = existing?.isOptimistic
          ? stage !== "complete"
          : false;
        const addingError = existing?.addingError ?? null;
        return {
          item,
          stage,
          isOptimistic: optimistic,
          addingError,
        } satisfies DisplayItem;
      });

      return [...errorItems, ...addingItems, ...sortedOtherEntries];
    },
    [filter, typeFilter, order],
  );

  useEffect(() => {
    setItems((prev) => sortDisplayItems(prev));
  }, [sortDisplayItems]);

  const upsertItem = useCallback(
    (
      item: ItemSummary,
      options: { optimistic?: boolean; addingError?: string | null } = {},
    ) => {
      setItems((prev) => {
        const existingIndex = prev.findIndex(
          (entry) => entry.item.id === item.id,
        );
        const stage = deriveLoadingStage(item);
        const shouldMarkOptimistic =
          options.optimistic ??
          (existingIndex >= 0 ? prev[existingIndex].isOptimistic : false);
        const resolvedError =
          options.addingError !== undefined
            ? options.addingError
            : existingIndex >= 0
              ? prev[existingIndex].addingError
              : null;
        const nextEntry: DisplayItem = {
          item,
          stage,
          isOptimistic: shouldMarkOptimistic && stage !== "complete",
          addingError: resolvedError,
        };

        const nextEntries =
          existingIndex >= 0
            ? prev.map((entry, index) =>
                index === existingIndex ? nextEntry : entry,
              )
            : [nextEntry, ...prev];

        return sortDisplayItems(nextEntries);
      });
    },
    [sortDisplayItems],
  );

  const markItemError = useCallback((itemId: string, message: string) => {
    // Stop polling immediately when an error occurs
    const timerId = pollTimers.current.get(itemId);
    if (timerId) {
      window.clearTimeout(timerId);
      pollTimers.current.delete(itemId);
    }

    setItems((prev) =>
      prev.map((entry) => {
        if (entry.item.id !== itemId) {
          return entry;
        }
        if (entry.item.client_status !== "adding") {
          return entry;
        }
        return { ...entry, addingError: message };
      }),
    );
  }, []);

  const removeItem = useCallback(async (
    itemId: string,
    options?: { alreadyDeleted?: boolean },
  ) => {
    // Check if this is a fake error item (starts with "error-")
    const isFakeItem = itemId.startsWith("error-");

    if (!isFakeItem && !options?.alreadyDeleted) {
      // This is a real database item, try to delete it from the backend
      try {
        const response = await fetch(`/api/items/${itemId}`, {
          method: "DELETE",
        });
        if (!response.ok) {
          console.warn("Failed to delete item from database", { itemId });
        }
      } catch (error) {
        console.error("Error deleting item from database", { itemId, error });
      }
    }

    // Remove from UI regardless of database deletion success
    setItems((prev) => prev.filter((entry) => entry.item.id !== itemId));
    const timerId = pollTimers.current.get(itemId);
    if (timerId) {
      window.clearTimeout(timerId);
      pollTimers.current.delete(itemId);
    }
  }, []);

  useEffect(() => {
    setItems((prev) => {
      if (initialItems.length === 0) {
        return sortDisplayItems(prev.filter((entry) => entry.isOptimistic));
      }

      const seen = new Set<string>();
      const merged: DisplayItem[] = initialItems.map((item) => {
        seen.add(item.id);
        const existing = prev.find((entry) => entry.item.id === item.id);
        const stage = deriveLoadingStage(item);
        const optimistic = existing?.isOptimistic
          ? stage !== "complete"
          : false;
        // Check if item has error status and set addingError accordingly
        const addingError =
          item.client_status === "error"
            ? "Failed to process URL"
            : (existing?.addingError ?? null);
        return {
          item,
          stage,
          isOptimistic: optimistic,
          addingError,
        } satisfies DisplayItem;
      });

      prev.forEach((entry) => {
        if (!seen.has(entry.item.id)) {
          merged.push(entry);
        }
      });

      return sortDisplayItems(merged);
    });
  }, [initialItems, sortDisplayItems]);

  const fetchLatest = useCallback(
    async (itemId: string): Promise<ItemLoadingStage | null> => {
      try {
        const response = await fetch(
          `/api/items?ids=${encodeURIComponent(itemId)}`,
          { cache: "no-store" },
        );
        if (!response.ok) {
          markItemError(itemId, "Error: Failed to fetch item update");
          return null;
        }
        const data = (await response.json()) as { items?: ItemSummary[] };
        const next = data.items?.find((entry) => entry.id === itemId);
        if (!next) {
          return null;
        }
        const stage = deriveLoadingStage(next);
        // Set addingError when item has error status
        const addingError =
          next.client_status === "error" ? "Failed to process URL" : null;
        upsertItem(next, { addingError });
        return stage;
      } catch (error) {
        console.error("Failed to fetch item update", { itemId, error });
        markItemError(itemId, `Error: ${error}`);
        return null;
      }
    },
    [markItemError, upsertItem],
  );

  const currentItemsRef = useRef<DisplayItem[]>([]);

  // Keep ref updated with current items
  useEffect(() => {
    currentItemsRef.current = items;
  }, [items]);

  const schedulePoll = useCallback(
    (itemId: string) => {
      const existingTimer = pollTimers.current.get(itemId);
      if (existingTimer) {
        console.log(`[Poll] Clearing existing timer for ${itemId}`);
        window.clearInterval(existingTimer);
        pollTimers.current.delete(itemId);
      }

      console.log(`[Poll] Starting poll for ${itemId}`);
      const interval = window.setInterval(async () => {
        // Get current item state from ref to avoid stale closure
        const currentItem = currentItemsRef.current.find(
          (entry) => entry.item.id === itemId,
        );

        console.log(`[Poll] Checking item ${itemId}:`, {
          exists: !!currentItem,
          hasError: !!currentItem?.addingError,
          status: currentItem?.item.client_status,
        });

        // Stop if item has error or is no longer in adding status
        if (
          !currentItem ||
          currentItem.addingError ||
          (currentItem.item.client_status !== "adding" &&
            currentItem.item.client_status !== "error")
        ) {
          console.log(
            `[Poll] Stopping poll for ${itemId} - not adding or has error`,
          );
          window.clearInterval(interval);
          pollTimers.current.delete(itemId);
          return;
        }

        // If item status is "error", convert to addingError and stop polling
        if (currentItem.item.client_status === "error") {
          console.log(
            `[Poll] Item ${itemId} has error status, marking as error`,
          );
          setItems((prev) =>
            prev.map((entry) => {
              if (entry.item.id !== itemId) {
                return entry;
              }
              return {
                ...entry,
                addingError: "Failed to process URL",
              };
            }),
          );
          window.clearInterval(interval);
          pollTimers.current.delete(itemId);
          return;
        }

        console.log(`[Poll] Fetching latest for ${itemId}`);
        await fetchLatest(itemId);

        // Check again after fetch to see if we should continue polling
        const updatedItem = currentItemsRef.current.find(
          (entry) => entry.item.id === itemId,
        );

        console.log(`[Poll] After fetch, item ${itemId}:`, {
          exists: !!updatedItem,
          hasError: !!updatedItem?.addingError,
          status: updatedItem?.item.client_status,
        });

        if (
          !updatedItem ||
          updatedItem.addingError ||
          updatedItem.item.client_status !== "adding"
        ) {
          console.log(`[Poll] Stopping poll for ${itemId} - no longer adding`);
          window.clearInterval(interval);
          pollTimers.current.delete(itemId);
        }
      }, POLL_INTERVAL);

      pollTimers.current.set(itemId, interval);
    },
    [fetchLatest],
  );

  useEffect(() => {
    console.log("[Poll Init] useEffect triggered", {
      initialItemsLength: initialItems.length,
      currentTimerCount: pollTimers.current.size,
    });

    if (initialItems.length === 0) {
      console.log("[Poll Init] No initial items, not starting any polls");
      return;
    }

    console.log(
      "[Poll Init] Initial items:",
      initialItems.map((item) => ({ id: item.id, status: item.client_status })),
    );

    const addingItems = initialItems.filter(
      (item) => item.client_status === "adding",
    );
    console.log("[Poll Init] Items with 'adding' status:", addingItems.length);

    initialItems.forEach((item) => {
      if (item.client_status === "adding") {
        console.log(`[Poll Init] Starting poll for adding item: ${item.id}`);
        schedulePoll(item.id);
      } else {
        console.log(
          `[Poll Init] Skipping item ${item.id} with status: ${item.client_status}`,
        );
      }
    });

    console.log(
      "[Poll Init] Active timers after init:",
      pollTimers.current.size,
    );
  }, [initialItems, schedulePoll]);

  useEffect(() => {
    const handleItemAdded = (event: Event) => {
      const custom = event as CustomEvent<AddedItemDetail>;
      const detail = custom.detail;
      if (!detail?.itemId) {
        return;
      }
      const placeholder = createPlaceholderItem(detail);

      if (detail.error) {
        // For error items, don't start polling
        console.log(
          `[Poll New] Error item ${detail.itemId}, not starting poll`,
        );
        upsertItem(placeholder, {
          optimistic: true,
          addingError: detail.error,
        });
      } else {
        // For successful items, start polling
        console.log(`[Poll New] Starting poll for new item: ${detail.itemId}`);
        upsertItem(placeholder, { optimistic: true, addingError: null });
        schedulePoll(placeholder.id);
      }
    };

    window.addEventListener(
      "queue:item-added",
      handleItemAdded as EventListener,
    );
    return () => {
      window.removeEventListener(
        "queue:item-added",
        handleItemAdded as EventListener,
      );
    };
  }, [schedulePoll, upsertItem]);

  useEffect(() => {
    const timers = pollTimers.current;

    // Log timer status every 5 seconds
    const debugInterval = setInterval(() => {
      if (timers.size > 0) {
        console.log(
          `[Poll Debug] Active timers: ${timers.size}`,
          Array.from(timers.keys()),
        );
        console.log(
          "[Poll Debug] Current items with adding status:",
          currentItemsRef.current
            .filter((item) => item.item.client_status === "adding")
            .map((item) => item.item.id),
        );
      }
    }, 5000);

    return () => {
      clearInterval(debugInterval);
    };
  }, []);

  // Cleanup polling timers on component unmount
  useEffect(() => {
    const timers = pollTimers.current;
    return () => {
      console.log("[Poll Cleanup] Clearing all timers:", timers.size);
      timers.forEach((timerId) => {
        window.clearInterval(timerId);
      });
      timers.clear();
    };
  }, []);

  return (
    <div
      ref={listRef}
      className="flex flex-col gap-[15px] p-[25px] pb-12 overflow-visible"
    >
      {items.map(({ item, stage, isOptimistic, addingError }) => (
        <Link
          key={item.id}
          href={item.url}
          target="_blank"
          rel="noopener noreferrer"
          data-queue-item-id={item.id}
          className="block rounded-lg outline-none transition focus-visible:-translate-y-1 focus-visible:drop-shadow-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-500 overflow-visible"
        >
          <ItemCard
            item={item}
            stage={stage}
            isOptimistic={isOptimistic}
            addingError={addingError}
            onDelete={removeItem}
          />
        </Link>
      ))}
    </div>
  );
}
