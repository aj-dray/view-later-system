import type { ItemSummary } from "@/app/_lib/items";

const STATUS_PRIORITY: Record<string, number> = {
  error: 0,
  bookmark: 1,
  queued: 2,
  adding: 2,
  paused: 3,
  completed: 4,
};

const QUEUE_LIKE_STATUSES = new Set(["queued", "adding"]);

export type QueueOrder = "date" | "random" | "priority";
export type QueueFilter = "queued" | "all";
export type QueueTypeFilter = "all" | "article" | "video" | "paper" | "podcast" | "post" | "newsletter" | "other";

function parseTimestamp(value?: string | null): number {
  if (!value) {
    return 0;
  }

  const numeric = Number(value);
  if (!Number.isNaN(numeric)) {
    const millisecondsThreshold = 10 ** 12;
    return numeric < millisecondsThreshold ? numeric * 1000 : numeric;
  }

  const parsedDate = Date.parse(value);
  return Number.isNaN(parsedDate) ? 0 : parsedDate;
}

function shuffleOrder<T>(input: readonly T[]): T[] {
  const arr = [...input];
  for (let index = arr.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [arr[index], arr[swapIndex]] = [arr[swapIndex], arr[index]];
  }
  return arr;
}

function sortByCreatedAt(items: readonly ItemSummary[]): ItemSummary[] {
  return [...items].sort(
    (a, b) => parseTimestamp(b.created_at) - parseTimestamp(a.created_at),
  );
}

function daysSince(dateValue: string | null | undefined): number {
  const ts = parseTimestamp(dateValue);
  if (!ts) return 0;
  const now = Date.now();
  const diffMs = Math.max(0, now - ts);
  return diffMs / (1000 * 60 * 60 * 24);
}

function calculatePriority(
  days_since_added: number,
  expiry_score: number,
  base_period = 3,
  k = 10,
): number {
  if (!Number.isFinite(days_since_added) || days_since_added < 0) {
    days_since_added = 0;
  }
  const safeExpiry = Math.min(1, Math.max(0.1, Number(expiry_score) || 0.1));
  const time_to_peak = base_period * (1 / safeExpiry);
  const progress = time_to_peak > 0 ? days_since_added / time_to_peak : 0;
  const priority = 1 / (1 + Math.exp(-k * (progress - 0.5)));
  return Math.min(1, Math.max(0, priority));
}

function sortByPriority(items: readonly ItemSummary[]): ItemSummary[] {
  // Compute priority for each item and sort descending by score
  return [...items]
    .map((item) => {
      const days = daysSince(item.created_at);
      const score = calculatePriority(days, item.expiry_score ?? 0.1);
      return { item, score } as const;
    })
    .sort((a, b) => b.score - a.score)
    .map((entry) => entry.item);
}

export function queueOrder(items: readonly ItemSummary[]): ItemSummary[] {
  return [...items].sort((a, b) => {
    const priorityDiff =
      (STATUS_PRIORITY[a.client_status] ?? Number.POSITIVE_INFINITY) -
      (STATUS_PRIORITY[b.client_status] ?? Number.POSITIVE_INFINITY);
    if (priorityDiff !== 0) {
      return priorityDiff;
    }

    return parseTimestamp(b.created_at) - parseTimestamp(a.created_at);
  });
}

export function applyQueueFilter(
  items: readonly ItemSummary[],
  filter: QueueFilter,
): ItemSummary[] {
  if (filter === "queued") {
    return items.filter(
      (item) =>
        QUEUE_LIKE_STATUSES.has(item.client_status) ||
        item.client_status === "error" ||
        item.client_status === "bookmark",
    );
  }
  return [...items];
}

export function applyTypeFilter(
  items: readonly ItemSummary[],
  typeFilter: QueueTypeFilter,
): ItemSummary[] {
  if (typeFilter === "all") {
    return [...items];
  }
  return items.filter((item) => item.type === typeFilter);
}

export function sortQueueItems(
  items: readonly ItemSummary[],
  order: QueueOrder,
): ItemSummary[] {
  if (order === "date") {
    return queueOrder(items);
  }

  if (order === "random") {
    const queueItems = items.filter((item) =>
      QUEUE_LIKE_STATUSES.has(item.client_status),
    );
    const nonQueueItems = items.filter(
      (item) => !QUEUE_LIKE_STATUSES.has(item.client_status),
    );

    const sortedNonQueue = queueOrder(nonQueueItems);
    const errorItems = sortedNonQueue.filter(
      (item) => item.client_status === "error",
    );
    const bookmarkedItems = sortedNonQueue.filter(
      (item) => item.client_status === "bookmark",
    );
    const pausedItems = sortedNonQueue.filter(
      (item) => item.client_status === "paused",
    );
    const completedItems = sortedNonQueue.filter(
      (item) => item.client_status === "completed",
    );
    const otherItems = sortedNonQueue.filter(
      (item) =>
        !["error", "bookmark", "paused", "completed"].includes(
          item.client_status,
        ),
    );

    return [
      ...errorItems,
      ...bookmarkedItems,
      ...shuffleOrder(queueItems),
      ...pausedItems,
      ...completedItems,
      ...sortByCreatedAt(otherItems),
    ];
  }

  if (order === "priority") {
    const queueItems = items.filter((item) =>
      QUEUE_LIKE_STATUSES.has(item.client_status),
    );
    const nonQueueItems = items.filter(
      (item) => !QUEUE_LIKE_STATUSES.has(item.client_status),
    );

    const sortedNonQueue = queueOrder(nonQueueItems);
    const errorItems = sortedNonQueue.filter(
      (item) => item.client_status === "error",
    );
    const bookmarkedItems = sortedNonQueue.filter(
      (item) => item.client_status === "bookmark",
    );
    const pausedItems = sortedNonQueue.filter(
      (item) => item.client_status === "paused",
    );
    const completedItems = sortedNonQueue.filter(
      (item) => item.client_status === "completed",
    );
    const otherItems = sortedNonQueue.filter(
      (item) =>
        !["error", "bookmark", "paused", "completed"].includes(
          item.client_status,
        ),
    );

    return [
      ...errorItems,
      ...bookmarkedItems,
      ...sortByPriority(queueItems),
      ...pausedItems,
      ...completedItems,
      ...sortByCreatedAt(otherItems),
    ];
  }

  console.warn(`Unknown order: ${order}`);
  return [...items];
}

export type ItemLoadingStage =
  | "creating"
  | "metadata"
  | "embedding"
  | "complete";

export function deriveLoadingStage(
  item: ItemSummary | null | undefined,
): ItemLoadingStage {
  if (!item) {
    return "creating";
  }

  // Handle error status - treat as complete since we're not processing
  if (item.client_status === "error") {
    return "complete";
  }

  // If client_status is still "adding", don't show as complete even if server processing is done
  if (item.client_status === "adding") {
    switch (item.server_status) {
      case "embedded":
      case "classified":
        return "embedding";
      case "summarised":
        return item.summary ? "embedding" : "metadata";
      case "extracted":
        return item.title ? "metadata" : "creating";
      default:
        return "creating";
    }
  }

  // Normal flow when client_status is not "adding"
  switch (item.server_status) {
    case "embedded":
    case "classified":
      return "complete";
    case "summarised":
      return item.summary ? "embedding" : "metadata";
    case "extracted":
      return item.title ? "metadata" : "creating";
    default:
      return "creating";
  }
}

export function stageLabel(stage: ItemLoadingStage): string {
  switch (stage) {
    case "creating":
      return "processing";
    case "metadata":
      return "parsing";
    case "embedding":
      return "embedding";
    case "complete":
    default:
      return "ready";
  }
}
