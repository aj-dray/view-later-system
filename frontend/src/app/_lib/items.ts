import "server-only";

import { authedFetch as authenticatedFetch } from "@/app/_lib/auth";

export { authedFetch } from "@/app/_lib/auth";

/* === RE-EXPORTS === */
export {
  CLIENT_UPDATABLE_STATUSES,
  CLIENT_STATUSES,
  SERVER_STATUSES,
  ITEM_COLUMNS,
  type ClientStatus,
  type ServerStatus,
  type ItemColumn,
  type MutableItemColumn,
  type FilterOperator,
  type FilterValue,
  type ItemFilter,
  type ItemSummary,
  type ItemDetail,
  type FetchItemsOptions,
  type UpdateItemPayload,
} from "@/app/_lib/items-types";

import {
  CLIENT_STATUSES,
  SERVER_STATUSES,
  ITEM_COLUMNS,
  type ClientStatus,
  type ServerStatus,
  type ItemColumn,
  type MutableItemColumn,
  type ItemFilter,
  type FilterOperator,
  type FilterValue,
  type ItemSummary,
  type UpdateItemPayload,
  type FetchItemsOptions,
} from "@/app/_lib/items-types";

const CLIENT_STATUS_SET = new Set<string>(CLIENT_STATUSES);
const SERVER_STATUS_SET = new Set<string>(SERVER_STATUSES);
const ITEM_COLUMN_SET = new Set<string>(ITEM_COLUMNS);

const IMMUTABLE_ITEM_COLUMNS = ["id", "user_id", "created_at"] as const;
const IMMUTABLE_COLUMN_SET = new Set<string>(IMMUTABLE_ITEM_COLUMNS);

const FILTER_OPERATORS = [
  "=",
  "!=",
  "<",
  "<=",
  ">",
  ">=",
  "LIKE",
  "ILIKE",
  "IN",
] as const;

const FILTER_OPERATOR_SET = new Set<string>(FILTER_OPERATORS);

/* === UTILITIES === */

function toStringOrNull(value: unknown): string | null {
  if (typeof value === "string") {
    return value;
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  return null;
}

function toRequiredString(value: unknown, field: string): string {
  const result = toStringOrNull(value);
  if (result) {
    return result;
  }
  throw new Error(`Missing required field: ${field}`);
}

function toNullableInteger(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.trunc(value);
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return Math.trunc(parsed);
    }
  }
  return null;
}

function toNullableNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

function toClientStatus(value: unknown): ClientStatus {
  if (typeof value === "string" && CLIENT_STATUS_SET.has(value)) {
    return value as ClientStatus;
  }
  return "queued";
}

function toServerStatus(value: unknown): ServerStatus {
  if (typeof value === "string" && SERVER_STATUS_SET.has(value)) {
    return value as ServerStatus;
  }
  return "saved";
}

function toNullableEmbedding(value: unknown): number[] | null {
  if (value == null) {
    return null;
  }
  if (Array.isArray(value)) {
    const numbers = value
      .map((entry) => {
        const num = Number(entry);
        return Number.isFinite(num) ? num : null;
      })
      .filter((entry): entry is number => entry !== null);
    return numbers.length ? numbers : null;
  }
  if (typeof value === "string" && value.trim()) {
    try {
      const parsed = JSON.parse(value);
      if (Array.isArray(parsed)) {
        return toNullableEmbedding(parsed);
      }
    } catch (error) {
      console.warn("Failed to parse embedding value", { value, error });
    }
  }
  return null;
}

function toItemSummary(raw: Record<string, unknown>): ItemSummary {
  const id = toRequiredString(raw.id, "id");
  const url = toRequiredString(raw.url, "url");
  const createdAt = toStringOrNull(raw.created_at);
  if (!createdAt) {
    throw new Error("Item payload missing created_at");
  }

  return {
    id,
    user_id: toStringOrNull(raw.user_id),
    url,
    canonical_url: toStringOrNull(raw.canonical_url),
    title: toStringOrNull(raw.title),
    source_site: toStringOrNull(raw.source_site),
    format: toStringOrNull(raw.format),
    author: toStringOrNull(raw.author),
    type: toStringOrNull(raw.type),
    publication_date: toStringOrNull(raw.publication_date),
    favicon_url: toStringOrNull(raw.favicon_url),
    content_markdown: toStringOrNull(raw.content_markdown),
    content_text: toStringOrNull(raw.content_text),
    content_token_count: toNullableInteger(raw.content_token_count),
    client_status: toClientStatus(raw.client_status),
    server_status: toServerStatus(raw.server_status),
    summary: toStringOrNull(raw.summary),
    expiry_score: toNullableNumber(raw.expiry_score),
    ts_embedding: toStringOrNull(raw.ts_embedding),
    mistral_embedding: toNullableEmbedding(raw.mistral_embedding),
    client_status_at: toStringOrNull(raw.client_status_at),
    server_status_at: toStringOrNull(raw.server_status_at),
    created_at: createdAt,
  };
}

function isItemColumn(column: unknown): column is ItemColumn {
  return typeof column === "string" && ITEM_COLUMN_SET.has(column);
}

function isMutableColumn(column: string): column is MutableItemColumn {
  return ITEM_COLUMN_SET.has(column) && !IMMUTABLE_COLUMN_SET.has(column);
}

function normaliseColumns(columns?: ItemColumn[]): ItemColumn[] | undefined {
  if (!columns || columns.length === 0) {
    return undefined;
  }
  const unique: ItemColumn[] = [];
  columns.forEach((column) => {
    if (isItemColumn(column) && !unique.includes(column)) {
      unique.push(column);
    }
  });
  return unique.length ? unique : undefined;
}

function normaliseStatuses<T extends string>(
  statuses: readonly T[] | undefined,
  validSet: Set<string>,
): T[] {
  if (!statuses || statuses.length === 0) {
    return [];
  }
  return statuses.filter((status) => validSet.has(status));
}

function normaliseFilters(filters?: Partial<ItemFilter>[]): ItemFilter[] {
  if (!filters || filters.length === 0) {
    return [];
  }

  return filters
    .map((filter) => {
      if (
        !filter ||
        !isItemColumn(filter.column) ||
        filter.value === undefined
      ) {
        return null;
      }
      const operator = (filter.operator ?? "=").toUpperCase();
      const normalisedOperator = FILTER_OPERATOR_SET.has(operator)
        ? (operator as FilterOperator)
        : "=";
      return {
        column: filter.column,
        operator: normalisedOperator,
        value: filter.value,
      } satisfies ItemFilter;
    })
    .filter((filter): filter is ItemFilter => filter !== null);
}

function formatFilterValue(
  value: FilterValue,
  operator: FilterOperator,
): string | null {
  if (Array.isArray(value)) {
    const values = value
      .map((entry) => {
        if (typeof entry === "boolean") {
          return entry ? "true" : "false";
        }
        return String(entry);
      })
      .filter((entry) => entry.trim().length > 0);

    if (values.length === 0) {
      return null;
    }
    if (operator === "IN") {
      return values.join(",");
    }
    return values[0] ?? null;
  }

  if (value === null || value === undefined) {
    return null;
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  const stringValue = String(value);
  return stringValue.trim() ? stringValue : null;
}

function buildFilterParam(filter: ItemFilter): string | null {
  const operator = filter.operator ?? "=";
  const formattedValue = formatFilterValue(filter.value, operator);
  if (!formattedValue) {
    return null;
  }
  return `${filter.column}:${operator}:${formattedValue}`;
}

function sanitiseUpdatePayload(
  payload: UpdateItemPayload,
): Record<string, unknown> {
  const result: Record<string, unknown> = {};

  Object.entries(payload ?? {}).forEach(([key, value]) => {
    if (value === undefined) {
      return;
    }
    if (!isMutableColumn(key)) {
      return;
    }

    if (key === "client_status" && !CLIENT_STATUS_SET.has(value as string)) {
      return;
    }

    if (key === "server_status" && !SERVER_STATUS_SET.has(value as string)) {
      return;
    }

    result[key] = value;
  });

  return result;
}

function assertRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

/* === FETCH === */

export async function fetchItems(
  options: FetchItemsOptions = {},
): Promise<ItemSummary[]> {
  const {
    clientStatuses,
    serverStatuses,
    filters,
    columns,
    limit = 50,
    offset = 0,
    orderBy = "created_at",
    order = "desc",
  } = options;

  const params = new URLSearchParams();

  params.set("limit", String(limit));
  params.set("offset", String(offset));

  if (isItemColumn(orderBy)) {
    params.set("order_by", orderBy);
  }
  params.set("order", order === "asc" ? "asc" : "desc");

  const selectedColumns = normaliseColumns(columns);
  selectedColumns?.forEach((column) => {
    params.append("columns", column);
  });

  const filterList: ItemFilter[] = [];
  filterList.push(...normaliseFilters(filters));

  const validClientStatuses = normaliseStatuses(
    clientStatuses,
    CLIENT_STATUS_SET,
  );
  if (validClientStatuses.length > 0) {
    filterList.push({
      column: "client_status",
      operator: "IN",
      value: validClientStatuses,
    });
  }

  const validServerStatuses = normaliseStatuses(
    serverStatuses,
    SERVER_STATUS_SET,
  );
  if (validServerStatuses.length > 0) {
    filterList.push({
      column: "server_status",
      operator: "IN",
      value: validServerStatuses,
    });
  }

  filterList.forEach((filter) => {
    const encoded = buildFilterParam(filter);
    if (encoded) {
      params.append("filter", encoded);
    }
  });

  const query = params.toString();
  const path = query ? `/items/select?${query}` : "/items/select";
  const response = await authenticatedFetch(path, {
    cache: "no-store",
    next: { revalidate: 0 },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch items: ${response.status}`);
  }

  const payload = (await response.json()) as unknown;
  if (!Array.isArray(payload)) {
    throw new Error("Unexpected response when fetching items");
  }

  return payload.map((entry) => {
    if (!assertRecord(entry)) {
      throw new Error("Invalid item payload received from service");
    }
    return toItemSummary(entry);
  });
}

/* === UPDATE === */

export async function updateItem(
  itemId: string,
  payload: UpdateItemPayload,
): Promise<void> {
  const safeItemId = itemId?.trim();
  if (!safeItemId) {
    throw new Error("Item id is required for update");
  }

  const bodyPayload = sanitiseUpdatePayload(payload);
  if (Object.keys(bodyPayload).length === 0) {
    throw new Error("No valid item fields supplied for update");
  }

  const response = await authenticatedFetch("/items/update", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      item_ids: [safeItemId],
      updates: bodyPayload,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to update item ${safeItemId}: ${response.status}`);
  }

  const json = (await response.json()) as unknown;
  if (!assertRecord(json)) {
    throw new Error("Unexpected response when updating item");
  }

  const results = json.results;
  if (!assertRecord(results)) {
    throw new Error("Update response missing results payload");
  }

  const entry = results[safeItemId];
  if (!assertRecord(entry)) {
    throw new Error("Update response missing item status");
  }

  const updated = entry.updated;
  if (updated !== true) {
    const errorMessage =
      typeof entry.error === "string"
        ? entry.error
        : "Item update was not applied";
    throw new Error(errorMessage);
  }
}

/* === DELETE === */

export async function deleteItem(itemId: string): Promise<void> {
  const safeItemId = itemId?.trim();
  if (!safeItemId) {
    throw new Error("Item id is required for deletion");
  }

  const response = await authenticatedFetch("/items/delete", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      item_ids: [safeItemId],
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to delete item ${safeItemId}: ${response.status}`);
  }

  const json = (await response.json()) as unknown;
  if (!assertRecord(json)) {
    throw new Error("Unexpected response when deleting item");
  }

  const results = json.results;
  if (!assertRecord(results)) {
    throw new Error("Delete response missing results payload");
  }

  const deleted = results[safeItemId];
  if (deleted !== true) {
    throw Object.assign(new Error("Item not found"), { code: 404 });
  }
}
