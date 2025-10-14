/* === DEFINITIONS === */

export const CLIENT_UPDATABLE_STATUSES = [
  "queued",
  "paused",
  "completed",
  "bookmark",
  "error",
] as const;

export const CLIENT_STATUSES = [
  "adding",
  ...CLIENT_UPDATABLE_STATUSES,
] as const;

export const SERVER_STATUSES = [
  "saved",
  "extracted",
  "summarised",
  "embedded",
  "classified",
] as const;

export type ClientStatus = (typeof CLIENT_STATUSES)[number];
export type ServerStatus = (typeof SERVER_STATUSES)[number];

export const ITEM_COLUMNS = [
  "id",
  "user_id",
  "url",
  "canonical_url",
  "title",
  "source_site",
  "format",
  "author",
  "type",
  "publication_date",
  "favicon_url",
  "content_markdown",
  "content_text",
  "content_token_count",
  "client_status",
  "server_status",
  "summary",
  "expiry_score",
  "ts_embedding",
  "mistral_embedding",
  "client_status_at",
  "server_status_at",
  "created_at",
] as const;

export type ItemColumn = (typeof ITEM_COLUMNS)[number];

export type ItemSummary = {
  id: string;
  user_id: string | null;
  url: string;
  canonical_url: string | null;
  title: string | null;
  source_site: string | null;
  format: string | null;
  author: string | null;
  type: string | null;
  publication_date: string | null;
  favicon_url: string | null;
  content_markdown: string | null;
  content_text: string | null;
  content_token_count: number | null;
  client_status: ClientStatus;
  server_status: ServerStatus;
  summary: string | null;
  expiry_score: number | null;
  ts_embedding: string | null;
  mistral_embedding: number[] | null;
  client_status_at: string | null;
  server_status_at: string | null;
  created_at: string;
};

export type ItemDetail = ItemSummary;

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

export type FilterOperator = (typeof FILTER_OPERATORS)[number];

type FilterPrimitive = string | number | boolean;
export type FilterValue = FilterPrimitive | readonly FilterPrimitive[];

export type ItemFilter = {
  column: ItemColumn;
  operator: FilterOperator;
  value: FilterValue;
};

const IMMUTABLE_ITEM_COLUMNS = ["id", "user_id", "created_at"] as const;
type ImmutableItemColumn = (typeof IMMUTABLE_ITEM_COLUMNS)[number];
export type MutableItemColumn = Exclude<ItemColumn, ImmutableItemColumn>;

export type FetchItemsOptions = {
  clientStatuses?: ClientStatus[];
  serverStatuses?: ServerStatus[];
  filters?: ItemFilter[];
  columns?: ItemColumn[];
  limit?: number;
  offset?: number;
  orderBy?: ItemColumn;
  order?: "asc" | "desc";
};

export type UpdateItemPayload = Partial<Pick<ItemSummary, MutableItemColumn>>;
