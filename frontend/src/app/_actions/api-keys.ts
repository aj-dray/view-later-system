"use server";

import { getSession } from "@/app/_lib/auth";
import { getServiceBaseUrl } from "@/app/_lib/utils";

export type ApiKeySummary = {
  tokenId: string;
  label: string | null;
  createdAt: string | null;
  expiresAt: string | null;
  revokedAt: string | null;
};

type TokenRow = {
  token_id?: unknown;
  label?: unknown;
  created_at?: unknown;
  expires_at?: unknown;
  revoked_at?: unknown;
};

type ListTokensResponse = {
  tokens?: TokenRow[];
};

type CreateTokenResponse = {
  access_token: string;
  token_id?: unknown;
  label?: unknown;
  created_at?: unknown;
  expires_at?: unknown;
};

const baseUrl = getServiceBaseUrl();

async function requireAuth() {
  const session = await getSession();
  if (!session) {
    throw new Error("Not authenticated");
  }
  return session;
}

function normaliseDate(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) {
    return value;
  }
  return null;
}

function normaliseTokenId(value: unknown): string {
  if (typeof value === "string" && value.trim()) {
    return value;
  }
  if (value != null) {
    return String(value);
  }
  throw new Error("Missing token id");
}

function toSummary(row?: TokenRow): ApiKeySummary {
  return {
    tokenId: normaliseTokenId(row?.token_id),
    label:
      typeof row?.label === "string" && row.label.trim()
        ? row.label
        : null,
    createdAt: normaliseDate(row?.created_at),
    expiresAt: normaliseDate(row?.expires_at),
    revokedAt: normaliseDate(row?.revoked_at),
  };
}

export async function getApiKeysAction(): Promise<ApiKeySummary[]> {
  const session = await requireAuth();

  const response = await fetch(`${baseUrl}/user/access-token`, {
    headers: {
      Authorization: `Bearer ${session.token}`,
    },
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(
      `Failed to fetch API keys: ${response.status} ${errorText}`.trim(),
    );
  }

  const data = (await response.json()) as ListTokensResponse;
  const rows = Array.isArray(data.tokens) ? data.tokens : [];
  return rows.map((row) => toSummary(row));
}

export async function createApiKeyAction(
  label: string,
): Promise<{ apiKey: string; token: ApiKeySummary }> {
  const session = await requireAuth();

  const response = await fetch(`${baseUrl}/user/access-token`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${session.token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ label }),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(
      `Failed to create API key: ${response.status} ${errorText}`.trim(),
    );
  }

  const data = (await response.json()) as CreateTokenResponse;
  const token = toSummary({
    token_id: data.token_id,
    label: data.label,
    created_at: data.created_at,
    expires_at: data.expires_at,
    revoked_at: null,
  });

  return {
    apiKey: data.access_token,
    token,
  };
}

export async function deleteApiKeyAction(tokenId: string): Promise<boolean> {
  const session = await requireAuth();

  const response = await fetch(`${baseUrl}/user/access-token/${tokenId}`, {
    method: "DELETE",
    headers: {
      Authorization: `Bearer ${session.token}`,
    },
  });

  if (response.status === 204) {
    return true;
  }
  if (response.status === 404) {
    return false;
  }

  const errorText = await response.text().catch(() => "");
  throw new Error(
    `Failed to delete API key: ${response.status} ${errorText}`.trim(),
  );
}
