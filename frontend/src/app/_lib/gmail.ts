/**
 * Gmail API client functions
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type GmailSender = {
  id: string;
  email_address: string;
  label: string | null;
  created_at: string;
};

export type GmailSettings = {
  connected: boolean;
  email_address: string | null;
  token_expiry: string | null;
  updated_at: string | null;
  senders: GmailSender[];
  default_senders: string[];
  oauth_available: boolean;
  legacy_credentials: boolean;
  has_senders: boolean;
  polling_active: boolean;
  polling_interval_seconds: number;
  last_poll_at: string | null;
};

export type GmailPollResult = {
  imported: Array<{
    item_id: string;
    message_id: string;
    slug: string;
    url: string;
    resolved_url: string | null;
  }>;
  duplicates: string[];
  skipped: Array<{
    message_id: string;
    reason: string;
  }>;
};

/**
 * Fetch current Gmail settings for the authenticated user
 */
export async function getGmailSettings(): Promise<GmailSettings> {
  const response = await fetch(`${API_BASE_URL}/gmail/settings`, {
    credentials: "include",
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch Gmail settings: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Poll Gmail for new newsletters
 */
export async function pollGmailNewsletters(
  maxResults?: number
): Promise<GmailPollResult> {
  const url = new URL(`${API_BASE_URL}/gmail/poll`);
  if (maxResults) {
    url.searchParams.set("max_results", String(maxResults));
  }

  const response = await fetch(url.toString(), {
    method: "POST",
    credentials: "include",
  });

  if (!response.ok) {
    throw new Error(`Failed to poll Gmail: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Add a new Gmail sender
 */
export async function addGmailSender(
  emailAddress: string,
  label?: string
): Promise<GmailSender> {
  const response = await fetch(`${API_BASE_URL}/gmail/senders`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    credentials: "include",
    body: JSON.stringify({
      email_address: emailAddress,
      label: label || null,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Failed to add sender: ${response.statusText}`);
  }

  const data = await response.json();
  return data.sender;
}

/**
 * Delete a Gmail sender
 */
export async function deleteGmailSender(senderId: string): Promise<boolean> {
  const response = await fetch(`${API_BASE_URL}/gmail/senders/${senderId}`, {
    method: "DELETE",
    credentials: "include",
  });

  if (!response.ok) {
    if (response.status === 404) {
      return false;
    }
    throw new Error(`Failed to delete sender: ${response.statusText}`);
  }

  const data = await response.json();
  return data.deleted === true;
}

/**
 * Start Gmail OAuth flow
 */
export async function startGmailAuth(): Promise<{ auth_url: string }> {
  const response = await fetch(`${API_BASE_URL}/gmail/auth/start`, {
    method: "POST",
    credentials: "include",
  });

  if (!response.ok) {
    throw new Error(`Failed to start Gmail auth: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Disconnect Gmail account
 */
export async function disconnectGmail(): Promise<boolean> {
  const response = await fetch(`${API_BASE_URL}/gmail/auth/disconnect`, {
    method: "POST",
    credentials: "include",
  });

  if (!response.ok) {
    throw new Error(`Failed to disconnect Gmail: ${response.statusText}`);
  }

  const data = await response.json();
  return data.disconnected === true;
}
