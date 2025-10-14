"use server";

import { getSession } from "@/app/_lib/auth";
import { getServiceBaseUrl } from "@/app/_lib/utils";
import type { GmailSettings, GmailSender } from "@/app/_lib/gmail";

const baseUrl = getServiceBaseUrl();

async function requireAuth() {
  const session = await getSession();
  if (!session) {
    throw new Error("Not authenticated");
  }
  return session;
}

export async function getGmailSettingsAction(): Promise<GmailSettings> {
  const session = await requireAuth();

  const response = await fetch(`${baseUrl}/gmail/settings`, {
    headers: {
      Authorization: `Bearer ${session.token}`,
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch Gmail settings: ${response.statusText}`);
  }

  return response.json();
}

export async function addGmailSenderAction(
  emailAddress: string,
  label?: string
): Promise<GmailSender> {
  const session = await requireAuth();

  const response = await fetch(`${baseUrl}/gmail/senders`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${session.token}`,
      "Content-Type": "application/json",
    },
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

export async function deleteGmailSenderAction(senderId: string): Promise<boolean> {
  const session = await requireAuth();

  const response = await fetch(`${baseUrl}/gmail/senders/${senderId}`, {
    method: "DELETE",
    headers: {
      Authorization: `Bearer ${session.token}`,
    },
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

export async function startGmailAuthAction(): Promise<{ auth_url: string }> {
  const session = await requireAuth();

  const response = await fetch(`${baseUrl}/gmail/auth/start`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${session.token}`,
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to start Gmail auth: ${response.statusText}`);
  }

  return response.json();
}

export async function disconnectGmailAction(): Promise<boolean> {
  const session = await requireAuth();

  const response = await fetch(`${baseUrl}/gmail/auth/disconnect`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${session.token}`,
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to disconnect Gmail: ${response.statusText}`);
  }

  const data = await response.json();
  return data.disconnected === true;
}

export async function pollGmailAction(): Promise<{
  imported: Array<{ item_id: string; message_id: string }>;
  duplicates: string[];
  skipped: Array<{ message_id: string; reason: string }>;
}> {
  const session = await requireAuth();

  const response = await fetch(`${baseUrl}/gmail/poll`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${session.token}`,
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to poll Gmail: ${response.statusText}`);
  }

  return response.json();
}
