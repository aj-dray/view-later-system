"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Icon } from "@iconify/react";
import type { GmailSettings } from "@/app/_lib/gmail";
import {
  addGmailSenderAction,
  deleteGmailSenderAction,
  disconnectGmailAction,
  getGmailSettingsAction,
  startGmailAuthAction,
  pollGmailAction,
} from "@/app/_actions/gmail";
import {
  createApiKeyAction,
  deleteApiKeyAction,
  getApiKeysAction,
  type ApiKeySummary,
} from "@/app/_actions/api-keys";
import { signOutAction } from "@/app/login/actions";
import SettingsList from "@/app/_components/SettingsList";

type SettingsClientProps = {
  username: string | null;
  gmailStatus: string | null;
};

type ApiKeyRecord = ApiKeySummary;

export default function SettingsClient({ username }: SettingsClientProps) {
  const [settings, setSettings] = useState<GmailSettings | null>(null);
  const [settingsLoading, setSettingsLoading] = useState(true);
  const [apiKeys, setApiKeys] = useState<ApiKeyRecord[]>([]);
  const [apiKeysLoading, setApiKeysLoading] = useState(true);
  const [newApiKey, setNewApiKey] = useState<string | null>(null);
  const [polling, setPolling] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const data = await getGmailSettingsAction();
        setSettings(data);

        // If we have credentials and senders but polling isn't active, trigger a poll to start it
        if (data.connected && data.has_senders && !data.polling_active) {
          try {
            await pollGmailAction();
            // Refresh settings to show polling is now active
            const refreshed = await getGmailSettingsAction();
            setSettings(refreshed);
          } catch (pollErr) {
            console.error("Failed to start polling:", pollErr);
          }
        }
      } catch (err) {
        console.error("Failed to load Gmail settings:", err);
        // Set empty settings on error to allow page to render
        setSettings({
          connected: false,
          email_address: null,
          token_expiry: null,
          updated_at: null,
          senders: [],
          default_senders: [],
          oauth_available: true,
          legacy_credentials: false,
          has_senders: false,
          polling_active: false,
          polling_interval_seconds: 300,
          last_poll_at: null,
        });
      } finally {
        setSettingsLoading(false);
      }
    };
    const fetchApiKeys = async () => {
      try {
        const data = await getApiKeysAction();
        setApiKeys(data);
      } catch (err) {
        console.error("Failed to load API keys:", err);
      } finally {
        setApiKeysLoading(false);
      }
    };

    void fetchSettings();
    void fetchApiKeys();
  }, []);

  const hasConnection = Boolean(settings?.connected);
  const gmailEmail = settings?.email_address || "";

  const handleConnectClick = useCallback(async () => {
    try {
      const { auth_url: authUrl } = await startGmailAuthAction();
      if (authUrl) {
        window.location.href = authUrl;
      }
    } catch (err) {
      console.error("Failed to start Gmail auth:", err);
    }
  }, []);

  const handleDisconnect = useCallback(async () => {
    try {
      await disconnectGmailAction();
      setSettings((prev) =>
        prev ? { ...prev, connected: false, email_address: null } : null,
      );
    } catch (err) {
      console.error("Failed to disconnect:", err);
    }
  }, []);

  const handleAddSender = useCallback(async (email: string) => {
    const sender = await addGmailSenderAction(email);
    setSettings((prev) =>
      prev ? { ...prev, senders: [...prev.senders, sender] } : prev,
    );
  }, []);

  const handleRemoveSender = useCallback(async (id: string) => {
    await deleteGmailSenderAction(id);
    setSettings((prev) =>
      prev
        ? {
            ...prev,
            senders: prev.senders.filter((s) => s.id !== id),
          }
        : prev,
    );
  }, []);

  const handleAddApiKey = useCallback(async (label: string) => {
    try {
      const { apiKey, token } = await createApiKeyAction(label);
      setApiKeys((prev) => [
        token,
        ...prev.filter((existing) => existing.tokenId !== token.tokenId),
      ]);
      setNewApiKey(apiKey);
    } catch (err) {
      console.error("Failed to create API key:", err);
      throw err;
    }
  }, []);

  const handleRemoveApiKey = useCallback(async (tokenId: string) => {
    try {
      await deleteApiKeyAction(tokenId);
    } catch (err) {
      console.error("Failed to delete API key:", err);
    } finally {
      setApiKeys((prev) => prev.filter((k) => k.tokenId !== tokenId));
    }
  }, []);

  const handleCopyApiKey = useCallback(() => {
    if (newApiKey) {
      navigator.clipboard.writeText(newApiKey);
    }
  }, [newApiKey]);

  const handleLogout = useCallback(async () => {
    await signOutAction();
    router.push("/login");
  }, [router]);

  const handlePollGmail = useCallback(async () => {
    try {
      setPolling(true);
      await pollGmailAction();
      // Refresh settings to get updated last_poll_at
      const data = await getGmailSettingsAction();
      setSettings(data);
    } catch (err) {
      console.error("Failed to poll Gmail:", err);
    } finally {
      setPolling(false);
    }
  }, []);

  const formatLastPoll = (lastPollAt: string | null): string => {
    if (!lastPollAt) return "Never";
    const date = new Date(lastPollAt);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins} min ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? "s" : ""} ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays} day${diffDays > 1 ? "s" : ""} ago`;
  };

  return (
    <div
      className="flex flex-col gap-[20px] py-[42px] text-black relative min-h-screen w-full max-w-[800px]"
      style={{
        paddingLeft: "77px",
        paddingRight: "77px",
      }}
    >
      {/* Back to application button - positioned where settings pill would be */}
      <button
        type="button"
        onClick={() => router.push("/queue")}
        className="fixed bottom-[25px] right-[25px] z-50 flex items-center gap-[10px] rounded-[25px] bg-[#C3C3C3] px-[10px] h-[40px] text-[12px] font-medium text-black hover:scale-105 hover:drop-shadow-md hover:bg-[#A8A8A8] transition-all duration-200"
      >
        <span>back to application</span>
        <Icon icon="mdi:arrow-left" width={24} height={24} />
      </button>

      {/* Settings Header */}
      <div className="flex flex-col gap-[10px] py-[20px]">
        <h1 className="text-[40px] font-bold leading-[1.21]">Settings</h1>
      </div>

      {/* Account Section */}
      <div className="panel-light flex flex-col gap-[15px] rounded-[20px] p-[20px]">
        <h2 className="text-[25px] font-bold leading-[1.21]">Account</h2>

        <div className="flex flex-col gap-[15px]">
          {/* Username */}
          <div className="flex items-center gap-[15px]">
            <label className="text-[12px] font-medium min-w-[100px]">
              Username
            </label>
            <div className="flex items-center justify-center bg-white/80 rounded-[10px] px-[10px] h-[20px]">
              <span className="text-[12px] font-medium">{username || "—"}</span>
            </div>
          </div>

          {/* Password */}
          {/*<div className="flex items-center gap-[15px]">
            <label className="text-[12px] font-medium min-w-[100px]">
              Password
            </label>
            <button
              type="button"
              className="flex items-center justify-center bg-[#CDCDCD]/80 rounded-[10px] px-[10px] h-[20px] hover:scale-105 hover:drop-shadow-md hover:bg-[#B0B0B0]/90 transition-all duration-200"
            >
              <span className="text-[12px] font-medium">change password</span>
            </button>
          </div>*/}

          {/* Logout */}
          <div className="flex items-center gap-[15px]">
            <label className="text-[12px] font-medium min-w-[100px]">
              Session
            </label>
            <button
              type="button"
              onClick={handleLogout}
              className="flex items-center justify-center bg-[#CDCDCD]/80 rounded-[10px] px-[10px] h-[20px] hover:scale-105 hover:drop-shadow-md hover:bg-[#B0B0B0]/90 transition-all duration-200"
            >
              <span className="text-[12px] font-medium">log out</span>
            </button>
          </div>
        </div>
      </div>

      {/* API Keys Section */}
      <div className="panel-light flex flex-col gap-[15px] rounded-[20px] p-[20px]">
        <h2 className="text-[25px] font-bold leading-[1.21]">API Keys</h2>

        <div className="flex gap-[15px]">
          <label className="text-[12px] font-medium min-w-[100px] h-[20px] flex items-center">
            Key List
          </label>
          <SettingsList
            items={apiKeys.map((apiKey) => ({
              id: apiKey.tokenId,
              value: apiKey.label ?? "unnamed key",
            }))}
            onAdd={handleAddApiKey}
            onRemove={handleRemoveApiKey}
            placeholder="key label"
            inputType="text"
            disabled={apiKeysLoading}
          />
        </div>
      </div>

      {/* Newsletters Section */}
      <div className="panel-light flex flex-col gap-[15px] rounded-[20px] p-[20px]">
        <h2 className="text-[25px] font-bold leading-[1.21]">Newsletters</h2>

        {/* Gmail */}
        <div className="flex flex-col gap-[15px]">
          <div className="flex items-center gap-[15px]">
            <label className="text-[12px] font-medium min-w-[100px]">
              Gmail
            </label>
            {hasConnection ? (
              <>
                <div className="flex items-center justify-center bg-white/80 rounded-[10px] px-[10px] h-[20px]">
                  <span className="text-[12px] font-medium">{gmailEmail}</span>
                </div>
                <button
                  type="button"
                  onClick={handleDisconnect}
                  className="flex items-center justify-center bg-[#CDCDCD]/80 rounded-[10px] px-[10px] h-[20px] hover:scale-105 hover:drop-shadow-md hover:bg-[#B0B0B0]/90 transition-all duration-200"
                >
                  <span className="text-[12px] font-medium">
                    deauthenticate
                  </span>
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={handleConnectClick}
                className="flex items-center justify-center bg-[#CDCDCD]/80 rounded-[10px] px-[10px] h-[20px] hover:scale-105 hover:drop-shadow-md hover:bg-[#B0B0B0]/90 transition-all duration-200"
              >
                <span className="text-[12px] font-medium">authenticate</span>
              </button>
            )}
          </div>

          {/* Sync Status */}
          <div className="flex items-center gap-[15px]">
            <label className="text-[12px] font-medium min-w-[100px]">
              Sync
            </label>
            <div className="flex items-center justify-center bg-white/80 rounded-[10px] px-[10px] h-[20px]">
              <span className="text-[12px] font-medium">
                {formatLastPoll(settings?.last_poll_at || null)}
              </span>
            </div>
            <button
              type="button"
              onClick={handlePollGmail}
              disabled={polling || !hasConnection}
              className="flex items-center justify-center bg-[#CDCDCD]/80 rounded-[10px] px-[10px] h-[20px] hover:scale-105 hover:drop-shadow-md hover:bg-[#B0B0B0]/90 transition-all duration-200 disabled:opacity-50 disabled:hover:scale-100 disabled:hover:drop-shadow-none"
            >
              <span className="text-[12px] font-medium">
                {polling ? "refreshing..." : "refresh"}
              </span>
            </button>
          </div>

          {/* Email List */}
          <div className="flex gap-[15px]">
            <label className="text-[12px] font-medium min-w-[100px] h-[20px] flex items-center">
              Email List
            </label>
            <SettingsList
              items={
                settings?.senders.map((sender) => ({
                  id: sender.id || "",
                  value: sender.email_address,
                })) || []
              }
              onAdd={handleAddSender}
              onRemove={handleRemoveSender}
              placeholder="newsletter@example.com"
              inputType="email"
              disabled={settingsLoading}
            />
          </div>
        </div>
      </div>

      {/* API Key Popup */}
      {newApiKey && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50">
          <div className="panel-light rounded-[20px] p-[30px] max-w-[500px] mx-4">
            <h3 className="text-[20px] font-bold mb-4">New API Key Created</h3>
            <p className="text-[14px] mb-4">
              Copy this API key now. You won’t be able to see it again!
            </p>
            <div className="flex gap-2 items-center mb-6">
              <input
                type="text"
                value={newApiKey}
                readOnly
                className="flex-1 bg-white/80 rounded-[10px] px-[10px] py-2 text-[12px] font-mono border border-gray-300"
              />
              <button
                type="button"
                onClick={handleCopyApiKey}
                className="flex items-center gap-2 bg-black text-white rounded-[10px] px-4 py-2 text-[12px] font-medium hover:scale-105 hover:drop-shadow-md hover:bg-gray-800 transition-all duration-200"
              >
                <Icon icon="mdi:content-copy" width={16} height={16} />
                Copy
              </button>
            </div>
            <button
              type="button"
              onClick={() => setNewApiKey(null)}
              className="w-full bg-[#CDCDCD]/80 rounded-[10px] px-4 py-2 text-[12px] font-medium hover:scale-105 hover:drop-shadow-md hover:bg-[#B0B0B0]/90 transition-all duration-200"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
