"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { usePathname } from "next/navigation";
import { updateUserControl, type UserControlStates } from "@/app/_lib/user-controls";
import { normalizePagePath } from "@/app/_lib/user-controls-utils";
import type { QueueTypeFilter } from "@/app/_lib/queue";

// Define the shape of all possible settings
export interface PageSettings {
  // Queue page settings
  order?: "date" | "random" | "priority";
  filter?: "queued" | "all";
  typeFilter?: QueueTypeFilter;
  // Search page settings
  mode?: "lexical" | "semantic" | "agentic";
  // Graph page settings
  visualisation?: "pca" | "tsne" | "umap";
  clustering?: "kmeans" | "hca" | "dbscan";
  clusters?: number;
  eps?: number;
}

// Default settings for each page
export const PAGE_DEFAULTS: Record<string, PageSettings> = {
  "queue": {
    order: "date",
    filter: "queued",
    typeFilter: "all",
  },
  "search": {
    mode: "lexical",
  },
  "graph": {
    filter: "queued",
    visualisation: "umap",
    clustering: "kmeans",
    clusters: 5,
    eps: 0.3,
  },
};

interface SettingsContextValue {
  // Get current page settings
  getPageSettings: () => PageSettings;

  // Get specific setting with fallback to default
  getSetting: <T extends keyof PageSettings>(key: T) => PageSettings[T];

  // Update a setting for current page
  updateSetting: <T extends keyof PageSettings>(key: T, value: PageSettings[T]) => Promise<void>;

  // Update multiple settings for current page
  updatePageSettings: (settings: Partial<PageSettings>) => Promise<void>;

  // Loading state
  loading: boolean;

  // Error state
  error: string | null;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

interface SettingsProviderProps {
  children: React.ReactNode;
  initialSettings?: Record<string, UserControlStates>;
}

export function SettingsProvider({
  children,
  initialSettings = {}
}: SettingsProviderProps) {
  const pathname = usePathname();
  const [allSettings, setAllSettings] = useState<Record<string, UserControlStates>>(initialSettings);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const currentPagePath = normalizePagePath(pathname);

  // Keep local state in sync when server-provided initial settings change
  // This happens on auth transitions (login/logout) and SSR navs
  useEffect(() => {
    setAllSettings(initialSettings);
  }, [initialSettings]);

  // Get current page settings with defaults
  const getPageSettings = useCallback((): PageSettings => {
    const pageDefaults = PAGE_DEFAULTS[currentPagePath] || {};
    const userSettings = allSettings[currentPagePath] || {};
    return { ...pageDefaults, ...userSettings };
  }, [currentPagePath, allSettings]);

  // Get specific setting with fallback to default
  const getSetting = useCallback(<T extends keyof PageSettings>(key: T): PageSettings[T] => {
    const pageSettings = getPageSettings();
    return pageSettings[key];
  }, [getPageSettings]);

  // Update a single setting
  const updateSetting = useCallback(async <T extends keyof PageSettings>(
    key: T,
    value: PageSettings[T]
  ): Promise<void> => {
    if (value === undefined) return;

    try {
      setLoading(true);
      setError(null);

      // Optimistically update local state
      setAllSettings(prev => ({
        ...prev,
        [currentPagePath]: {
          ...prev[currentPagePath],
          [key]: value,
        },
      }));

      // Save to backend
      const success = await updateUserControl(currentPagePath, key as string, value);

      if (!success) {
        // Revert on failure
        setAllSettings(prev => {
          const pageSettings = { ...prev[currentPagePath] };
          const pageDefaults = PAGE_DEFAULTS[currentPagePath] || {};

          if (pageDefaults[key] !== undefined) {
            pageSettings[key as string] = pageDefaults[key];
          } else {
            delete pageSettings[key as string];
          }

          return {
            ...prev,
            [currentPagePath]: pageSettings,
          };
        });
        setError("Failed to save setting");
      }
    } catch (err) {
      // Revert on error
      setAllSettings(prev => {
        const pageSettings = { ...prev[currentPagePath] };
        const pageDefaults = PAGE_DEFAULTS[currentPagePath] || {};

        if (pageDefaults[key] !== undefined) {
          pageSettings[key as string] = pageDefaults[key];
        } else {
          delete pageSettings[key as string];
        }

        return {
          ...prev,
          [currentPagePath]: pageSettings,
        };
      });
      setError(err instanceof Error ? err.message : "Failed to save setting");
      console.error("Error updating setting:", err);
    } finally {
      setLoading(false);
    }
  }, [currentPagePath]);

  // Update multiple settings
  const updatePageSettings = useCallback(async (settings: Partial<PageSettings>): Promise<void> => {
    try {
      setLoading(true);
      setError(null);

      const previousSettings = allSettings[currentPagePath] || {};

      // Optimistically update local state
      setAllSettings(prev => ({
        ...prev,
        [currentPagePath]: {
          ...prev[currentPagePath],
          ...settings,
        },
      }));

      // Save each setting to backend
      const promises = Object.entries(settings).map(([key, value]) =>
        updateUserControl(currentPagePath, key, value)
      );

      const results = await Promise.all(promises);
      const hasFailure = results.some(success => !success);

      if (hasFailure) {
        // Revert on any failure
        setAllSettings(prev => ({
          ...prev,
          [currentPagePath]: previousSettings,
        }));
        setError("Failed to save some settings");
      }
    } catch (err) {
      // Revert on error
      setAllSettings(prev => ({
        ...prev,
        [currentPagePath]: allSettings[currentPagePath] || {},
      }));
      setError(err instanceof Error ? err.message : "Failed to save settings");
      console.error("Error updating settings:", err);
    } finally {
      setLoading(false);
    }
  }, [currentPagePath, allSettings]);

  const contextValue: SettingsContextValue = {
    getPageSettings,
    getSetting,
    updateSetting,
    updatePageSettings,
    loading,
    error,
  };

  return (
    <SettingsContext.Provider value={contextValue}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error("useSettings must be used within a SettingsProvider");
  }
  return context;
}

// Convenience hook for getting a specific setting
export function useSetting<T extends keyof PageSettings>(key: T): [
  PageSettings[T],
  (value: PageSettings[T]) => Promise<void>
] {
  const { getSetting, updateSetting } = useSettings();

  const value = getSetting(key);
  const setValue = useCallback((newValue: PageSettings[T]) =>
    updateSetting(key, newValue), [key, updateSetting]);

  return [value, setValue];
}
