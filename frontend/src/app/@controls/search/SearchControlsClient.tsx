"use client";

import { useCallback } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

import { ControlStrip } from "../default";
import { DropdownComponent, type DropdownOption } from "../../_components/IO";
import type { SearchMode } from "@/app/_lib/search";
import { useSetting } from "@/app/_contexts/SettingsContext";

interface SearchControlsClientProps {
  modeOptions: DropdownOption[];
  defaults: {
    mode: SearchMode;
  };
}

export default function SearchControlsClient({
  modeOptions,
  defaults,
}: SearchControlsClientProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  // Use global settings with URL sync
  const [mode, setMode] = useSetting("mode");

  const updateParam = useCallback(
    (key: "mode", value: string, defaultValue: string) => {
      const currentQuery = searchParams.toString();
      const next = new URLSearchParams(currentQuery);
      if (value === defaultValue) {
        next.delete(key);
      } else {
        next.set(key, value);
      }

      const nextQuery = next.toString();
      const target = nextQuery ? `${pathname}?${nextQuery}` : pathname;
      const current = currentQuery ? `${pathname}?${currentQuery}` : pathname;

      if (target === current) {
        return;
      }

      router.replace(target);
    },
    [pathname, router, searchParams],
  );

  const handleModeSelect = useCallback(
    (option: DropdownOption) => {
      const newMode = option.value as SearchMode;
      if (newMode === mode) {
        return;
      }

      // Update global settings and URL
      setMode(newMode);
      updateParam("mode", newMode, defaults.mode);
    },
    [mode, updateParam, setMode, defaults.mode],
  );

  return (
    <>
      <ControlStrip
        label="Type"
        io={DropdownComponent}
        ioProps={{
          options: modeOptions,
          selectedValue: mode,
          onSelect: handleModeSelect,
          placeholder: "Select type",
        }}
      />
    </>
  );
}
