import { ControlStrip } from "../default";
import { DropdownComponent, type DropdownOption } from "../../_components/IO";
import type { SearchMode } from "@/app/_lib/search";

import SearchControlsClient from "./SearchControlsClient";

export const modeOptions: DropdownOption[] = [
  { value: "lexical", label: "Lexical" },
  { value: "semantic", label: "Semantic" },
  { value: "agentic", label: "Agentic" },
];

export const DEFAULT_MODE: SearchMode = "lexical";

export default function SearchControls() {
  const defaults = {
    mode: DEFAULT_MODE,
  };

  return (
    <SearchControlsClient
      modeOptions={modeOptions}
      defaults={defaults}
    />
  );
}
