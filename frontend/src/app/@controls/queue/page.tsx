import { ControlStrip } from "../default";
import { DropdownComponent, type DropdownOption } from "../../_components/IO";
import type {
  QueueFilter,
  QueueOrder,
  QueueTypeFilter,
} from "@/app/_lib/queue";
import QueueControlsClient from "./QueueControlsClient";

export const orderOptions: DropdownOption[] = [
  { value: "date", label: "Date Added" },
  { value: "random", label: "Random" },
  { value: "priority", label: "Priority" },
];

export const filterOptions: DropdownOption[] = [
  { value: "queued", label: "Queued" },
  { value: "all", label: "All" },
];

export const typeFilterOptions: DropdownOption[] = [
  { value: "all", label: "All" },
  { value: "article", label: "Article" },
  { value: "video", label: "Video" },
  { value: "paper", label: "Paper" },
  { value: "podcast", label: "Podcast" },
  { value: "post", label: "Post" },
  { value: "newsletter", label: "Newsletter" },
  { value: "other", label: "Other" },
];

export const DEFAULT_ORDER: QueueOrder = "date";
export const DEFAULT_FILTER: QueueFilter = "queued";
export const DEFAULT_TYPE_FILTER: QueueTypeFilter = "all";

export default function QueueControls() {
  const defaults = {
    order: DEFAULT_ORDER,
    filter: DEFAULT_FILTER,
    typeFilter: DEFAULT_TYPE_FILTER,
  };

  return (
    <QueueControlsClient
      orderOptions={orderOptions}
      filterOptions={filterOptions}
      typeFilterOptions={typeFilterOptions}
      defaults={defaults}
    />
  );
}
