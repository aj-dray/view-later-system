export const dynamic = "force-dynamic";

import { redirect } from "next/navigation";

import { getSession } from "@/app/_lib/auth";
import {
  searchItems,
  type SearchMode,
} from "@/app/_lib/search";
import { MAX_SEARCH_LIMIT } from "@/app/_lib/search-config";
import type { SearchResponse } from "@/app/_lib/search";
import SearchClient from "./SearchClient";

type SearchPageProps = {
  searchParams: Promise<{
    q?: string;
    mode?: string;
  }>;
};

function sanitiseMode(value: string | undefined): SearchMode {
  if (value === "semantic" || value === "agentic") {
    return value;
  }
  return "lexical";
}

export default async function SearchPage({ searchParams }: SearchPageProps) {
  const session = await getSession();
  if (!session) {
    redirect("/login");
  }

  const resolvedParams = await searchParams;
  const query = resolvedParams?.q?.trim() ?? "";
  const mode = sanitiseMode(resolvedParams?.mode);

  // Do not auto-run searches on refresh or mode changes.
  // Always start with empty results; user must press Enter or the button.
  const initialData: SearchResponse = { results: [], agent: null };

  return (
    <div
      className="flex h-full w-full flex-col bg-[#F0F0F0] overflow-auto"
      style={{
        paddingRight: "calc(var(--panels-width, 0px) + 25px",
      }}
    >
      <SearchClient initialQuery={query} initialMode={mode} initialData={initialData} />
    </div>
  );
}
