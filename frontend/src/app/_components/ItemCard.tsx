"use client";

import { useEffect, useMemo, useState } from "react";
import type { MouseEvent, PointerEvent, ReactNode } from "react";
import { Icon } from "@iconify/react";
import Image from "next/image";
import type { ItemSummary, ClientStatus } from "@/app/_lib/items";
import { deriveLoadingStage, type ItemLoadingStage } from "@/app/_lib/queue";
import ItemActionPanel from "./ItemActionPanel";

/* === UTILITIES === */

function formatSavedAt(savedAt?: ItemSummary["created_at"]) {
  if (!savedAt) {
    return null;
  }

  const numeric = Number(savedAt);
  if (!Number.isNaN(numeric)) {
    const millisecondsThreshold = 10 ** 12;
    const timestamp =
      numeric < millisecondsThreshold ? numeric * 1000 : numeric;
    return new Date(timestamp).toLocaleDateString("en-US", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });
  }

  const parsed = Date.parse(savedAt);
  return Number.isNaN(parsed)
    ? null
    : new Date(parsed).toLocaleDateString("en-UK", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
      });
}

function estimateReadingTime(tokenCount: number | string): string {
  if (!tokenCount) return "-- min";

  const tokens =
    typeof tokenCount === "string" ? Number(tokenCount) : tokenCount;
  if (Number.isNaN(tokens) || tokens <= 0) return "--:--";

  const wordCount = Math.ceil(tokens * 0.75);
  const minutes = Math.ceil(wordCount / 200);
  const roundedMinutes = Math.ceil(minutes / 5) * 5;

  return `${roundedMinutes} min`;
}

/* === INFO PANELS === */

type InfoPanelProps = {
  icon: ReactNode;
  label: string;
};

function InfoPanel({ icon, label }: InfoPanelProps) {
  return (
    <div className="flex items-center gap-[5px] px-[10px] rounded-xl bg-transparent">
      <span className="text-[10px] font-semibold font-mono text-black">
        {label}
      </span>
      {icon && (
        <div className="w-[15px] h-[15px] flex items-center justify-center text-current">
          {icon}
        </div>
      )}
    </div>
  );
}

function SkeletonBlock({ className }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded bg-slate-200/80 ${className ?? ""}`}
    />
  );
}

/* === ITEM CARD === */

type ItemCardProps = {
  item: ItemSummary;
  stage?: ItemLoadingStage;
  isOptimistic?: boolean;
  addingError?: string | null;
  onDelete?: (itemId: string, options?: { alreadyDeleted?: boolean }) => void;
};

export default function ItemCard({
  item,
  stage,
  isOptimistic = false,
  addingError = null,
  onDelete,
}: ItemCardProps) {
  const [currentStatus, setCurrentStatus] = useState<ClientStatus>(
    item.client_status,
  );
  const [isStatusExpanded, setIsStatusExpanded] = useState(false);
  const [faviconError, setFaviconError] = useState(false);

  const isAdding = currentStatus === "adding";
  const hasError = Boolean(addingError);

  useEffect(() => {
    setCurrentStatus(item.client_status);
  }, [item.client_status]);

  useEffect(() => {
    if (hasError && isStatusExpanded) {
      setIsStatusExpanded(false);
    }
  }, [hasError, isStatusExpanded]);

  const handleStatusUpdate = (newStatus: ClientStatus) => {
    setCurrentStatus(newStatus);
  };

  const loadingStage = useMemo(
    () => stage ?? deriveLoadingStage(item),
    [item, stage],
  );

  const faviconUrl = item.favicon_url;
  const savedDate = formatSavedAt(item.created_at);
  const readingTime = estimateReadingTime(item.content_token_count || 0);

  const showMetadata =
    loadingStage !== "creating" || Boolean(item.title || item.source_site);

  const showSummaryContent = Boolean(item.summary);
  const showSummary =
    loadingStage === "embedding" || loadingStage === "complete";

  const showSummarySkeleton =
    !showSummaryContent && loadingStage !== "complete";

  const showReadingTime =
    loadingStage === "complete" && Boolean(item.content_token_count);

  const showSavedDate = Boolean(savedDate);

  const showActions = !isAdding && loadingStage !== "creating" && !hasError;

  const downloadIcon = (
    <Icon icon="mingcute:download-fill" width="13" height="13" />
  );
  const clockIcon = <Icon icon="mingcute:time-fill" width="13" height="13" />;

  const handleCardClick = (event: MouseEvent<HTMLElement>) => {
    if (isStatusExpanded || hasError) {
      event.preventDefault();
      event.stopPropagation();
    }
  };

  const handlePointerDown = (event: PointerEvent<HTMLElement>) => {
    if (isStatusExpanded || hasError) {
      event.preventDefault();
      event.stopPropagation();
    }
  };

  const handleDismissError = (event: MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (onDelete) {
      onDelete(item.id);
    }
  };

  const borderClass =
    loadingStage === "complete" ? "border-transparent" : "border-slate-200";

  const cardBorderClass = isAdding
    ? "border-slate-300 border-dashed"
    : borderClass;

  const cardBackgroundClass = isAdding ? "bg-slate-50" : "bg-white";

  const optimisticClass =
    !addingError && (isOptimistic || isAdding) ? "animate-pulse" : "";

  return (
    <article
      className="relative group cursor-default"
      onClick={handleCardClick}
      onPointerDown={handlePointerDown}
      style={{ transformOrigin: "center top" }}
    >
      <div
        className={`flex rounded-2xl overflow-hidden border transition-all hover:scale-101 hover:drop-shadow-md duration-300 ease-out ${cardBackgroundClass} ${cardBorderClass} ${optimisticClass}`}
      >
        {/* Favicon/Icon Section */}
        <div
          className={`flex justify-center ${isAdding ? "bg-slate-100" : "bg-white"} p-[15px]`}
        >
          <div className="flex justify-center items-center w-[45px] rounded-xl h-[45px] bg-[#D8D8D8]">
            <div className="bg-gray-100 h-[35px] w-[35px] rounded-[8px] flex items-center justify-center border border-gray-200">
              {faviconUrl && !faviconError ? (
                <Image
                  src={faviconUrl}
                  alt={`${item.source_site ?? item.title ?? "Item"} favicon`}
                  className="w-[35px] h-[35px] object-contain rounded-[8px]"
                  width={35}
                  height={35}
                  onError={() => setFaviconError(true)}
                />
              ) : (
                <Icon
                  icon="streamline-plump:web"
                  width="20"
                  height="20"
                  className="text-gray-400"
                />
              )}
            </div>
          </div>
        </div>

        {/* Main Content Section */}
        <div className="flex flex-col px-[15px] py-[12px] flex-1 gap-[6px]">
          <div className="min-h-[20px] flex items-center">
            {showMetadata ? (
              <h2 className="text-[14px] font-bold text-black line-clamp-1">
                {item.title ?? item.source_site ?? item.url ?? "Untitled"}
              </h2>
            ) : (
              <SkeletonBlock className="h-[20px] w-3/4" />
            )}
          </div>
          {showSummary ? (
            showSummaryContent ? (
              <p className="text-[9pt] text-black text-left line-clamp-3 font-sans">
                {item.summary}
              </p>
            ) : null
          ) : showSummarySkeleton ? (
            <div className="flex flex-col gap-[4px] min-h-[48px]">
              <SkeletonBlock className="h-[12px] w-full max-w-[220px]" />
              <SkeletonBlock className="h-[12px] w-3/4 max-w-[180px]" />
              <SkeletonBlock className="h-[12px] w-4/5 max-w-[200px]" />
            </div>
          ) : null}
        </div>

        {/* Info Panels Section */}
        <div className="flex flex-col items-end justify-start gap-[6px] p-[15px] ml-auto">
          {!hasError && (
            <div
              className={`relative transition-opacity duration-200 ${showActions ? "opacity-100" : "opacity-40 pointer-events-none"} ${isStatusExpanded ? "z-40" : "z-10"}`}
            >
              <ItemActionPanel
                itemId={item.id}
                currentStatus={currentStatus}
                onStatusUpdate={handleStatusUpdate}
                isExpanded={isStatusExpanded}
                onExpandedChange={setIsStatusExpanded}
                onDelete={onDelete}
              />
            </div>
          )}

          <div
            className={`flex flex-col items-end gap-[6px] transition-opacity duration-200 ${
              isStatusExpanded
                ? "opacity-0 pointer-events-none"
                : "opacity-100 pointer-events-auto"
            }`}
          >
            {showReadingTime ? (
              <InfoPanel icon={clockIcon} label={readingTime} />
            ) : (
              <div className="flex items-center gap-[5px] px-[10px] rounded-xl bg-transparent h-[20px]">
                <SkeletonBlock className="h-[12px] w-[35px]" />
                <SkeletonBlock className="h-[15px] w-[15px]" />
              </div>
            )}

            {showSavedDate ? (
              <InfoPanel icon={downloadIcon} label={savedDate ?? ""} />
            ) : (
              <div className="flex items-center gap-[5px] px-[10px] rounded-xl bg-transparent h-[20px]">
                <SkeletonBlock className="h-[12px] w-[30px]" />
                <SkeletonBlock className="h-[15px] w-[15px]" />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Adding Hover Cancel Overlay */}
      {isAdding && !hasError && (
        <div
          className="absolute inset-0 z-10 rounded-2xl flex items-center justify-center bg-white/70 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none group-hover:pointer-events-auto"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
        >
          <button
            className="px-6 py-2 rounded-[20px] panel-light text-slate-700 text-sm font-semibold transition-all duration-200 hover:bg-slate-200 hover:drop-shadow-sm hover:scale-105"
            title="Cancel adding"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              onDelete?.(item.id);
            }}
          >
            Cancel
          </button>
        </div>
      )}

      {/* Error Overlay */}
      {hasError && (
        <div className="absolute inset-0 bg-white border-2 border-red-500 rounded-2xl flex flex-col justify-center p-4">
          {/* Close Button */}
          <button
            onClick={handleDismissError}
            className="absolute top-2 right-2 w-6 h-6 bg-red-500 hover:bg-red-600 rounded-full flex items-center justify-center transition-colors duration-200 shadow-sm z-10"
            title="Dismiss error"
          >
            <Icon
              icon="mingcute:close-fill"
              width="12"
              height="12"
              className="text-white"
            />
          </button>

          {/* Error Content */}
          <div className="text-center">
            <div className="flex items-center justify-center gap-2 mb-3">
              <Icon
                icon="mingcute:alert-fill"
                width="20"
                height="20"
                className="text-red-500"
              />
              <h3 className="text-sm font-semibold text-red-700">
                Error for URL: {item.url}
              </h3>
            </div>
            <p className="text-xs text-red-600 leading-relaxed">
              {addingError}
            </p>
          </div>
        </div>
      )}
    </article>
  );
}
