"use client";

import { useEffect, useState } from "react";
import { Icon } from "@iconify/react";
import { usePathname, useRouter } from "next/navigation";

import { updateItemStatusInQueue } from "@/app/queue/actions";
import { deleteItemAction } from "@/app/_actions/items";
import type { ClientStatus } from "@/app/_lib/items-types";
import { CLIENT_UPDATABLE_STATUSES } from "@/app/_lib/items-types";

type ClientUpdatableStatus = (typeof CLIENT_UPDATABLE_STATUSES)[number];

type ActionButtonProps = {
  icon: string;
  onClick: () => void;
  disabled?: boolean;
  bgColor: string;
  title?: string;
  isActive?: boolean;
  isLoading?: boolean;
  delay?: string;
};

function ActionButton({
  icon,
  onClick,
  disabled = false,
  bgColor,
  title,
  isActive = false,
  isLoading = false,
  delay = "0ms",
}: ActionButtonProps) {
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    onClick();
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      onMouseDown={(e) => {
        e.stopPropagation();
        e.preventDefault();
      }}
      onPointerDown={(e) => {
        e.stopPropagation();
        e.preventDefault();
      }}
      disabled={disabled}
      className={`
        flex items-center justify-center w-[45px] h-[30px] rounded-[16px] backdrop-blur-sm
        transition-all duration-200 ease-out ${bgColor}
        ${
          isActive
            ? "opacity-40 cursor-not-allowed bg-gray-300"
            : "opacity-75 hover:opacity-90 hover:drop-shadow-md hover:scale-105 backdrop-blur-sm"
        }
        ${isLoading ? "animate-pulse" : ""}
      `}
      title={title}
      style={{ transitionDelay: delay }}
    >
      <div className="w-4 h-4 flex items-center justify-center">
        <Icon
          icon={icon}
          width="14"
          height="14"
          className={`
            transition-all duration-200
            ${isActive ? "opacity-50" : isLoading ? "opacity-50" : "opacity-80"}
          `}
        />
      </div>
    </button>
  );
}

type ItemActionPanelProps = {
  itemId: string;
  currentStatus: ClientStatus;
  onStatusUpdate?: (newStatus: ClientStatus) => void;
  isExpanded: boolean;
  onExpandedChange?: (expanded: boolean) => void;
  onDelete?: (itemId: string, options?: { alreadyDeleted?: boolean }) => void;
};

const STATUS_ACTIONS: Record<
  ClientStatus,
  { bgColor: string; icon: string; label: string; action: ClientStatus }
> = {
  adding: {
    bgColor: "bg-slate-200",
    icon: "mingcute:time-fill",
    label: "Adding",
    action: "adding",
  },
  completed: {
    bgColor: "bg-green-400",
    icon: "mingcute:check-fill",
    label: "Completed",
    action: "completed",
  },
  queued: {
    bgColor: "bg-blue-400",
    icon: "f7:forward-fill",
    label: "Queued",
    action: "queued",
  },
  paused: {
    bgColor: "bg-gray-400",
    icon: "mingcute:pause-fill",
    label: "Paused",
    action: "paused",
  },
  bookmark: {
    bgColor: "bg-purple-400",
    icon: "mdi:bookmark",
    label: "Bookmarked",
    action: "bookmark",
  },
  error: {
    bgColor: "bg-red-400",
    icon: "mingcute:alert-fill",
    label: "Error",
    action: "error",
  },
};

export default function ItemActionPanel({
  itemId,
  currentStatus,
  onStatusUpdate,
  isExpanded,
  onExpandedChange,
  onDelete,
}: ItemActionPanelProps) {
  const router = useRouter();
  const pathname = usePathname();
  const [isUpdating, setIsUpdating] = useState<string | null>(null);
  const isProcessing = currentStatus === "adding";

  useEffect(() => {
    if (isProcessing && isExpanded) {
      onExpandedChange?.(false);
    }
  }, [isExpanded, isProcessing, onExpandedChange]);

  const handleAction = async (action: ClientStatus | "delete") => {
    if (isUpdating || isProcessing) return;

    setIsUpdating(action);

    try {
      if (action === "delete") {
        // Revalidate the current route along with queue/search on the server
        const targetPath = pathname || "/queue";
        const result = await deleteItemAction(itemId, targetPath);
        if (result?.success) {
          onDelete?.(itemId, { alreadyDeleted: true });
          router.refresh();
        }
      } else if (
        CLIENT_UPDATABLE_STATUSES.includes(action as ClientUpdatableStatus)
      ) {
        onStatusUpdate?.(action);
        const result = await updateItemStatusInQueue(
          itemId,
          action as ClientUpdatableStatus,
        );
        if (!result.success) {
          console.error("Failed to update item status:", result.error);
        }
      }
    } catch (error) {
      console.error(`Error handling action \"${action}\":`, error);
    } finally {
      setIsUpdating(null);
      onExpandedChange?.(false);
    }
  };

  const currentButton = STATUS_ACTIONS[currentStatus];

  const queuePauseAction =
    currentStatus === "queued" ? STATUS_ACTIONS.paused : STATUS_ACTIONS.queued;

  const collapsePanel = () => {
    onExpandedChange?.(false);
  };

  const wrapperClassName = `relative ${isExpanded ? "z-40" : "z-0"}`;

  return (
    <div
      className={wrapperClassName}
      onClick={(e) => {
        e.stopPropagation();
        e.preventDefault();
      }}
      onMouseDown={(e) => {
        e.stopPropagation();
        e.preventDefault();
      }}
      onPointerDown={(e) => {
        e.stopPropagation();
        e.preventDefault();
      }}
      onMouseLeave={collapsePanel}
      onPointerLeave={collapsePanel}
    >
      {/* Current Status Button - always visible */}
      <div
        className={`
          relative flex items-center justify-center h-[20px] rounded-[20px]
          transition-all duration-300 ease-out
          ${currentButton.bgColor}
          ${
            isExpanded
              ? "opacity-0 pointer-events-none z-0"
              : "opacity-100 z-10"
          }
        `}
        onMouseEnter={() => {
          if (!isProcessing) {
            onExpandedChange?.(true);
          }
        }}
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();
        }}
        onMouseDown={(e) => {
          e.stopPropagation();
          e.preventDefault();
        }}
        title={`${currentButton.label} item`}
      >
        <div className="h-[15px] flex items-center justify-center gap-[5px] p-[10px]">
          <div className="text-[10px] font-mono font-semibold">
            {currentButton.label}
          </div>
          <Icon
            icon={currentButton.icon}
            width="15"
            height="15"
            className="opacity-100"
          />
        </div>
      </div>

      {/* Expanded Panel - 2x2 Grid */}
      <div
        className={`
          absolute top-0 right-0 z-50
          transition-all duration-300 ease-out origin-top-right
          ${
            isExpanded
              ? "opacity-100 scale-100 pointer-events-auto"
              : "opacity-0 scale-90 pointer-events-none"
          }
          ${isProcessing ? "pointer-events-none" : ""}
        `}
        onMouseLeave={() => onExpandedChange?.(false)}
        onMouseEnter={() => {
          if (!isProcessing) {
            onExpandedChange?.(true);
          }
        }}
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();
        }}
        onMouseDown={(e) => {
          e.stopPropagation();
          e.preventDefault();
        }}
        onPointerDown={(e) => {
          e.stopPropagation();
          e.preventDefault();
        }}
      >
        <div
          className={`
            panel-base panel-light grid gap-[5px] p-[5px] drop-shadow-lg w-fit
            [grid-template-columns:repeat(2,45px)]
            [grid-auto-rows:30px]
            translate-x-[5px] translate-y-[-5px]
          `}
          onClick={(e) => {
            e.stopPropagation();
            e.preventDefault();
          }}
          onMouseDown={(e) => {
            e.stopPropagation();
            e.preventDefault();
          }}
        >
          {/* Top Left: Bookmark */}
          <div className="col-start-1 row-start-1">
            <ActionButton
              icon="mdi:bookmark"
              onClick={() => handleAction("bookmark")}
              disabled={
                isProcessing ||
                currentStatus === "bookmark" ||
                isUpdating === "bookmark"
              }
              bgColor="bg-purple-400"
              title="Mark as bookmark"
              isActive={currentStatus === "bookmark"}
              isLoading={isUpdating === "bookmark"}
              delay={isExpanded ? "50ms" : "0ms"}
            />
          </div>

          {/* Top Right: Complete */}
          <div className="col-start-2 row-start-1">
            <ActionButton
              icon="mingcute:check-fill"
              onClick={() => handleAction("completed")}
              disabled={
                isProcessing ||
                currentStatus === "completed" ||
                isUpdating === "completed"
              }
              bgColor="bg-green-400"
              title="Complete item"
              isActive={currentStatus === "completed"}
              isLoading={isUpdating === "completed"}
              delay={isExpanded ? "0ms" : "0ms"}
            />
          </div>

          {/* Bottom Left: Delete */}
          <div className="col-start-1 row-start-2">
            <ActionButton
              icon="mingcute:close-fill"
              onClick={() => handleAction("delete")}
              disabled={isProcessing || isUpdating === "delete"}
              bgColor="bg-red-400"
              title="Delete item"
              isLoading={isUpdating === "delete"}
              delay={isExpanded ? "150ms" : "0ms"}
            />
          </div>

          {/* Bottom Right: Queue/Pause */}
          <div className="col-start-2 row-start-2">
            <ActionButton
              icon={queuePauseAction.icon}
              onClick={() => handleAction(queuePauseAction.action)}
              disabled={
                isProcessing ||
                currentStatus === queuePauseAction.action ||
                isUpdating === queuePauseAction.action
              }
              bgColor={queuePauseAction.bgColor}
              title={queuePauseAction.label}
              isActive={currentStatus === queuePauseAction.action}
              isLoading={isUpdating === queuePauseAction.action}
              delay={isExpanded ? "100ms" : "0ms"}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
