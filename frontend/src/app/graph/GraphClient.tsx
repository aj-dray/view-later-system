"use client";

import { Icon } from "@iconify/react";
import Image from "next/image";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";

import type { ClientStatus, ItemSummary } from "@/app/_lib/items-types";
import { CLIENT_UPDATABLE_STATUSES } from "@/app/_lib/items-types";
import { applyQueueFilter, type QueueFilter } from "@/app/_lib/queue";
import { updateItemStatusInQueue } from "@/app/queue/actions";
import ItemActionPanel from "@/app/_components/ItemActionPanel";
import {
  formatSavedAt,
  estimateReadingTime,
  estimateReadMinutes,
} from "@/app/_lib/item-utils";

type ClientUpdatableStatus = (typeof CLIENT_UPDATABLE_STATUSES)[number];

import {
  colorForClusterId,
  DEFAULT_CLUSTER_COLOR,
  type RGB,
  rgbToHex,
} from "./clusterColors";
import {
  CLUSTER_EVENT_NAME,
  CLUSTER_STATE_EVENT_NAME,
  type ClusterLegendEventDetail,
  type ClusterLegendItem,
  type ClusterStateEventDetail,
  type ClusteringState,
} from "./clusterEvents";

type VisualisationMode = "pca" | "tsne" | "umap";
type ClusteringMode = "kmeans" | "hca" | "dbscan";

type GraphClientProps = {
  items: ItemSummary[];
  filter: QueueFilter;
  visualisation: VisualisationMode;
  clustering: ClusteringMode;
  clusterKwarg: number | null;
};

type GraphNode = {
  id: string;
  item: ItemSummary;
  position: [number, number];
  targetPosition: [number, number] | null;
  status: "loading" | "settling" | "settled";
  pulsePhase: number;
  // Smooth loading trajectory parameters (for nodes without target positions)
  pathPhase: number; // evolving phase for parametric motion
  pathSpeed: number; // radians per frame
  pathAmplitude: [number, number]; // x/y amplitudes
  pathFrequency: [number, number]; // x/y frequencies (close, not identical)
  pathCenter: [number, number]; // anchor for random motion
  pathEasing: number; // 0..1 ramp for easing into motion
  readMinutes: number;
  clusterId: number | null;
  targetClusterId: number | null;
  color: RGB;
  targetColor: RGB;
  dimAmount: number; // 0 = full color, 1 = fully dimmed (for smooth transitions)
};

type CanvasSize = {
  width: number;
  height: number;
};

type CanvasViewState = {
  center: [number, number];
  scale: number;
};

type HoverState = {
  node: GraphNode;
  nodePosition: [number, number]; // Graph coordinates of the node
  currentStatus: ClientStatus;
  isStatusExpanded: boolean;
  isPinned: boolean; // Whether the card is pinned (clicked)
};

const DEFAULT_VIEW_STATE: CanvasViewState = {
  center: [0, 0],
  scale: 180,
};

const MIN_VIEW_SCALE = 30;
const MAX_VIEW_SCALE = 2000;
const GRAPH_PADDING = 0.15;
const DRIFT_RANGE = 0.8;

// Layout helpers: account for right-side panels when centering and distributing
function getPanelsWidthPx(): number {
  if (typeof window === "undefined") {
    return 0;
  }
  const root = document.documentElement;
  const raw = getComputedStyle(root).getPropertyValue("--panels-width");
  const parsed = Number.parseFloat(raw || "0");
  return Number.isFinite(parsed) ? parsed : 0;
}

function getReservedRightPx(): number {
  // Panel sits at right with a 25px gap similar to queue page padding
  return getPanelsWidthPx() + 25;
}

const VISUALISATION_LABELS: Record<VisualisationMode, string> = {
  umap: "UMAP",
  tsne: "t-SNE",
  pca: "PCA",
};

const CLUSTERING_LABELS: Record<ClusteringMode, string> = {
  kmeans: "k-means",
  hca: "HCA",
  dbscan: "DBSCAN",
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function randomBetween(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

function cloneColor(color: RGB): RGB {
  return [color[0], color[1], color[2]];
}

function createRandomPosition(): [number, number] {
  // Sample uniformly within a circle, slightly biased away from right-side panel
  const maxR = 1.5;
  const u = Math.random();
  const r = Math.sqrt(u) * maxR;
  const theta = randomBetween(0, Math.PI * 2);

  let x = r * Math.cos(theta);
  let y = r * Math.sin(theta);

  // Bias initial random X away from the right-side panel area
  let viewportWidth = 0;
  if (typeof window !== "undefined") {
    viewportWidth = window.innerWidth || 0;
  }
  const reserved = getReservedRightPx();
  const fraction =
    viewportWidth > 0 ? clamp(reserved / viewportWidth, 0, 0.6) : 0;
  const bias = fraction * 0.8; // shift left proportionally
  x -= bias;

  // If outside circle after bias, gently pull back to boundary
  const dist = Math.hypot(x, y);
  if (dist > maxR) {
    x = (x / dist) * maxR;
    y = (y / dist) * maxR;
  }

  return [x, y];
}

function createNode(item: ItemSummary): GraphNode {
  return {
    id: item.id,
    item,
    position: createRandomPosition(),
    targetPosition: null,
    status: "loading",
    pulsePhase: randomBetween(0, Math.PI * 2),
    // Initialize smooth trajectory - very slow, uniform motion with slight phase variation
    pathPhase: randomBetween(0, Math.PI * 2),
    pathSpeed: randomBetween(0.002, 0.004), // Much slower, reduced from 0.004-0.008
    pathAmplitude: [randomBetween(0.05, 0.12), randomBetween(0.05, 0.12)], // Slightly wider motion
    pathFrequency: [randomBetween(0.95, 1.05), randomBetween(0.95, 1.05)], // Very similar frequencies
    pathCenter: [0, 0],
    pathEasing: 0,
    readMinutes: estimateReadMinutes(item.content_token_count),
    clusterId: null,
    targetClusterId: null,
    color: cloneColor(DEFAULT_CLUSTER_COLOR),
    targetColor: cloneColor(DEFAULT_CLUSTER_COLOR),
    dimAmount: 0,
  };
}

function computeRadius(minutes: number): number {
  const minRadius = 20;
  const maxRadius = 40;
  const minTime = 5;

  // Formula: radius increases by 5px for every doubling of time from 5m base
  const safeMinutes = Math.max(minutes, minTime);
  const radius = minRadius + 5 * Math.log2(safeMinutes / minTime);

  return clamp(radius, minRadius, maxRadius);
}

function normalisePositions(
  coordinates: Map<string, [number, number]>,
): Map<string, [number, number]> {
  if (coordinates.size === 0) {
    return coordinates;
  }
  let sumX = 0;
  let sumY = 0;
  let count = 0;
  coordinates.forEach(([x, y]) => {
    if (Number.isFinite(x) && Number.isFinite(y)) {
      sumX += x;
      sumY += y;
      count += 1;
    }
  });

  if (count === 0) {
    return coordinates;
  }

  const meanX = sumX / count;
  const meanY = sumY / count;

  let varianceX = 0;
  let varianceY = 0;
  coordinates.forEach(([x, y]) => {
    if (Number.isFinite(x)) {
      const centeredX = x - meanX;
      varianceX += centeredX * centeredX;
    }
    if (Number.isFinite(y)) {
      const centeredY = y - meanY;
      varianceY += centeredY * centeredY;
    }
  });

  const stdX = Math.sqrt(varianceX / count) || 1;
  const stdY = Math.sqrt(varianceY / count) || 1;

  const normalised = new Map<string, [number, number]>();
  coordinates.forEach(([x, y], id) => {
    const safeX = Number.isFinite(x) ? (x - meanX) / stdX : 0;
    const safeY = Number.isFinite(y) ? (y - meanY) / stdY : 0;
    normalised.set(id, [safeX, safeY]);
  });

  return normalised;
}

function extractPositions(
  data: unknown,
  fallbackOrder: readonly string[],
): Map<string, [number, number]> {
  if (!data || typeof data !== "object") {
    return new Map();
  }

  const raw = data as Record<string, unknown>;
  const embeddings = Array.isArray(raw.reduced_embeddings)
    ? raw.reduced_embeddings
    : [];
  const itemIds = Array.isArray(raw.item_ids) ? (raw.item_ids as string[]) : [];

  const mapping = new Map<string, [number, number]>();

  if (itemIds.length === embeddings.length && itemIds.length > 0) {
    itemIds.forEach((id, index) => {
      const pair = embeddings[index];
      if (Array.isArray(pair) && pair.length >= 2) {
        const [x, y] = pair;
        if (Number.isFinite(Number(x)) && Number.isFinite(Number(y))) {
          mapping.set(id, [Number(x), Number(y)]);
        }
      }
    });
    return mapping;
  }

  if (embeddings.length === fallbackOrder.length) {
    fallbackOrder.forEach((id, index) => {
      const pair = embeddings[index];
      if (Array.isArray(pair) && pair.length >= 2) {
        const [x, y] = pair;
        if (Number.isFinite(Number(x)) && Number.isFinite(Number(y))) {
          mapping.set(id, [Number(x), Number(y)]);
        }
      }
    });
  }

  return mapping;
}

function centerViewOnPoints(
  previous: CanvasViewState,
  points: readonly [number, number][],
  canvasSize: CanvasSize,
): CanvasViewState {
  if (
    points.length === 0 ||
    canvasSize.width === 0 ||
    canvasSize.height === 0
  ) {
    return previous;
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  points.forEach(([x, y]) => {
    if (Number.isFinite(x)) {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
    }
    if (Number.isFinite(y)) {
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  });

  if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
    return previous;
  }

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const spanX = maxX - minX || 0.1;
  const spanY = maxY - minY || 0.1;
  const maxSpan = Math.max(spanX, spanY, 0.1);

  const reservedRight = getReservedRightPx();
  const usableWidth = Math.max(
    (canvasSize.width - reservedRight) * (1 - GRAPH_PADDING),
    1,
  );
  const usableHeight = canvasSize.height * (1 - GRAPH_PADDING);
  const scaleFactor = Math.min(usableWidth, usableHeight) / maxSpan;
  const scale = clamp(scaleFactor, MIN_VIEW_SCALE, MAX_VIEW_SCALE);

  // Shift the logical center left so content visually centers in the visible frame
  const screenShift = reservedRight / 2; // center of visible area
  const graphShiftX = screenShift / scale;

  return {
    center: [centerX + graphShiftX, centerY],
    scale,
  };
}

function buildClusterLegend(
  items: readonly ItemSummary[],
  assignments: Map<string, number | null>,
  labels: Map<number, string>,
  isLoadingLabels: boolean,
): ClusterLegendItem[] {
  const counts = new Map<number | null, number>();

  items.forEach((item) => {
    const assigned = assignments.get(item.id);
    const normalised =
      assigned != null && Number.isFinite(assigned) && assigned >= 0
        ? Math.trunc(assigned)
        : null;
    const current = counts.get(normalised) ?? 0;
    counts.set(normalised, current + 1);
  });

  const entries = Array.from(counts.entries()).sort((a, b) => {
    const [clusterA] = a;
    const [clusterB] = b;
    if (clusterA == null && clusterB == null) {
      return 0;
    }
    if (clusterA == null) {
      return 1;
    }
    if (clusterB == null) {
      return -1;
    }
    return clusterA - clusterB;
  });

  return entries.map(([clusterId, count]) => {
    let label: string;
    let isLoading = false;

    if (clusterId == null) {
      label = "N/A";
    } else {
      const actualLabel = labels.get(clusterId);
      if (actualLabel) {
        label = actualLabel;
      } else {
        label = `Cluster ${clusterId + 1}`;
        isLoading = isLoadingLabels;
      }
    }

    const color = rgbToHex(colorForClusterId(clusterId));
    return {
      clusterId,
      label,
      color,
      count,
      isLoading,
    };
  });
}

export default function GraphClient({
  items,
  filter,
  visualisation,
  clustering,
  clusterKwarg,
}: GraphClientProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const isPanningRef = useRef(false);
  const lastPointerPositionRef = useRef<[number, number] | null>(null);
  const autoCenteredRef = useRef(false);

  const [canvasSize, setCanvasSize] = useState<CanvasSize>({
    width: 0,
    height: 0,
  });
  const [viewState, setViewState] =
    useState<CanvasViewState>(DEFAULT_VIEW_STATE);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [hovered, setHovered] = useState<HoverState | null>(null);

  // Remove a node immediately in the UI after deletion
  const removeNode = useCallback(
    (itemId: string, _options?: { alreadyDeleted?: boolean }) => {
      setNodes((prev) => prev.filter((node) => node.id !== itemId));
      setHovered((prev) => (prev?.node.id === itemId ? null : prev));
    },
    [],
  );
  const [hoverFaviconError, setHoverFaviconError] = useState(false);
  const [isLoadingReduction, setIsLoadingReduction] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isLoadingClusters, setIsLoadingClusters] = useState(false);
  const [clusterError, setClusterError] = useState<string | null>(null);
  const [isLoadingLabels, setIsLoadingLabels] = useState(false);
  const [clusterLabels, setClusterLabels] = useState<Map<number, string>>(
    new Map(),
  );
  const [hasReductionResult, setHasReductionResult] = useState(false);
  const [hasClusterLabels, setHasClusterLabels] = useState(false);

  const viewStateRef = useRef(viewState);
  const nodesRef = useRef(nodes);
  const hoveredRef = useRef(hovered);
  const lastLabelFetchKeyRef = useRef<string>("");
  const lastReductionFetchKeyRef = useRef<string>("");
  const lastClusterFetchKeyRef = useRef<string>("");
  const clusterAssignmentsRef = useRef<Map<string, number | null>>(new Map());
  const hasClusterLabelsRef = useRef(hasClusterLabels);

  useEffect(() => {
    viewStateRef.current = viewState;
  }, [viewState]);

  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  useEffect(() => {
    hoveredRef.current = hovered;
  }, [hovered]);

  useEffect(() => {
    hasClusterLabelsRef.current = hasClusterLabels;
  }, [hasClusterLabels]);

  const emitClusterLegend = useCallback(
    (items: ClusterLegendItem[]) => {
      if (typeof window === "undefined") {
        return;
      }

      const detail: ClusterLegendEventDetail = {
        filter,
        clustering,
        clusterKwarg,
        items,
      };

      window.dispatchEvent(
        new CustomEvent<ClusterLegendEventDetail>(CLUSTER_EVENT_NAME, {
          detail,
        }),
      );
    },
    [filter, clustering, clusterKwarg],
  );

  const emitClusteringState = useCallback((state: ClusteringState) => {
    if (typeof window === "undefined") {
      return;
    }

    const detail: ClusterStateEventDetail = { state };
    window.dispatchEvent(
      new CustomEvent<ClusterStateEventDetail>(CLUSTER_STATE_EVENT_NAME, {
        detail,
      }),
    );
  }, []);

  useEffect(() => {
    return () => {
      emitClusterLegend([]);
    };
  }, [emitClusterLegend]);

  useEffect(() => {
    return () => {
      lastReductionFetchKeyRef.current = "";
      lastClusterFetchKeyRef.current = "";
      lastLabelFetchKeyRef.current = "";
      setHasReductionResult(false);
      setHasClusterLabels(false);
    };
  }, []);

  const filteredItems = useMemo(
    () => applyQueueFilter(items, filter),
    [items, filter],
  );

  const filteredItemsRef = useRef(filteredItems);
  useEffect(() => {
    filteredItemsRef.current = filteredItems;
  }, [filteredItems]);
  const filteredItemKey = useMemo(
    () => filteredItems.map((item) => item.id).join("|"),
    [filteredItems],
  );

  useEffect(() => {
    setHasReductionResult(false);
    setHasClusterLabels(false);
    lastReductionFetchKeyRef.current = "";
    lastClusterFetchKeyRef.current = "";
    lastLabelFetchKeyRef.current = "";
  }, [filteredItemKey]);

  useEffect(() => {
    setHasReductionResult(false);
    lastReductionFetchKeyRef.current = "";
    if (!hasClusterLabelsRef.current) {
      lastClusterFetchKeyRef.current = "";
    }
  }, [visualisation]);

  useEffect(() => {
    setHasClusterLabels(false);
    lastClusterFetchKeyRef.current = "";
    lastLabelFetchKeyRef.current = "";
  }, [clustering, clusterKwarg]);

  useEffect(() => {
    autoCenteredRef.current = false;
  }, [visualisation, filter, clustering]);

  useEffect(() => {
    const resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      const { width, height } = entry.contentRect;
      setCanvasSize({ width, height });
    });

    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const { width, height } = canvasSize;
    if (width === 0 || height === 0) {
      return;
    }

    const dpr = window.devicePixelRatio ?? 1;
    canvas.width = Math.max(Math.floor(width * dpr), 1);
    canvas.height = Math.max(Math.floor(height * dpr), 1);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }, [canvasSize]);

  useEffect(() => {
    setHovered(null);
    setNodes((prev) => {
      const currentById = new Map(prev.map((node) => [node.id, node]));

      const nextNodes: GraphNode[] = filteredItems.map((item) => {
        const existing = currentById.get(item.id);

        if (existing) {
          // Keep existing node, just update item data
          return {
            ...existing,
            item,
            readMinutes: estimateReadMinutes(item.content_token_count),
            // Ensure pathCenter defaults to current position if missing
            pathCenter: existing.pathCenter ?? existing.position,
          };
        }

        // Create new node with random position
        const created = createNode(item);
        created.pathCenter = created.position;
        created.pathEasing = 0;
        return created;
      });
      return nextNodes;
    });
  }, [filteredItems]);

  // Reset favicon error state when hovered item changes or its favicon changes
  useEffect(() => {
    if (!hovered) {
      setHoverFaviconError(false);
      return;
    }
    setHoverFaviconError(false);
  }, [hovered?.node.item.id, hovered?.node.item.favicon_url]);

  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();

    const performFetch = async () => {
      const activeItems = filteredItemsRef.current;

      if (!activeItems.length) {
        lastReductionFetchKeyRef.current = "";
        if (isMounted) {
          setIsLoadingReduction(false);
          setLoadError(null);
          setNodes([]);
        }
        return;
      }

      const reductionKey = `${visualisation}|${filteredItemKey}`;
      if (reductionKey === lastReductionFetchKeyRef.current) {
        return;
      }
      lastReductionFetchKeyRef.current = reductionKey;

      // Immediately switch to loading motion while awaiting new positions
      setNodes((prev) =>
        prev.map((node) => ({
          ...node,
          targetPosition: null,
          status: "loading",
          // Anchor random motion around current position
          pathCenter: node.position,
          pathEasing: 0,
          // Slightly randomize path parameters to avoid lockstep
          pathSpeed: node.pathSpeed * randomBetween(0.9, 1.1),
          pathPhase: node.pathPhase + randomBetween(-0.5, 0.5),
        })),
      );

      setIsLoadingReduction(true);
      setLoadError(null);

      const params = new URLSearchParams();
      activeItems.forEach((item) => {
        params.append("item_ids", item.id);
      });
      params.set("mode", visualisation);

      try {
        const response = await fetch(
          `/api/graph/dimensional-reduction?${params.toString()}`,
          { signal: controller.signal },
        );

        if (!response.ok) {
          const message = await response.text();
          throw new Error(
            message ||
              `Failed to load dimensional reduction (status ${response.status})`,
          );
        }

        const payload = await response.json();
        if (!isMounted) {
          return;
        }

        const currentIds = activeItems.map((i) => i.id);
        const fetchedPositions = normalisePositions(
          extractPositions(payload, currentIds),
        );

        setNodes((prev) => {
          const previousById = new Map(prev.map((node) => [node.id, node]));

          return activeItems.map((item) => {
            const existing = previousById.get(item.id);
            const target = fetchedPositions.get(item.id) ?? null;

            let base: GraphNode;
            if (existing) {
              // Keep existing node (maintains animation state and cluster info)
              base = existing;
            } else {
              // New node - random position
              base = createNode(item);
              base.pathCenter = base.position;
            }

            return {
              ...base,
              item,
              readMinutes: estimateReadMinutes(item.content_token_count),
              targetPosition: target,
              status: target ? ("settling" as const) : ("loading" as const),
            };
          });
        });

        if (
          fetchedPositions.size > 0 &&
          !autoCenteredRef.current &&
          canvasSize.width > 0 &&
          canvasSize.height > 0
        ) {
          const coords = Array.from(fetchedPositions.values());
          autoCenteredRef.current = true;
          setViewState((prev) => centerViewOnPoints(prev, coords, canvasSize));
        }

        setHasReductionResult(true);
        setIsLoadingReduction(false);
      } catch (error) {
        if (!isMounted || controller.signal.aborted) {
          return;
        }
        console.error("Failed to fetch dimensional reduction", error);
        lastReductionFetchKeyRef.current = "";
        setLoadError(
          error instanceof Error ? error.message : "Unable to position items",
        );
        setHasReductionResult(false);
        setIsLoadingReduction(false);
        setNodes((prev) =>
          prev.map((node) => ({
            ...node,
            targetPosition: null,
            status: "loading",
            pathCenter: node.position,
          })),
        );
      }
    };

    performFetch();

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [filteredItemKey, visualisation]);

  useEffect(() => {
    const currentItems = filteredItemsRef.current;

    if (!currentItems.length) {
      clusterAssignmentsRef.current = new Map();
      setIsLoadingClusters(false);
      setClusterError(null);
      setHasClusterLabels(false);
      setNodes((prev) =>
        prev.map((node) => ({
          ...node,
          targetClusterId: null,
          targetColor: cloneColor(DEFAULT_CLUSTER_COLOR),
        })),
      );
      emitClusterLegend([]);
      emitClusteringState("idle");
      lastClusterFetchKeyRef.current = "";
      return;
    }

    if (!hasReductionResult) {
      return;
    }

    let isMounted = true;
    const controller = new AbortController();

    const performFetch = async () => {
      const activeItems = filteredItemsRef.current;

      const clusterKeyParts = [
        clustering,
        Number.isFinite(clusterKwarg) ? String(clusterKwarg) : "null",
        filteredItemKey,
      ];
      const clusterKey = clusterKeyParts.join("|");
      if (clusterKey === lastClusterFetchKeyRef.current) {
        return;
      }
      lastClusterFetchKeyRef.current = clusterKey;

      setHasClusterLabels(false);
      setIsLoadingClusters(true);
      setClusterError(null);
      emitClusteringState("clustering");

      const params = new URLSearchParams();
      activeItems.forEach((item) => {
        params.append("item_ids", item.id);
      });
      params.set("mode", clustering);

      if (clusterKwarg != null && Number.isFinite(clusterKwarg)) {
        if (clustering === "kmeans") {
          params.set("k", `${clusterKwarg}`);
        } else if (clustering === "hca") {
          params.set("k", `${clusterKwarg}`);
        } else if (clustering === "dbscan") {
          // clusterKwarg for DBSCAN represents eps if provided
          params.set("eps", `${clusterKwarg}`);
        }
      }

      try {
        const response = await fetch(
          `/api/graph/clustering?${params.toString()}`,
          { signal: controller.signal },
        );

        if (!response.ok) {
          const message = await response.text();
          throw new Error(
            message || `Failed to load clustering (status ${response.status})`,
          );
        }

        const payload = await response.json();
        if (!isMounted) {
          return;
        }

        const assignments = new Map<string, number | null>();
        const rawClusters = Array.isArray(payload?.clusters)
          ? payload.clusters
          : [];
        const rawIds = Array.isArray(payload?.item_ids) ? payload.item_ids : [];

        rawIds.forEach((id: unknown, index: number) => {
          const clusterValue = rawClusters[index];
          const numeric =
            typeof clusterValue === "number"
              ? clusterValue
              : typeof clusterValue === "string"
                ? Number.parseInt(clusterValue, 10)
                : null;

          const normalised =
            numeric != null && Number.isFinite(numeric) && numeric >= 0
              ? Math.trunc(numeric)
              : null;

          if (typeof id === "string") {
            assignments.set(id, normalised);
          } else if (id != null) {
            assignments.set(String(id), normalised);
          }
        });

        clusterAssignmentsRef.current = assignments;

        setNodes((prev) =>
          prev.map((node) => {
            const assignment = assignments.get(node.id) ?? null;
            const targetColor = cloneColor(colorForClusterId(assignment));
            return {
              ...node,
              targetClusterId: assignment,
              targetColor,
            };
          }),
        );

        const legendItems = buildClusterLegend(
          activeItems,
          assignments,
          new Map(),
          true,
        );

        console.log(
          "[Clustering Complete] Emitting legend items:",
          legendItems,
        );
        emitClusterLegend(legendItems);
        setIsLoadingClusters(false);
        emitClusteringState("labeling");
      } catch (error) {
        if (!isMounted || controller.signal.aborted) {
          return;
        }
        console.error("Failed to fetch clustering", error);
        lastClusterFetchKeyRef.current = "";
        clusterAssignmentsRef.current = new Map();
        setClusterError(
          error instanceof Error ? error.message : "Unable to cluster items",
        );
        setHasClusterLabels(false);
        setIsLoadingClusters(false);
        setNodes((prev) =>
          prev.map((node) => ({
            ...node,
            targetClusterId: null,
            targetColor: cloneColor(DEFAULT_CLUSTER_COLOR),
          })),
        );
        emitClusterLegend([]);
        emitClusteringState("idle");
      }
    };

    performFetch();

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [
    filteredItemKey,
    clustering,
    clusterKwarg,
    emitClusterLegend,
    emitClusteringState,
    hasReductionResult,
  ]);

  useEffect(() => {
    console.log("[Label Fetch] Effect triggered", {
      isLoadingClusters,
      filteredItemsCount: filteredItems.length,
    });

    if (isLoadingClusters || !filteredItems.length) {
      console.log("[Label Fetch] Skipping: isLoadingClusters or no items");
      setClusterLabels(new Map());
      lastLabelFetchKeyRef.current = "";
      setHasClusterLabels(false);
      return;
    }

    const assignments = clusterAssignmentsRef.current;
    const clusterIds = new Set<number>();
    assignments.forEach((clusterId) => {
      if (clusterId != null && clusterId >= 0) {
        clusterIds.add(clusterId);
      }
    });

    console.log("[Label Fetch] Cluster IDs found:", Array.from(clusterIds));
    console.log("[Label Fetch] Total assignments:", assignments.size);

    if (clusterIds.size === 0) {
      console.log("[Label Fetch] Skipping: no clusters found");
      setClusterLabels(new Map());
      lastLabelFetchKeyRef.current = "";
      setHasClusterLabels(false);
      return;
    }

    const clusterKey = Array.from(assignments.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([id, cluster]) => `${id}:${cluster}`)
      .join("|");

    console.log("[Label Fetch] Cluster key:", clusterKey.substring(0, 100));
    console.log(
      "[Label Fetch] Last key:",
      lastLabelFetchKeyRef.current.substring(0, 100),
    );

    if (clusterKey === lastLabelFetchKeyRef.current) {
      console.log("[Label Fetch] Skipping: same cluster key");
      return;
    }

    lastLabelFetchKeyRef.current = clusterKey;

    let isMounted = true;
    const controller = new AbortController();

    const fetchLabels = async () => {
      console.log("[Label Fetch] Starting fetch...");
      setIsLoadingLabels(true);
      setHasClusterLabels(false);

      const params = new URLSearchParams();
      filteredItems.forEach((item) => {
        params.append("item_ids", item.id);
      });

      const clusterArray = filteredItems.map(
        (item) => assignments.get(item.id) ?? -1,
      );
      params.set("clusters", JSON.stringify(clusterArray));

      const url = `/api/graph/labels?${params.toString()}`;
      console.log("[Label Fetch] URL:", url.substring(0, 200));

      try {
        const response = await fetch(url, {
          signal: controller.signal,
        });

        console.log("[Label Fetch] Response status:", response.status);

        if (!response.ok) {
          throw new Error(`Failed to fetch labels (status ${response.status})`);
        }

        const payload = await response.json();
        console.log("[Label Fetch] Payload received:", payload);

        if (!isMounted) {
          console.log("[Label Fetch] Component unmounted, skipping update");
          return;
        }

        const labelsMap = new Map<number, string>();
        if (payload.labels && typeof payload.labels === "object") {
          Object.entries(payload.labels).forEach(([key, value]) => {
            const clusterId = Number(key);
            if (Number.isFinite(clusterId) && typeof value === "string") {
              labelsMap.set(clusterId, value);
            }
          });
        }

        console.log(
          "[Label Fetch] Labels parsed:",
          Object.fromEntries(labelsMap),
        );
        setClusterLabels(labelsMap);
        setIsLoadingLabels(false);
        setHasClusterLabels(true);

        const legendItems = buildClusterLegend(
          filteredItems,
          assignments,
          labelsMap,
          false,
        );
        console.log("[Label Fetch] Emitting legend with labels:", legendItems);
        emitClusterLegend(legendItems);
        emitClusteringState("complete");
      } catch (error) {
        if (!isMounted || controller.signal.aborted) {
          console.log("[Label Fetch] Fetch cancelled or unmounted");
          return;
        }
        console.error("[Label Fetch] Error:", error);
        setIsLoadingLabels(false);
        setHasClusterLabels(false);
      }
    };

    fetchLabels();

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [
    isLoadingClusters,
    filteredItems,
    emitClusterLegend,
    emitClusteringState,
  ]);

  useEffect(() => {
    if (autoCenteredRef.current) {
      return;
    }
    if (!nodesRef.current.length) {
      return;
    }
    if (canvasSize.width === 0 || canvasSize.height === 0) {
      return;
    }
    const coords = nodesRef.current
      .map((node) => node.targetPosition ?? null)
      .filter(
        (value): value is [number, number] =>
          Array.isArray(value) && value.length >= 2,
      );
    if (!coords.length) {
      return;
    }
    autoCenteredRef.current = true;
    setViewState((prev) => centerViewOnPoints(prev, coords, canvasSize));
  }, [canvasSize]);

  useEffect(() => {
    if (!nodes.length) {
      return;
    }

    let animationFrame: number;

    const step = () => {
      const currentHovered = hoveredRef.current;
      const hoveredIdCurrent = currentHovered?.node.id ?? null;
      const isPinnedCurrent = currentHovered?.isPinned ?? false;

      setNodes((prev) => {
        let changed = false;
        const next = prev.map((node) => {
          const target = node.targetPosition;
          let [x, y] = node.position;
          let status = node.status;
          // Slower, smoother pulse progression
          const newPulse = node.pulsePhase + 0.035;
          const desiredColor = node.targetColor ?? node.color;
          const colorEasing = 0.12;
          let nextColor: RGB = [
            node.color[0] + (desiredColor[0] - node.color[0]) * colorEasing,
            node.color[1] + (desiredColor[1] - node.color[1]) * colorEasing,
            node.color[2] + (desiredColor[2] - node.color[2]) * colorEasing,
          ];
          let clusterId = node.clusterId;

          // Smooth dimming transition
          const targetDim =
            isPinnedCurrent && node.id !== hoveredIdCurrent ? 1 : 0;
          const dimEasing = 0.08;
          let nextDim =
            node.dimAmount + (targetDim - node.dimAmount) * dimEasing;

          // Clamp near zero/one to avoid infinite tiny updates
          if (Math.abs(nextDim - targetDim) < 0.001) {
            nextDim = targetDim;
          }

          if (target) {
            const dx = target[0] - x;
            const dy = target[1] - y;
            const distance = Math.hypot(dx, dy);

            if (distance > 0.002) {
              const easing = status === "settling" ? 0.16 : 0.12;
              x += dx * easing;
              y += dy * easing;
              status = "settling";
            } else {
              x = target[0];
              y = target[1];
              status = "settled";
            }
          } else {
            // Smooth, gentle anchored Lissajous-like paths for loading state
            const nextPhase = node.pathPhase + node.pathSpeed;
            const baseX = Math.sin(nextPhase * node.pathFrequency[0]);
            const baseY = Math.sin(
              nextPhase * node.pathFrequency[1] + Math.PI / 3,
            );
            const driftXFull = baseX * node.pathAmplitude[0];
            const driftYFull = baseY * node.pathAmplitude[1];

            // Ease drift from 0 to full to avoid initial jump
            const easingRate = 0.06;
            const nextEasing = Math.min(1, (node.pathEasing ?? 0) + easingRate);
            const driftX = driftXFull * nextEasing;
            const driftY = driftYFull * nextEasing;

            // Anchor around pathCenter
            const anchorX = node.pathCenter ? node.pathCenter[0] : 0;
            const anchorY = node.pathCenter ? node.pathCenter[1] : 0;

            x = anchorX + driftX;
            y = anchorY + driftY;

            // Keep within circular domain gently
            const maxR = 1.5;
            const dist = Math.hypot(x, y);
            if (dist > maxR) {
              const pull = 0.02;
              x = x - (x / dist) * (dist - maxR) * pull;
              y = y - (y / dist) * (dist - maxR) * pull;
            }

            status = "loading";
            // Persist updated phase and easing into node below via change detection
            (node as GraphNode).pathPhase = nextPhase;
            (node as GraphNode).pathEasing = nextEasing;
          }

          const colorClose =
            Math.abs(nextColor[0] - desiredColor[0]) < 0.5 &&
            Math.abs(nextColor[1] - desiredColor[1]) < 0.5 &&
            Math.abs(nextColor[2] - desiredColor[2]) < 0.5;

          if (colorClose) {
            nextColor = cloneColor(desiredColor);
            clusterId = node.targetClusterId ?? null;
          }

          if (
            x !== node.position[0] ||
            y !== node.position[1] ||
            status !== node.status ||
            newPulse !== node.pulsePhase ||
            // pathPhase may change every frame while loading
            (status === "loading" &&
              (node.pathPhase || 0) !== (node.pathPhase as number)) ||
            nextColor[0] !== node.color[0] ||
            nextColor[1] !== node.color[1] ||
            nextColor[2] !== node.color[2] ||
            clusterId !== node.clusterId ||
            nextDim !== node.dimAmount
          ) {
            changed = true;
            return {
              ...node,
              position: [x, y] as [number, number],
              status,
              pulsePhase: newPulse,
              pathPhase: (node as GraphNode).pathPhase,
              color: nextColor,
              clusterId,
              dimAmount: nextDim,
            };
          }
          return node;
        });
        return changed ? next : prev;
      });

      animationFrame = requestAnimationFrame(step);
    };

    animationFrame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animationFrame);
  }, [nodes.length]);

  const drawScene = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    const dpr = window.devicePixelRatio ?? 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);

    if (canvasSize.width === 0 || canvasSize.height === 0) {
      return;
    }

    const { center, scale } = viewStateRef.current;
    const hoveredId = hovered?.node.id ?? null;
    const isPinned = hovered?.isPinned ?? false;

    nodesRef.current.forEach((node) => {
      const [screenX, screenY] = graphToScreen(
        node.position,
        { center, scale },
        canvasSize,
      );

      const baseRadius = computeRadius(node.readMinutes);
      const pulse =
        node.status === "loading"
          ? 1 + 0.08 * Math.sin(node.pulsePhase)
          : node.status === "settling"
            ? 1 + 0.035 * Math.sin(node.pulsePhase)
            : 1;
      const radius = baseRadius * pulse;

      // Pulse in grey while loading (until clustered), then fade toward cluster color
      const isUnclusteredLoading =
        node.status !== "settled" && node.targetClusterId == null;
      const baseAlpha = node.status === "settled" ? 210 : 170;
      const pulseAlpha = Math.sin(node.pulsePhase) * 0.5 + 0.5;
      let alpha = clamp(
        baseAlpha +
          (node.status === "loading" ? pulseAlpha * 70 : pulseAlpha * 28),
        90,
        230,
      );

      // Apply smooth dimming based on dimAmount
      let [rC, gC, bC] = node.color;
      if (node.dimAmount > 0.01) {
        alpha *= 1 - node.dimAmount * 0.7; // Reduce opacity smoothly
        // Desaturate by blending with grey
        const grey = (rC + gC + bC) / 3;
        const desaturation = 0.7 * node.dimAmount;
        rC = rC * (1 - desaturation) + grey * desaturation;
        gC = gC * (1 - desaturation) + grey * desaturation;
        bC = bC * (1 - desaturation) + grey * desaturation;
      }

      const fillAlpha = alpha / 255;
      const grey = 200; // light grey tone
      const [r, g, b] = isUnclusteredLoading
        ? [grey, grey, grey]
        : [rC, gC, bC];

      ctx.beginPath();
      ctx.arc(screenX, screenY, radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${fillAlpha})`;
      ctx.fill();

      if (hoveredId === node.id) {
        ctx.strokeStyle = isPinned
          ? "rgba(40, 70, 120, 1)"
          : "rgba(40, 70, 120, 0.9)";
        ctx.lineWidth = isPinned ? 3 : 2;
        ctx.stroke();
      }
    });
  }, [canvasSize, hovered]);

  useEffect(() => {
    drawScene();
  }, [drawScene, nodes, viewState, hovered, canvasSize]);

  const handleZoom = useCallback((delta: number) => {
    setViewState((prev) => {
      const nextScale = clamp(
        prev.scale * delta,
        MIN_VIEW_SCALE,
        MAX_VIEW_SCALE,
      );
      return { ...prev, scale: nextScale };
    });
  }, []);

  const handleCenter = useCallback(() => {
    if (!nodesRef.current.length) {
      setViewState(DEFAULT_VIEW_STATE);
      return;
    }
    const coords = nodesRef.current.map(
      (node) => node.targetPosition ?? node.position,
    );
    setViewState((prev) => centerViewOnPoints(prev, coords, canvasSize));
  }, [canvasSize]);

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) {
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const cssX = event.clientX - rect.left;
      const cssY = event.clientY - rect.top;

      // Check if clicking on a hovered node to pin it
      if (hovered && !hovered.isPinned) {
        const [graphX, graphY] = screenToGraph(
          [cssX, cssY],
          viewStateRef.current,
          canvasSize,
        );

        const dx = hovered.node.position[0] - graphX;
        const dy = hovered.node.position[1] - graphY;
        const distance = Math.hypot(dx, dy);

        const radius = computeRadius(hovered.node.readMinutes);

        const screenDistance = distance * viewStateRef.current.scale;

        // If clicking within the node radius, pin the card
        if (screenDistance <= radius * 1.5) {
          event.preventDefault();
          event.stopPropagation();
          setHovered((prev) => (prev ? { ...prev, isPinned: true } : null));
          return;
        }
      }

      // If clicking outside when card is pinned, unpin it
      if (hovered?.isPinned) {
        setHovered(null);
        return;
      }

      isPanningRef.current = true;
      lastPointerPositionRef.current = [event.clientX, event.clientY];
      setHovered(null);
      canvas.setPointerCapture(event.pointerId);
    },
    [hovered, canvasSize],
  );

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) {
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const cssX = event.clientX - rect.left;
      const cssY = event.clientY - rect.top;

      if (isPanningRef.current && lastPointerPositionRef.current) {
        const [lastClientX, lastClientY] = lastPointerPositionRef.current;
        const dx = event.clientX - lastClientX;
        const dy = event.clientY - lastClientY;

        const scale = viewStateRef.current.scale || 1;
        const offsetX = -dx / scale;
        const offsetY = dy / scale;

        setViewState((prev) => ({
          ...prev,
          center: [prev.center[0] + offsetX, prev.center[1] + offsetY],
        }));

        lastPointerPositionRef.current = [event.clientX, event.clientY];
        return;
      }

      // Don't update hover state if card is pinned
      if (hovered?.isPinned) {
        return;
      }

      if (!nodesRef.current.length) {
        setHovered(null);
        return;
      }

      const [graphX, graphY] = screenToGraph(
        [cssX, cssY],
        viewStateRef.current,
        canvasSize,
      );

      const candidates = nodesRef.current
        .map((node) => {
          const dx = node.position[0] - graphX;
          const dy = node.position[1] - graphY;
          const distance = Math.hypot(dx, dy);
          return { node, distance };
        })
        .sort((a, b) => a.distance - b.distance);

      const closest = candidates[0];
      if (!closest) {
        setHovered(null);
        return;
      }

      const radius = computeRadius(closest.node.readMinutes);

      const screenDistance = closest.distance * viewStateRef.current.scale;
      if (screenDistance > radius * 1.5) {
        setHovered(null);
        return;
      }

      setHovered({
        node: closest.node,
        nodePosition: closest.node.position,
        currentStatus: closest.node.item.client_status,
        isStatusExpanded: false,
        isPinned: false,
      });
    },
    [canvasSize, hovered?.isPinned],
  );

  const handlePointerUp = useCallback(
    (event: ReactPointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) {
        return;
      }
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      isPanningRef.current = false;
      lastPointerPositionRef.current = null;
    },
    [],
  );

  const handlePointerLeave = useCallback(() => {
    isPanningRef.current = false;
    lastPointerPositionRef.current = null;
    // Only clear hover state if not pinned
    setHovered((prev) => (prev?.isPinned ? prev : null));
  }, []);

  const handleHoverStatusUpdate = useCallback(
    async (newStatus: ClientStatus) => {
      if (!hovered) return;

      setHovered((prev) =>
        prev ? { ...prev, currentStatus: newStatus } : null,
      );

      if (
        !CLIENT_UPDATABLE_STATUSES.includes(newStatus as ClientUpdatableStatus)
      ) {
        console.error("Cannot update to status:", newStatus);
        return;
      }

      try {
        const result = await updateItemStatusInQueue(
          hovered.node.id,
          newStatus as ClientUpdatableStatus,
        );
        if (!result.success) {
          console.error("Failed to update item status:", result.error);
        }
      } catch (error) {
        console.error("Error updating status:", error);
      }
    },
    [hovered],
  );

  const handleHoverStatusExpandedChange = useCallback((expanded: boolean) => {
    setHovered((prev) =>
      prev ? { ...prev, isStatusExpanded: expanded } : null,
    );
  }, []);

  // Handle Escape key to unpin card
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && hovered?.isPinned) {
        setHovered(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [hovered?.isPinned]);

  const hasItems = nodes.length > 0;
  const statusMessages = [] as string[];
  if (isLoadingReduction) {
    statusMessages.push("Updating positions…");
  }
  if (isLoadingClusters) {
    statusMessages.push("Updating clusters…");
  }

  return (
    <div className="flex h-full w-full flex-col">
      <div className="relative flex-1 min-h-0">
        <div className="relative h-full w-full overflow-hidden">
          <div ref={containerRef} className="absolute inset-0">
            <canvas
              ref={canvasRef}
              className={`h-full w-full ${hovered?.isPinned ? "cursor-default" : "cursor-grab"}`}
              style={{
                touchAction: "none",
                filter: hovered?.isPinned ? "blur(0px)" : "none",
              }}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerLeave}
            />
          </div>

          {/* Graph Panels */}
          <div className="pointer-events-auto justify-start gap-[10px] absolute left-[25px] bottom-[25px] flex flex-col-reverse">
            {/* Graph Control Panel*/}
            <div className="flex flex-col-reverse p-[5px] gap-[5px] rounded-[20px]  bg-transparent">
              <button
                type="button"
                onClick={handleCenter}
                className="flex w-[45px] h-[30px]  items-center justify-center rounded-full backdrop-blur-md bg-[#D0D0D0]/50 text-slate-600 transition hover:scale-[1.05] hover:drop-shadow-md hover:bg-[#C0C0C0]/75"
                title="Center view"
              >
                <Icon icon="ri:focus-mode" className="h-5 w-5" />
              </button>
              <button
                type="button"
                onClick={() => handleZoom(0.8)}
                className="flex w-[45px] h-[30px]  items-center justify-center rounded-full backdrop-blur-md bg-[#D0D0D0]/50 text-slate-600 transition hover:scale-[1.05] hover:drop-shadow-md hover:bg-[#C0C0C0]/75"
                title="Zoom out"
              >
                <Icon icon="mingcute:zoom-out-fill" className="h-5 w-5" />
              </button>
              <button
                type="button"
                onClick={() => handleZoom(1.25)}
                className="flex items-center w-[45px] h-[30px] justify-center rounded-full backdrop-blur-md bg-[#D0D0D0]/50 text-slate-600 transition hover:scale-[1.05] hover:drop-shadow-md hover:bg-[#C0C0C0]/75"
                title="Zoom in"
              >
                <Icon icon="mingcute:zoom-in-fill" className="h-5 w-5" />
              </button>
            </div>

            {/* Status Messages Panel */}
            {/*{statusMessages.length > 0 && (
              <div className="flex flex-col gap-2 items-end">
                {statusMessages.map((message, index) => (
                  <div
                    key={`${message}-${index}`}
                    className="rounded-full bg-white/85 px-4 py-2 text-xs font-medium text-slate-600 shadow backdrop-blur-md"
                  >
                    {message}
                  </div>
                ))}
              </div>
            )}*/}

            {/* Error Message Panel */}
            {(loadError || clusterError) && (
              <div className="flex flex-col gap-2 items-end">
                {loadError && (
                  <div className="max-w-[360px] rounded-full bg-rose-50 px-4 py-2 text-center text-xs font-medium text-rose-600 drop-shadow backdrop-blur-md">
                    {loadError}
                  </div>
                )}
                {clusterError && (
                  <div className="max-w-[360px] rounded-full bg-rose-50 px-4 py-2 text-center text-xs font-medium text-rose-600 drop-shadow backdrop-blur-md">
                    {clusterError}
                  </div>
                )}
              </div>
            )}

            {/* Cluster Label Panel*/}
            <div></div>
          </div>

          {/* No Items Message */}
          {!hasItems && !isLoadingReduction && !loadError && (
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2 text-center text-sm text-slate-500">
              <Icon
                icon="mingcute:info-fill"
                className="h-6 w-6 text-slate-400"
              />
              <span>No items available for the selected filter.</span>
            </div>
          )}

          {/* Hovered Item Preview */}
          {hovered &&
            (() => {
              const [screenX, screenY] = graphToScreen(
                hovered.nodePosition,
                viewState,
                canvasSize,
              );
              const baseRadius = computeRadius(hovered.node.readMinutes);

              const cardContent = (
                <div className="flex rounded-xl overflow-hidden border border-slate-200 bg-white/75 backdrop-blur-md drop-shadow-md w-[600px]">
                  {/* Favicon Section */}
                  <div className="flex justify-center p-[10px]">
                    <div className="flex justify-center items-center w-[35px] rounded-lg h-[35px] bg-[#D8D8D8]">
                      <div className="bg-gray-100 h-[28px] w-[28px] rounded-[6px] flex items-center justify-center border border-gray-200">
                        {hovered.node.item.favicon_url && !hoverFaviconError ? (
                          <Image
                            src={hovered.node.item.favicon_url}
                            alt={`${hovered.node.item.source_site ?? hovered.node.item.title ?? "Item"} favicon`}
                            className="w-[28px] h-[28px] object-contain rounded-[6px]"
                            width={28}
                            height={28}
                            onError={() => setHoverFaviconError(true)}
                          />
                        ) : (
                          <Icon
                            icon="streamline-plump:web"
                            width="16"
                            height="16"
                            className="text-gray-400"
                          />
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Main Content Section */}
                  <div className="flex flex-col px-[10px] py-[10px] flex-1 gap-[4px]">
                    <h2 className="text-[12px] font-bold text-black line-clamp-1">
                      {hovered.node.item.title?.trim() || "Untitled"}
                    </h2>
                    {hovered.node.item.summary && (
                      <p className="text-[10px] text-black text-left line-clamp-3 leading-[1.4]">
                        {hovered.node.item.summary}
                      </p>
                    )}
                  </div>

                  {/* Info Pills Section */}
                  <div className="flex flex-col items-end justify-start gap-[4px] p-[10px]">
                    <ItemActionPanel
                      itemId={hovered.node.item.id}
                      currentStatus={hovered.currentStatus}
                      onStatusUpdate={handleHoverStatusUpdate}
                      isExpanded={hovered.isStatusExpanded}
                      onExpandedChange={handleHoverStatusExpandedChange}
                      onDelete={removeNode}
                    />

                    {/* Reading Time Pill */}
                    <div className="flex items-center gap-[3px] px-[8px] rounded-xl bg-transparent h-[20px]">
                      <span className="text-[10px] font-medium font-mono text-black">
                        {estimateReadingTime(
                          hovered.node.item.content_token_count,
                        )}
                      </span>
                      <Icon icon="mingcute:time-fill" width="13" height="13" />
                    </div>

                    {/* Saved Date Pill */}
                    {formatSavedAt(hovered.node.item.created_at) && (
                      <div className="flex items-center gap-[3px] px-[8px] rounded-xl bg-transparent h-[20px]">
                        <span className="text-[10px] font-medium font-mono text-black">
                          {formatSavedAt(hovered.node.item.created_at)}
                        </span>
                        <Icon
                          icon="mingcute:download-fill"
                          width="13"
                          height="13"
                        />
                      </div>
                    )}
                  </div>
                </div>
              );

              return (
                <div
                  className="pointer-events-auto absolute z-10 -translate-x-1/2"
                  style={{
                    left: screenX,
                    top: screenY + baseRadius + 20,
                  }}
                >
                  {hovered.isPinned && hovered.node.item.url ? (
                    <a
                      href={hovered.node.item.url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {cardContent}
                    </a>
                  ) : (
                    cardContent
                  )}
                </div>
              );
            })()}
        </div>
      </div>
    </div>
  );
}

function graphToScreen(
  position: [number, number],
  view: CanvasViewState,
  canvasSize: CanvasSize,
): [number, number] {
  const centerX = canvasSize.width / 2;
  const centerY = canvasSize.height / 2;
  const scale = view.scale || 1;
  const x = centerX + (position[0] - view.center[0]) * scale;
  const y = centerY - (position[1] - view.center[1]) * scale;
  return [x, y];
}

function screenToGraph(
  screen: [number, number],
  view: CanvasViewState,
  canvasSize: CanvasSize,
): [number, number] {
  const centerX = canvasSize.width / 2;
  const centerY = canvasSize.height / 2;
  const scale = view.scale || 1;
  const x = view.center[0] + (screen[0] - centerX) / scale;
  const y = view.center[1] - (screen[1] - centerY) / scale;
  return [x, y];
}
