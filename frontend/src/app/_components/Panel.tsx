"use client";

import { ReactNode, useRef, useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import TabPanel from "./TabPanel";
import AddPanel from "./AddPanel";
import SettingsPill from "./SettingsPill";
import LabelPanel from "./LabelPanel";
import { useClusterContext } from "@/app/_contexts/ClusterContext";
import {
  CLUSTER_EVENT_NAME,
  CLUSTER_STATE_EVENT_NAME,
  type ClusterLegendEventDetail,
  type ClusterStateEventDetail,
} from "@/app/graph/clusterEvents";

type PanelSession = { user_id: string; username: string } | null;

type PanelProps = {
  children?: ReactNode;
  session: PanelSession;
};

export default function Panel({ children, session }: PanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const pathname = usePathname();
  const [isTransitioning, setIsTransitioning] = useState(false);
  const { updateClusters, clearClusters, setClusteringState } =
    useClusterContext();

  // Determine if AddPanel should be shown (hide on graph and search pages)
  const shouldShowAddPanel = pathname !== "/graph" && pathname !== "/search";

  // Determine if LabelPanel should be shown (only on graph page)
  const shouldShowLabelPanel = pathname === "/graph";

  // Listen for cluster events
  useEffect(() => {
    const handleClusterEvent = (
      event: CustomEvent<ClusterLegendEventDetail>,
    ) => {
      updateClusters(event.detail.items);
    };

    const handleClusterStateEvent = (
      event: CustomEvent<ClusterStateEventDetail>,
    ) => {
      setClusteringState(event.detail.state);
    };

    window.addEventListener(
      CLUSTER_EVENT_NAME,
      handleClusterEvent as EventListener,
    );

    window.addEventListener(
      CLUSTER_STATE_EVENT_NAME,
      handleClusterStateEvent as EventListener,
    );

    return () => {
      window.removeEventListener(
        CLUSTER_EVENT_NAME,
        handleClusterEvent as EventListener,
      );
      window.removeEventListener(
        CLUSTER_STATE_EVENT_NAME,
        handleClusterStateEvent as EventListener,
      );
    };
  }, [updateClusters, setClusteringState]);

  // Clear clusters when leaving graph page
  useEffect(() => {
    if (!shouldShowLabelPanel) {
      clearClusters();
    }
  }, [shouldShowLabelPanel, clearClusters]);

  // Update panels width CSS variable
  useEffect(() => {
    const updatePanelsWidth = () => {
      const width = panelRef.current?.offsetWidth ?? 0;
      document.documentElement.style.setProperty(
        "--panels-width",
        `${width}px`,
      );
    };

    const observedPanel = panelRef.current;
    const resizeObserver =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(updatePanelsWidth)
        : null;

    if (observedPanel) {
      updatePanelsWidth();
      resizeObserver?.observe(observedPanel);
    }

    window.addEventListener("resize", updatePanelsWidth);

    return () => {
      window.removeEventListener("resize", updatePanelsWidth);
      resizeObserver?.disconnect();
      document.documentElement.style.setProperty("--panels-width", "0px");
    };
  }, []);

  // Handle route changes for smooth transitions
  useEffect(() => {
    setIsTransitioning(true);
    const timer = setTimeout(() => {
      setIsTransitioning(false);
    }, 150);

    return () => clearTimeout(timer);
  }, [pathname]);

  return (
    <>
      {/*Top Panel*/}
      <motion.div
        ref={panelRef}
        className="flex flex-col gap-[15px] items-end justify-center absolute right-[25px] top-[25px]"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
      >
        <TabPanel />

        {/* Control Container with Framer Motion animations */}
        <motion.div
          className="panel-light relative z-0 overflow-hidden p-[10px]"
          layout
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{
            opacity: isTransitioning ? 0.9 : 1,
            scale: isTransitioning ? 0.98 : 1,
          }}
          transition={{
            layout: { duration: 0.3, ease: [0.4, 0, 0.2, 1] },
            opacity: { duration: 0.3, ease: [0.4, 0, 0.2, 1] },
            scale: { duration: 0.3, ease: [0.4, 0, 0.2, 1] },
          }}
        >
          <AnimatePresence mode="wait">
            {/*<motion.div
              key={pathname}
              className="flex flex-col items-center gap-[5px] w-fit"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
            >*/}
            <div>{children}</div>
            {/*</motion.div>*/}
          </AnimatePresence>
        </motion.div>

        {/* Label Panel */}
        <AnimatePresence>
          {shouldShowLabelPanel && (
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{
                opacity: isTransitioning ? 0.9 : 1,
                y: 0,
                scale: isTransitioning ? 0.98 : 1,
              }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            >
              <LabelPanel />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Bottom Panel */}
      <motion.div
        className="flex gap-[15px] items-end justify-center flex-col-reverse absolute right-[25px] bottom-[25px]"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut", delay: 0.1 }}
      >
        {/* Settings Pill */}
        <div className="w-fit min-w-[100px] flex flex-col">
          <AnimatePresence>
            {session && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.2 }}
              >
                <SettingsPill username={session.username} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Add Panel - with smooth show/hide animation */}
        <AnimatePresence>
          {shouldShowAddPanel && (
            <motion.div
              className="flex w-[200px]"
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            >
              <AddPanel />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </>
  );
}
