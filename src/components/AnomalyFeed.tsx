"use client";

import { motion, AnimatePresence } from "framer-motion";
import type { Alert, AnomalyType, AlertLevel } from "@/lib/types";

interface AnomalyFeedProps {
  alerts: Alert[];
  maxItems?: number;
  compact?: boolean;
  onAlertClick?: (alert: Alert) => void;
  highlightIds?: Set<string>;
}

// ── Color mapping by alert level ─────────────────────────────
const levelColors: Record<AlertLevel, { dot: string; border: string; pulse?: string }> = {
  CRITICAL: { dot: "bg-red-500", border: "border-l-red-500", pulse: "live-indicator" },
  WARNING: { dot: "bg-amber-500", border: "border-l-amber-500" },
  INFO: { dot: "bg-teal-500", border: "border-l-teal-500" },
  NORMAL: { dot: "bg-zinc-500", border: "border-l-zinc-500" },
};

// ── Readable anomaly type labels ─────────────────────────────
const anomalyLabels: Record<AnomalyType, string> = {
  RANGE_ANOMALY: "Range Anomaly",
  TEMPORAL_ANOMALY: "Temporal Anomaly",
  CLUSTER_ANOMALY: "Cluster Event",
  RARITY_ANOMALY: "Rarity Anomaly",
  CAPTIVE_ESCAPE: "Captive Escape",
  MISIDENTIFICATION: "Misidentification",
  HABITAT_MISMATCH: "Habitat Mismatch",
  POACHING_INDICATOR: "Poaching Risk",
  NORMAL: "Normal",
};

// ── Relative time helper ─────────────────────────────────────
function timeAgo(timestamp: string): string {
  const now = Date.now();
  const then = new Date(timestamp).getTime();
  const diffMs = now - then;

  if (diffMs < 0) return "just now";

  const seconds = Math.floor(diffMs / 1000);
  if (seconds < 60) return `${seconds}s ago`;

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;

  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;

  const months = Math.floor(days / 30);
  return `${months}mo ago`;
}

export default function AnomalyFeed({
  alerts,
  maxItems = 10,
  compact = false,
  onAlertClick,
  highlightIds,
}: AnomalyFeedProps) {
  const sorted = [...alerts]
    .sort(
      (a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    )
    .slice(0, maxItems);

  return (
    <div className="space-y-2">
      <AnimatePresence initial={false}>
        {sorted.map((alert, index) => {
          const colors = levelColors[alert.alert_level];
          const isNew = highlightIds?.has(alert.alert_id);
          const clickable = !!onAlertClick;

          return (
            <motion.div
              key={alert.alert_id}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 12 }}
              transition={{ duration: 0.25, delay: index * 0.04 }}
              onClick={() => onAlertClick?.(alert)}
              className={`rounded-lg border border-zinc-800/60 border-l-2 ${colors.border} bg-white/[0.02] transition-all ${
                compact ? "px-3 py-2" : "px-4 py-3"
              } ${
                clickable
                  ? "cursor-pointer hover:bg-white/[0.05] hover:border-zinc-700/60 active:scale-[0.99]"
                  : ""
              } ${
                isNew ? "ring-1 ring-radar-green/30 bg-radar-green/[0.03]" : ""
              }`}
            >
              <div className="flex items-start gap-3">
                {/* Alert level dot */}
                <div className="pt-1.5 shrink-0">
                  <span
                    className={`block w-2 h-2 rounded-full ${colors.dot} ${
                      colors.pulse || ""
                    }`}
                  />
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0 space-y-1">
                  {/* Top row: species + anomaly type */}
                  <div className="flex items-center justify-between gap-2">
                    <span
                      className={`font-semibold text-white truncate ${compact ? "text-xs" : "text-sm"}`}
                    >
                      {alert.common_name}
                    </span>
                    <span
                      className={`shrink-0 text-zinc-500 ${compact ? "text-[10px]" : "text-xs"}`}
                    >
                      {timeAgo(alert.timestamp)}
                    </span>
                  </div>

                  {/* Anomaly type tag */}
                  <span
                    className={`inline-block px-1.5 py-0.5 rounded bg-white/5 text-zinc-400 ${compact ? "text-[10px]" : "text-[11px]"}`}
                  >
                    {anomalyLabels[alert.anomaly_type]}
                  </span>

                  {/* Location */}
                  {alert.place_name && (
                    <p
                      className={`text-zinc-500 truncate ${compact ? "text-[10px]" : "text-xs"}`}
                    >
                      {alert.place_name}
                    </p>
                  )}

                  {/* Confidence — hidden in compact mode */}
                  {!compact && (
                    <div className="flex items-center gap-2">
                      <p className="text-[11px] text-zinc-600">
                        Confidence:{" "}
                        <span className="text-zinc-400">
                          {Math.round(alert.confidence_score * 100)}%
                        </span>
                      </p>
                      {clickable && (
                        <span className="text-[10px] text-zinc-600 ml-auto">
                          Click to investigate
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>

      {sorted.length === 0 && (
        <div className="text-center text-zinc-600 text-sm py-8">
          No anomaly alerts
        </div>
      )}
    </div>
  );
}
