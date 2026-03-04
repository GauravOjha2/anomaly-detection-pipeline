"use client";

import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { Alert, AnomalyType } from "@/lib/types";

// ── Color palette ────────────────────────────────────────────
const TYPE_COLORS: Record<string, string> = {
  RANGE_ANOMALY: "#2dd4bf",
  TEMPORAL_ANOMALY: "#8b5cf6",
  CLUSTER_ANOMALY: "#06b6d4",
  RARITY_ANOMALY: "#f59e0b",
  CAPTIVE_ESCAPE: "#ec4899",
  MISIDENTIFICATION: "#6b7280",
  HABITAT_MISMATCH: "#10b981",
  POACHING_INDICATOR: "#ef4444",
  NORMAL: "#52525b",
};

const TYPE_LABELS: Record<AnomalyType, string> = {
  RANGE_ANOMALY: "Range",
  TEMPORAL_ANOMALY: "Temporal",
  CLUSTER_ANOMALY: "Cluster",
  RARITY_ANOMALY: "Rarity",
  CAPTIVE_ESCAPE: "Captive",
  MISIDENTIFICATION: "Misid.",
  HABITAT_MISMATCH: "Habitat",
  POACHING_INDICATOR: "Poaching",
  NORMAL: "Normal",
};

// ── Distribution by type (donut chart) ───────────────────────

interface TypeDistributionProps {
  alerts: Alert[];
  className?: string;
}

export function AnomalyTypeChart({ alerts, className = "" }: TypeDistributionProps) {
  const counts = new Map<string, number>();
  for (const a of alerts) {
    if (a.anomaly_type === "NORMAL") continue;
    counts.set(a.anomaly_type, (counts.get(a.anomaly_type) || 0) + 1);
  }

  const data = Array.from(counts.entries())
    .map(([type, count]) => ({
      name: TYPE_LABELS[type as AnomalyType] || type,
      value: count,
      color: TYPE_COLORS[type] || "#6b7280",
    }))
    .sort((a, b) => b.value - a.value);

  if (data.length === 0) {
    return (
      <div className={`flex items-center justify-center h-[200px] text-zinc-600 text-sm ${className}`}>
        No anomaly data
      </div>
    );
  }

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={50}
            outerRadius={80}
            paddingAngle={2}
            dataKey="value"
            stroke="none"
          >
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              background: "#18181b",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 8,
              color: "#fff",
              fontSize: 12,
            }}
          />
        </PieChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 mt-2 px-2">
        {data.map((entry) => (
          <div key={entry.name} className="flex items-center gap-1.5 text-[10px] text-zinc-400">
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: entry.color }}
            />
            {entry.name} ({entry.value})
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Distribution by species (bar chart) ──────────────────────

interface SpeciesDistributionProps {
  alerts: Alert[];
  className?: string;
}

export function SpeciesBarChart({ alerts, className = "" }: SpeciesDistributionProps) {
  const counts = new Map<string, number>();
  for (const a of alerts) {
    counts.set(a.common_name, (counts.get(a.common_name) || 0) + 1);
  }

  const data = Array.from(counts.entries())
    .map(([name, count]) => ({ name: name.length > 14 ? name.slice(0, 12) + "..." : name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);

  if (data.length === 0) {
    return (
      <div className={`flex items-center justify-center h-[200px] text-zinc-600 text-sm ${className}`}>
        No species data
      </div>
    );
  }

  return (
    <div className={className} style={{ background: "transparent" }}>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 5 }} style={{ background: "transparent" }}>
          <XAxis
            dataKey="name"
            tick={{ fill: "#71717a", fontSize: 9 }}
            axisLine={false}
            tickLine={false}
            angle={-30}
            textAnchor="end"
            height={50}
          />
          <YAxis
            tick={{ fill: "#71717a", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            allowDecimals={false}
          />
          <Tooltip
            contentStyle={{
              background: "#18181b",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 8,
              color: "#fff",
              fontSize: 12,
            }}
          />
          <Bar dataKey="count" fill="#2dd4bf" radius={[4, 4, 0, 0]} maxBarSize={36} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
