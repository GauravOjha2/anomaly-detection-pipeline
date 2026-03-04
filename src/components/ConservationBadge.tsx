"use client";

import type { ConservationStatus } from "@/lib/types";

interface ConservationBadgeProps {
  status: ConservationStatus;
  size?: "sm" | "md" | "lg";
}

const statusColors: Record<
  ConservationStatus,
  { bg: string; text: string; border: string }
> = {
  critically_endangered: {
    bg: "bg-red-500/15",
    text: "text-red-400",
    border: "border-red-500/30",
  },
  endangered: {
    bg: "bg-orange-500/15",
    text: "text-orange-400",
    border: "border-orange-500/30",
  },
  vulnerable: {
    bg: "bg-yellow-500/15",
    text: "text-yellow-400",
    border: "border-yellow-500/30",
  },
  near_threatened: {
    bg: "bg-teal-500/15",
    text: "text-teal-400",
    border: "border-teal-500/30",
  },
  least_concern: {
    bg: "bg-emerald-500/15",
    text: "text-emerald-400",
    border: "border-emerald-500/30",
  },
  unknown: {
    bg: "bg-zinc-500/15",
    text: "text-zinc-400",
    border: "border-zinc-500/30",
  },
};

const statusLabels: Record<ConservationStatus, string> = {
  critically_endangered: "Critically Endangered",
  endangered: "Endangered",
  vulnerable: "Vulnerable",
  near_threatened: "Near Threatened",
  least_concern: "Least Concern",
  unknown: "Unknown",
};

const sizeClasses: Record<"sm" | "md" | "lg", string> = {
  sm: "text-[10px] px-2 py-0.5",
  md: "text-xs px-2.5 py-1",
  lg: "text-sm px-3 py-1.5",
};

export default function ConservationBadge({
  status,
  size = "md",
}: ConservationBadgeProps) {
  const colors = statusColors[status];
  const label = statusLabels[status];

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-medium border ${colors.bg} ${colors.text} ${colors.border} ${sizeClasses[size]}`}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current" />
      {label}
    </span>
  );
}
