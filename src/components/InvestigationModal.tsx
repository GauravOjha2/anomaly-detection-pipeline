"use client";

import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  MapPin,
  Clock,
  AlertTriangle,
  Shield,
  Eye,
  Crosshair,
  TrendingUp,
  ArrowRight,
} from "lucide-react";
import Link from "next/link";
import ConservationBadge from "./ConservationBadge";
import type { Alert, AnomalyType, AlertLevel } from "@/lib/types";

interface InvestigationModalProps {
  alert: Alert | null;
  onClose: () => void;
  nearbyAlerts?: Alert[];
}

// ── Anomaly type explanations ────────────────────────────────
const anomalyExplanations: Record<AnomalyType, { label: string; description: string; action: string }> = {
  RANGE_ANOMALY: {
    label: "Range Anomaly",
    description: "This species was observed significantly outside its known geographic range. This could indicate migration shifts, habitat displacement, or a misidentified specimen.",
    action: "Verify sighting coordinates and cross-reference with recent range expansion studies. Report to local wildlife authorities if confirmed.",
  },
  TEMPORAL_ANOMALY: {
    label: "Temporal Anomaly",
    description: "This sighting occurred at an unusual time — either seasonally unexpected or during atypical hours for this species\u2019 known behavioral patterns.",
    action: "Check for environmental disturbances (fires, flooding, construction) that may have altered the species\u2019 routine.",
  },
  CLUSTER_ANOMALY: {
    label: "Cluster Event",
    description: "An unusual concentration of sightings detected in a localized area. This could indicate a food source, breeding event, or environmental stressor causing aggregation.",
    action: "Monitor the area for potential causes. Deploy additional camera traps if resources allow.",
  },
  RARITY_ANOMALY: {
    label: "Rarity Anomaly",
    description: "An extremely rare or critically endangered species was detected in an unexpected location. Given the species\u2019 low population count, every sighting is significant.",
    action: "Prioritize verification. Coordinate with conservation teams for potential population monitoring.",
  },
  CAPTIVE_ESCAPE: {
    label: "Possible Captive Escape",
    description: "Indicators suggest this individual may have escaped from captivity. The location and behavior patterns are inconsistent with wild populations.",
    action: "Contact nearby zoos, sanctuaries, and wildlife facilities. Report to wildlife enforcement if confirmed.",
  },
  MISIDENTIFICATION: {
    label: "Potential Misidentification",
    description: "Statistical analysis suggests this identification may be incorrect. The observation\u2019s features are significantly outside the expected parameters for this species.",
    action: "Request additional photos or expert review. Flag for community verification.",
  },
  HABITAT_MISMATCH: {
    label: "Habitat Mismatch",
    description: "This species was observed in a habitat type that doesn\u2019t match its known ecological requirements. This may indicate habitat loss forcing displacement.",
    action: "Investigate nearby habitat conditions. Document potential land use changes in the area.",
  },
  POACHING_INDICATOR: {
    label: "Poaching Risk Indicator",
    description: "Detection patterns suggest potential poaching activity. This could include unusual absence patterns, suspicious timing, or geographic indicators near known trafficking routes.",
    action: "Alert local anti-poaching units immediately. Do not share precise coordinates publicly.",
  },
  NORMAL: {
    label: "Normal Observation",
    description: "This observation falls within expected parameters for this species.",
    action: "No action required. Continue routine monitoring.",
  },
};

const levelStyles: Record<AlertLevel, { bg: string; text: string; glow: string }> = {
  CRITICAL: { bg: "bg-red-500/15", text: "text-red-400", glow: "shadow-red-500/20" },
  WARNING: { bg: "bg-amber-500/15", text: "text-amber-400", glow: "shadow-amber-500/20" },
  INFO: { bg: "bg-teal-500/15", text: "text-teal-400", glow: "shadow-teal-500/20" },
  NORMAL: { bg: "bg-zinc-500/15", text: "text-zinc-400", glow: "shadow-zinc-500/20" },
};

function timeAgo(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime();
  if (diff < 0) return "just now";
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function speciesNameToId(scientificName: string): string {
  return scientificName.toLowerCase().replace(/\s+/g, "-");
}

export default function InvestigationModal({
  alert,
  onClose,
  nearbyAlerts = [],
}: InvestigationModalProps) {
  if (!alert) return null;

  const explanation = anomalyExplanations[alert.anomaly_type];
  const style = levelStyles[alert.alert_level];
  const confidence = Math.round(alert.confidence_score * 100);

  return (
    <AnimatePresence>
      {alert && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-[80] bg-black/60 backdrop-blur-sm"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ type: "spring", stiffness: 400, damping: 30 }}
            className="fixed inset-4 md:inset-auto md:top-1/2 md:left-1/2 md:-translate-x-1/2 md:-translate-y-1/2 md:w-[680px] md:max-h-[85vh] z-[90] bg-[#0c0c0e] border border-white/[0.08] rounded-2xl shadow-2xl overflow-y-auto"
          >
            {/* Close button */}
            <button
              onClick={onClose}
              className="absolute top-4 right-4 p-1.5 rounded-lg bg-black/60 hover:bg-black/80 text-white/80 hover:text-white transition-colors z-10 backdrop-blur-sm"
            >
              <X className="w-4 h-4" />
            </button>

            {/* Header with photo */}
            <div className="relative">
              {alert.photo_url && (
                <div className="relative h-40 overflow-hidden rounded-t-2xl">
                  <img
                    src={alert.photo_url}
                    alt={alert.common_name}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-[#0c0c0e] via-[#0c0c0e]/40 to-transparent" />
                </div>
              )}

              <div className={`px-6 ${alert.photo_url ? "-mt-16 relative" : "pt-6"}`}>
                {/* Level badge */}
                <div className="flex items-center gap-2 mb-2">
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded font-mono font-bold ${style.bg} ${style.text}`}
                  >
                    {alert.alert_level}
                  </span>
                  <span className={`text-[10px] px-2 py-0.5 rounded bg-white/[0.05] text-zinc-400`}>
                    {explanation.label}
                  </span>
                </div>

                {/* Species name */}
                <h2 className="text-xl font-bold text-white">{alert.common_name}</h2>
                <p className="text-sm text-zinc-400 italic">{alert.species_name}</p>
              </div>
            </div>

            <div className="px-6 pb-6 space-y-5 mt-4">
              {/* Meta row */}
              <div className="flex flex-wrap items-center gap-4 text-xs text-zinc-500">
                {alert.place_name && (
                  <span className="flex items-center gap-1.5">
                    <MapPin className="w-3.5 h-3.5" />
                    {alert.place_name}
                  </span>
                )}
                <span className="flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5" />
                  {timeAgo(alert.timestamp)}
                </span>
                <span className="flex items-center gap-1.5">
                  <Crosshair className="w-3.5 h-3.5" />
                  {alert.location.lat.toFixed(4)}, {alert.location.lng.toFixed(4)}
                </span>
                <ConservationBadge status={alert.conservation_status} size="sm" />
              </div>

              {/* Why it was flagged */}
              <div className="space-y-2">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400" />
                  Why This Was Flagged
                </h3>
                <p className="text-sm text-zinc-300 leading-relaxed">
                  {explanation.description}
                </p>
              </div>

              {/* Confidence breakdown */}
              <div className="space-y-2">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-radar-green" />
                  Detection Confidence
                </h3>
                <div className="flex items-center gap-3">
                  <div className="flex-1 h-2 bg-white/[0.06] rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${confidence}%` }}
                      transition={{ duration: 0.8, ease: "easeOut" }}
                      className={`h-full rounded-full ${
                        confidence >= 80
                          ? "bg-red-500"
                          : confidence >= 60
                          ? "bg-amber-500"
                          : "bg-teal-500"
                      }`}
                    />
                  </div>
                  <span className="text-sm font-mono text-white font-semibold w-12 text-right">
                    {confidence}%
                  </span>
                </div>
                <div className="flex items-center gap-4 text-[11px] text-zinc-500">
                  <span>
                    Models: <span className="text-zinc-400">{alert.models_used.length}</span>
                  </span>
                  <span>
                    Features: <span className="text-zinc-400">{alert.features_extracted}</span>
                  </span>
                </div>
              </div>

              {/* Recommended action */}
              <div className="space-y-2">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <Shield className="w-4 h-4 text-emerald-400" />
                  Recommended Action
                </h3>
                <p className="text-sm text-zinc-300 leading-relaxed">
                  {explanation.action}
                </p>
              </div>

              {/* Cross-links */}
              <div className="flex flex-wrap gap-2">
                <Link
                  href={`/species/${speciesNameToId(alert.species_name)}`}
                  onClick={onClose}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-radar-green/10 text-radar-green border border-radar-green/20 hover:bg-radar-green/20 transition-all"
                >
                  View Species Profile
                  <ArrowRight className="w-3.5 h-3.5" />
                </Link>
                <Link
                  href={`/alerts?species=${speciesNameToId(alert.species_name)}`}
                  onClick={onClose}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-white/[0.05] text-zinc-400 border border-white/[0.08] hover:text-white hover:border-white/[0.15] transition-all"
                >
                  All {alert.common_name} Alerts
                  <ArrowRight className="w-3.5 h-3.5" />
                </Link>
              </div>

              {/* Nearby sightings */}
              {nearbyAlerts.length > 0 && (
                <div className="space-y-2">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <Eye className="w-4 h-4 text-teal-400" />
                    Nearby Sightings ({nearbyAlerts.length})
                  </h3>
                  <div className="space-y-1.5 max-h-32 overflow-y-auto">
                    {nearbyAlerts.slice(0, 5).map((nearby) => (
                      <div
                        key={nearby.alert_id}
                        className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/[0.03] text-xs"
                      >
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            nearby.alert_level === "CRITICAL"
                              ? "bg-red-500"
                              : nearby.alert_level === "WARNING"
                              ? "bg-amber-500"
                              : "bg-teal-500"
                          }`}
                        />
                        <span className="text-white font-medium truncate flex-1">
                          {nearby.common_name}
                        </span>
                        <span className="text-zinc-500">{timeAgo(nearby.timestamp)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
