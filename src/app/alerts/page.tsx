"use client";

import { useState, useMemo, useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  Filter,
  RefreshCw,
  MapPin,
  Clock,
  Shield,
  Search,
  ArrowRight,
} from "lucide-react";
import Navbar from "@/components/Navbar";
import ConservationBadge from "@/components/ConservationBadge";
import InvestigationModal from "@/components/InvestigationModal";
import { useSentinel } from "@/lib/SentinelContext";
import { getSpeciesById } from "@/lib/species-data";
import { getRelativeTime } from "@/lib/time";
import type { Alert, AlertLevel, AnomalyType } from "@/lib/types";

const ANOMALY_LABELS: Record<AnomalyType, string> = {
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

const LEVEL_STYLES: Record<AlertLevel, { bg: string; text: string; border: string }> = {
  CRITICAL: { bg: "bg-red-500/15", text: "text-red-400", border: "border-l-red-500" },
  WARNING: { bg: "bg-amber-500/15", text: "text-amber-400", border: "border-l-amber-500" },
  INFO: { bg: "bg-teal-500/15", text: "text-teal-400", border: "border-l-teal-500" },
  NORMAL: { bg: "bg-zinc-500/15", text: "text-zinc-400", border: "border-l-zinc-500" },
};

const LEVEL_FILTERS: { label: string; value: AlertLevel | "ALL" }[] = [
  { label: "All", value: "ALL" },
  { label: "Critical", value: "CRITICAL" },
  { label: "Warning", value: "WARNING" },
  { label: "Info", value: "INFO" },
];

const TYPE_FILTERS: { label: string; value: AnomalyType | "ALL" }[] = [
  { label: "All Types", value: "ALL" },
  { label: "Range Anomaly", value: "RANGE_ANOMALY" },
  { label: "Cluster Event", value: "CLUSTER_ANOMALY" },
  { label: "Poaching Risk", value: "POACHING_INDICATOR" },
  { label: "Habitat Mismatch", value: "HABITAT_MISMATCH" },
  { label: "Captive Escape", value: "CAPTIVE_ESCAPE" },
  { label: "Rarity Anomaly", value: "RARITY_ANOMALY" },
  { label: "Temporal Anomaly", value: "TEMPORAL_ANOMALY" },
];

function AlertsPageContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { alerts, isLoading, error, fetchAlerts } = useSentinel();
  
  const [levelFilter, setLevelFilter] = useState<AlertLevel | "ALL">(
    (searchParams.get("level") as AlertLevel) || "ALL"
  );
  const [typeFilter, setTypeFilter] = useState<AnomalyType | "ALL">(
    (searchParams.get("type") as AnomalyType) || "ALL"
  );
  const [speciesFilter, setSpeciesFilter] = useState(searchParams.get("species") || "");
  const [searchQuery, setSearchQuery] = useState(searchParams.get("q") || "");
  const [investigatingAlert, setInvestigatingAlert] = useState<Alert | null>(null);

  useEffect(() => {
    const params = new URLSearchParams();
    if (levelFilter !== "ALL") params.set("level", levelFilter);
    if (typeFilter !== "ALL") params.set("type", typeFilter);
    if (speciesFilter) params.set("species", speciesFilter);
    if (searchQuery) params.set("q", searchQuery);
    const newUrl = params.toString() ? `?${params.toString()}` : window.location.pathname;
    router.replace(newUrl, { scroll: false });
  }, [levelFilter, typeFilter, speciesFilter, searchQuery, router]);

  const filtered = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    return alerts
      .filter((a) => (levelFilter === "ALL" ? true : a.alert_level === levelFilter))
      .filter((a) => (typeFilter === "ALL" ? true : a.anomaly_type === typeFilter))
      .filter((a) => {
        if (speciesFilter) {
          const species = getSpeciesById(speciesFilter);
          if (species && a.species_name !== species.scientificName) return false;
        }
        if (!q) return true;
        return (
          a.common_name.toLowerCase().includes(q) ||
          a.species_name.toLowerCase().includes(q) ||
          (a.place_name && a.place_name.toLowerCase().includes(q)) ||
          ANOMALY_LABELS[a.anomaly_type]?.toLowerCase().includes(q)
        );
      })
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }, [alerts, levelFilter, typeFilter, speciesFilter, searchQuery]);

  const criticalCount = filtered.filter((a) => a.alert_level === "CRITICAL").length;

  const nearbyAlerts = useMemo(() => {
    if (!investigatingAlert) return [];
    return alerts.filter(
      (a) =>
        a.alert_id !== investigatingAlert.alert_id &&
        Math.abs(a.location.lat - investigatingAlert.location.lat) < 5 &&
        Math.abs(a.location.lng - investigatingAlert.location.lng) < 5
    );
  }, [investigatingAlert, alerts]);

  const hasActiveFilters = levelFilter !== "ALL" || typeFilter !== "ALL" || speciesFilter || searchQuery;

  return (
    <div className="min-h-screen bg-[#09090b] text-white">
      <Navbar />

      <div className="max-w-4xl mx-auto px-4 md:px-6 pt-24 pb-12">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <div className="flex items-center justify-between mb-1">
            <h1 className="text-2xl font-bold text-white">Anomaly Alerts</h1>
            <button
              onClick={() => fetchAlerts()}
              disabled={isLoading}
              className="p-2 rounded-lg border border-white/[0.06] hover:border-radar-green/30 hover:bg-radar-green/5 transition-all disabled:opacity-50"
              aria-label="Refresh"
            >
              <RefreshCw className={`w-4 h-4 text-zinc-400 ${isLoading ? "animate-spin" : ""}`} />
            </button>
          </div>
          <p className="text-sm text-zinc-400 mb-4">Real-time wildlife anomaly detection alerts</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.06 }}
          className="relative mb-4"
        >
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by species, location, anomaly type..."
            className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-white/[0.03] border border-white/[0.06] text-sm text-white placeholder-zinc-500 outline-none focus:border-radar-green/30 focus:ring-1 focus:ring-radar-green/20 transition-all"
          />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="flex flex-wrap items-center gap-4 mb-6"
        >
          <div className="flex items-center gap-1.5">
            <Filter className="w-3.5 h-3.5 text-zinc-500" />
            {LEVEL_FILTERS.map((f) => (
              <button
                key={f.value}
                onClick={() => setLevelFilter(f.value)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  levelFilter === f.value
                    ? "bg-radar-green/15 text-radar-green border border-radar-green/30"
                    : "text-zinc-500 hover:text-zinc-300 border border-white/[0.06] hover:border-white/[0.1]"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            {TYPE_FILTERS.map((f) => (
              <button
                key={f.value}
                onClick={() => setTypeFilter(f.value)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  typeFilter === f.value
                    ? "bg-radar-green/15 text-radar-green border border-radar-green/30"
                    : "text-zinc-500 hover:text-zinc-300 border border-white/[0.06] hover:border-white/[0.1]"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </motion.div>

        {!isLoading && (
          <p className="text-xs text-zinc-500 mb-4">
            <span className="text-white font-medium">{filtered.length}</span> alerts detected
            {criticalCount > 0 && (
              <> · <span className="text-red-400 font-medium">{criticalCount} critical</span></>
            )}
          </p>
        )}

        {isLoading && alerts.length === 0 && (
          <div className="space-y-3">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="radar-card rounded-xl h-24 skeleton" style={{ animationDelay: `${i * 80}ms` }} />
            ))}
          </div>
        )}

        {error && alerts.length === 0 && (
          <div className="p-6 rounded-xl bg-red-500/10 border border-red-500/20 text-center">
            <AlertTriangle className="w-6 h-6 text-red-400 mx-auto mb-2" />
            <p className="text-sm text-red-400 mb-3">{error}</p>
            <button onClick={() => fetchAlerts()} className="text-xs text-red-400 hover:text-red-300 underline">
              Retry
            </button>
          </div>
        )}

        {!isLoading && !error && filtered.length === 0 && (
          <div className="text-center py-20">
            <Shield className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
            <p className="text-sm text-zinc-500">No alerts match your filters</p>
            {hasActiveFilters && (
              <button
                onClick={() => {
                  setLevelFilter("ALL");
                  setTypeFilter("ALL");
                  setSpeciesFilter("");
                  setSearchQuery("");
                }}
                className="text-xs text-radar-green hover:underline mt-3"
              >
                Clear all filters
              </button>
            )}
          </div>
        )}

        {!isLoading && filtered.length > 0 && (
          <div className="space-y-3">
            {filtered.map((alert, i) => {
              const style = LEVEL_STYLES[alert.alert_level];
              const species = getSpeciesById(alert.species_name.toLowerCase().replace(/ /g, '-'));
              return (
                <motion.div
                  key={alert.alert_id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25, delay: i * 0.03 }}
                  onClick={() => setInvestigatingAlert(alert)}
                  className={`radar-card rounded-xl border-l-4 ${style.border} flex gap-4 p-4 cursor-pointer hover:bg-white/[0.04] active:scale-[0.995] transition-all`}
                >
                  {alert.photo_url && (
                    <img
                      src={alert.photo_url}
                      alt={alert.common_name}
                      className="w-16 h-16 rounded-lg object-cover border border-white/[0.08] shrink-0 hidden sm:block"
                      loading="lazy"
                    />
                  )}

                  <div className="flex-1 min-w-0 space-y-1.5">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-semibold ${style.bg} ${style.text}`}>
                        {alert.alert_level}
                      </span>
                      <span className="text-sm font-semibold text-white truncate">{alert.common_name}</span>
                      {species && (
                        <Link
                          href={`/species/${species.id}`}
                          onClick={(e) => e.stopPropagation()}
                          className="text-xs text-radar-green hover:underline flex items-center gap-0.5"
                        >
                          <ArrowRight className="w-3 h-3 rotate-[-90deg]" />
                          Profile
                        </Link>
                      )}
                    </div>

                    <span className="inline-block text-[11px] px-2 py-0.5 rounded bg-white/[0.05] text-zinc-400">
                      {ANOMALY_LABELS[alert.anomaly_type]}
                    </span>

                    <div className="flex flex-wrap items-center gap-3 text-[11px] text-zinc-500">
                      {alert.place_name && (
                        <span className="flex items-center gap-1 truncate">
                          <MapPin className="w-3 h-3 shrink-0" />
                          {alert.place_name}
                        </span>
                      )}
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3 shrink-0" />
                        {getRelativeTime(alert.timestamp)}
                      </span>
                    </div>

                    <div className="flex items-center gap-2 pt-0.5">
                      <span className="text-[10px] text-zinc-600">Confidence</span>
                      <div className="flex-1 max-w-[120px] h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
                        <div
                          className="h-full bg-radar-green rounded-full transition-all"
                          style={{ width: `${alert.confidence_score * 100}%` }}
                        />
                      </div>
                      <span className="text-[10px] text-zinc-400 font-mono">
                        {Math.round(alert.confidence_score * 100)}%
                      </span>
                    </div>
                  </div>

                  <div className="shrink-0 hidden md:flex flex-col items-end justify-between">
                    <ConservationBadge status={alert.conservation_status} size="sm" />
                    <span className="text-[10px] text-zinc-600 mt-auto">Click to investigate</span>
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>

      <InvestigationModal
        alert={investigatingAlert}
        onClose={() => setInvestigatingAlert(null)}
        nearbyAlerts={nearbyAlerts}
      />
    </div>
  );
}

import { Suspense } from "react";

export default function AlertsPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#09090b]" />}>
      <AlertsPageContent />
    </Suspense>
  );
}
