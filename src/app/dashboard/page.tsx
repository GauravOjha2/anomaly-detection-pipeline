"use client";

import { useState, useEffect, useRef, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  Eye,
  RefreshCw,
  Clock,
  Shield,
  Heart,
  Radio,
  PawPrint,
  Locate,
  TrendingDown,
  TrendingUp,
  ArrowRight,
  Zap,
  Target,
  Globe2,
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import Navbar from "@/components/Navbar";
import WildlifeMap from "@/components/WildlifeMap";
import AnomalyFeed from "@/components/AnomalyFeed";
import ConservationBadge from "@/components/ConservationBadge";
import InvestigationModal from "@/components/InvestigationModal";
import { AnomalyTypeChart, SpeciesBarChart } from "@/components/AnomalyCharts";
import { useSentinel } from "@/lib/SentinelContext";
import { useAnimalTracking } from "@/lib/AnimalTrackingContext";
import { useWatchlist } from "@/lib/useWatchlist";
import { getTimeSince } from "@/lib/time";
import { TRACKED_SPECIES } from "@/lib/species-data";
import type { Alert } from "@/lib/types";

const ANOMALY_LABELS: Record<string, string> = {
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

function rangeColorByStatus(status: string): string {
  switch (status) {
    case "critically_endangered":
      return "#ef4444";
    case "endangered":
      return "#f97316";
    case "vulnerable":
      return "#eab308";
    default:
      return "#2dd4bf";
  }
}

function buildActivityChart(alerts: Alert[]): { hour: string; count: number }[] {
  const buckets: Record<string, number> = {};
  for (const a of alerts) {
    const d = new Date(a.timestamp);
    const h = `${d.getHours().toString().padStart(2, "0")}:00`;
    buckets[h] = (buckets[h] || 0) + 1;
  }
  return Object.entries(buckets)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([hour, count]) => ({ hour, count }));
}

function findNearbyAlerts(target: Alert, all: Alert[], maxDistance = 5): Alert[] {
  return all.filter((a) => {
    if (a.alert_id === target.alert_id) return false;
    const dLat = Math.abs(a.location.lat - target.location.lat);
    const dLng = Math.abs(a.location.lng - target.location.lng);
    return dLat < maxDistance && dLng < maxDistance;
  });
}

export default function DashboardPage() {
  const { alerts, isLoading, error, fetchAlerts, species, lastFetched } = useSentinel();
  const { animals } = useAnimalTracking();
  const { isWatched } = useWatchlist();
  const [investigatingAlert, setInvestigatingAlert] = useState<Alert | null>(null);
  const [activeMapMarker, setActiveMapMarker] = useState<string | null>(null);
  const [newAlertIds, setNewAlertIds] = useState<Set<string>>(new Set());
  const prevAlertIdsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const prevIds = prevAlertIdsRef.current;
    const incomingIds = alerts.map((a) => a.alert_id);
    const newIds = new Set<string>();
    for (const id of incomingIds) {
      if (!prevIds.has(id)) newIds.add(id);
    }
    if (newIds.size > 0) {
      setNewAlertIds(newIds);
      setTimeout(() => setNewAlertIds(new Set()), 8000);
    }
    prevAlertIdsRef.current = new Set(incomingIds);
  }, [alerts]);

  const criticalCount = alerts.filter((a) => a.alert_level === "CRITICAL").length;
  const warningCount = alerts.filter((a) => a.alert_level === "WARNING").length;
  const speciesAnalyzed = new Set(alerts.map((a) => a.species_name)).size;

  // Conservation insights
  const insights = useMemo(() => {
    const declining = TRACKED_SPECIES.filter((s) => s.population.trend === "decreasing");
    const critical = TRACKED_SPECIES.filter((s) => s.conservationStatus === "critically_endangered");
    const activeAnimals = animals.filter((a) => a.status === "active").length;
    const totalPop = TRACKED_SPECIES.reduce((sum, s) => sum + s.population.estimated, 0);

    // Threat summary from alerts
    const threatCounts: Record<string, number> = {};
    for (const a of alerts) {
      const label = ANOMALY_LABELS[a.anomaly_type] || a.anomaly_type;
      threatCounts[label] = (threatCounts[label] || 0) + 1;
    }
    const topThreats = Object.entries(threatCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3);

    // Regions from alerts
    const regions = new Set<string>();
    for (const a of alerts) {
      if (a.place_name) regions.add(a.place_name.split(",")[0].trim());
    }

    return { declining, critical, activeAnimals, totalPop, topThreats, regionCount: regions.size };
  }, [alerts, animals]);

  const markers = alerts.map((a) => ({
    id: a.alert_id,
    lat: a.location.lat,
    lng: a.location.lng,
    label: a.common_name,
    type: (a.alert_level === "CRITICAL"
      ? "critical"
      : a.alert_level === "WARNING"
      ? "anomaly"
      : "normal") as "normal" | "anomaly" | "critical",
    species: a.common_name,
    details: ANOMALY_LABELS[a.anomaly_type] || a.anomaly_type,
  }));

  const ranges = species.map((sp) => ({
    center: [sp.range.center.lat, sp.range.center.lng] as [number, number],
    radiusDeg: sp.range.radiusDeg,
    label: sp.commonName,
    color: rangeColorByStatus(sp.conservationStatus),
  }));

  const chartData = buildActivityChart(alerts);

  const watchedSpecies = species.filter((sp) => isWatched(sp.id));

  const handleMarkerClick = (markerId: string) => {
    setActiveMapMarker(markerId);
    const alert = alerts.find((a) => a.alert_id === markerId);
    if (alert) setInvestigatingAlert(alert);
  };

  const handleAlertClick = (alert: Alert) => {
    setActiveMapMarker(alert.alert_id);
    setInvestigatingAlert(alert);
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-white">
      <Navbar />

      <div className="max-w-7xl mx-auto px-4 md:px-6 pt-24 pb-12">
        {/* Header */}
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white">Wildlife Activity</h1>
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 live-indicator" />
              <span className="text-[10px] text-emerald-400 font-medium">LIVE</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {lastFetched && (
              <span className="text-xs text-zinc-500 hidden sm:inline">
                <Clock className="w-3 h-3 inline mr-1 -mt-0.5" />
                {getTimeSince(lastFetched)} ago
              </span>
            )}
            <button
              onClick={() => fetchAlerts()}
              disabled={isLoading}
              className="p-2 rounded-lg border border-white/[0.06] hover:border-radar-green/30 hover:bg-radar-green/5 transition-all disabled:opacity-50"
              aria-label="Refresh"
            >
              <RefreshCw
                className={`w-4 h-4 text-zinc-400 ${isLoading ? "animate-spin" : ""}`}
              />
            </button>
          </div>
        </div>
        <p className="text-sm text-zinc-400 mb-6">
          Monitoring {speciesAnalyzed} species across {alerts.length} observations
          <span className="text-zinc-600 ml-2">
            <Radio className="w-3 h-3 inline -mt-0.5" /> Refreshing every 60s
          </span>
        </p>

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center gap-3"
            >
              <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
              <p className="text-sm text-red-400 flex-1">{error}</p>
              <button
                onClick={() => fetchAlerts()}
                className="text-xs text-red-400 hover:text-red-300 underline"
              >
                Retry
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Loading skeleton */}
        {isLoading && alerts.length === 0 && (
          <div className="space-y-6 mb-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="radar-card rounded-xl p-5">
                  <div className="h-3 w-20 skeleton rounded mb-3" />
                  <div className="h-8 w-16 skeleton rounded" style={{ animationDelay: `${i * 100}ms` }} />
                </div>
              ))}
            </div>
            <div className="radar-card rounded-xl h-[500px] skeleton" />
          </div>
        )}

        {alerts.length > 0 && (
          <>
            {/* ── Stat cards ── */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              {[
                { icon: Eye, label: "Observations", value: alerts.length, accent: "teal" as const },
                { icon: AlertTriangle, label: "Warnings", value: warningCount, accent: "amber" as const },
                { icon: Shield, label: "Critical Alerts", value: criticalCount, accent: "red" as const },
                { icon: Activity, label: "Detection Rate", value: `${(criticalCount / alerts.length * 100).toFixed(1)}%`, accent: "teal" as const },
              ].map((card, i) => (
                <motion.div
                  key={card.label}
                  initial={{ opacity: 0, y: 18 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: i * 0.08 }}
                >
                  <StatCard {...card} />
                </motion.div>
              ))}
            </div>
          </>
        )}

        {/* ── Quick navigation row (always visible) ── */}
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35, duration: 0.5 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6"
        >
          {[
            { href: "/species", icon: PawPrint, label: "Species Directory", count: `${TRACKED_SPECIES.length} species`, color: "text-emerald-400" },
            { href: "/animals", icon: Locate, label: "Animal Tracking", count: `${animals.length} individuals`, color: "text-cyan-400" },
            { href: "/alerts", icon: AlertTriangle, label: "All Alerts", count: `${alerts.length} total`, color: "text-amber-400" },
            { href: "/alerts?level=CRITICAL", icon: Zap, label: "Critical Only", count: `${criticalCount} critical`, color: "text-red-400" },
          ].map((nav) => (
            <Link
              key={nav.href}
              href={nav.href}
              className="radar-card rounded-lg px-4 py-3 flex items-center gap-3 hover:bg-white/[0.04] transition-all group"
            >
              <nav.icon className={`w-4 h-4 ${nav.color} shrink-0`} />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-zinc-300 group-hover:text-white transition-colors truncate">
                  {nav.label}
                </p>
                <p className="text-[10px] text-zinc-600">{nav.count}</p>
              </div>
              <ArrowRight className="w-3 h-3 text-zinc-700 group-hover:text-radar-green group-hover:translate-x-0.5 transition-all shrink-0" />
            </Link>
          ))}
        </motion.div>

        {/* ── Main content grid (always visible) ── */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="grid lg:grid-cols-3 gap-6"
        >
          {/* Left — map + charts */}
          <div className="lg:col-span-2 space-y-6">
            {/* Map */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.35 }}
              className="radar-card rounded-xl overflow-hidden"
            >
              <WildlifeMap
                markers={markers}
                ranges={ranges}
                center={[20, 0]}
                zoom={2}
                height="500px"
                onMarkerClick={handleMarkerClick}
                activeMarkerId={activeMapMarker}
              />
            </motion.div>

            {/* Conservation Insights Panel */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.5 }}
              className="radar-card rounded-xl p-5"
            >
              <div className="flex items-center gap-2 mb-4">
                <Globe2 className="w-4 h-4 text-radar-green" />
                <h3 className="text-sm font-medium text-zinc-300">Conservation Insights</h3>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
                <div className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                  <p className="text-lg font-bold text-white">{insights.critical.length}</p>
                  <p className="text-[10px] text-red-400 uppercase tracking-wide">Critically Endangered</p>
                </div>
                <div className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                  <p className="text-lg font-bold text-white">{insights.declining.length}</p>
                  <p className="text-[10px] text-amber-400 uppercase tracking-wide">Pop. Declining</p>
                </div>
                <div className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                  <p className="text-lg font-bold text-white">{insights.activeAnimals}</p>
                  <p className="text-[10px] text-emerald-400 uppercase tracking-wide">GPS Active</p>
                </div>
                <div className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                  <p className="text-lg font-bold text-white">
                    {insights.totalPop > 1000 ? `${Math.round(insights.totalPop / 1000)}K` : insights.totalPop}
                  </p>
                  <p className="text-[10px] text-radar-green uppercase tracking-wide">Total Population</p>
                </div>
              </div>

              {/* Top threats */}
              {insights.topThreats.length > 0 && (
                <div>
                  <p className="text-[10px] text-zinc-500 uppercase tracking-wide mb-2">Top Threat Types</p>
                  <div className="flex flex-wrap gap-2">
                    {insights.topThreats.map(([name, count]) => (
                      <div
                        key={name}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/[0.03] border border-white/[0.06] text-xs text-zinc-400"
                      >
                        <Target className="w-3 h-3 text-amber-400" />
                        {name}
                        <span className="text-zinc-600 font-mono">{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>

            {/* Charts row (only if alerts exist) */}
            {alerts.length > 0 && (
              <div className="grid md:grid-cols-2 gap-6">
                {chartData.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-60px" }}
                    transition={{ duration: 0.5 }}
                    className="radar-card rounded-xl p-5"
                  >
                    <h3 className="text-sm font-medium text-zinc-300 mb-4">Detection Activity</h3>
                    <ResponsiveContainer width="100%" height={200}>
                      <AreaChart data={chartData}>
                        <defs>
                          <linearGradient id="areaFill" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#2dd4bf" stopOpacity={0.2} />
                            <stop offset="100%" stopColor="#2dd4bf" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <XAxis dataKey="hour" tick={{ fill: "#71717a", fontSize: 11 }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fill: "#71717a", fontSize: 11 }} axisLine={false} tickLine={false} allowDecimals={false} />
                        <Tooltip contentStyle={{ background: "#18181b", border: "1px solid rgba(45,212,191,0.2)", borderRadius: 8, color: "#fff", fontSize: 12 }} />
                        <Area type="monotone" dataKey="count" stroke="#2dd4bf" fill="url(#areaFill)" strokeWidth={2} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </motion.div>
                )}

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-60px" }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                  className="radar-card rounded-xl p-5"
                >
                  <h3 className="text-sm font-medium text-zinc-300 mb-4">Anomaly Distribution</h3>
                  <AnomalyTypeChart alerts={alerts} />
                </motion.div>
              </div>
            )}

            {/* Species bar chart (only if alerts exist) */}
            {alerts.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{ duration: 0.5 }}
                className="radar-card rounded-xl p-5"
              >
                <h3 className="text-sm font-medium text-zinc-300 mb-4">Alerts by Species</h3>
                <SpeciesBarChart alerts={alerts} />
              </motion.div>
            )}
          </div>

          {/* Right sidebar (always visible) */}
          <div className="lg:col-span-1 space-y-6">
            {/* Species at risk panel */}
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5, duration: 0.5 }}
              className="radar-card rounded-xl overflow-hidden"
            >
              <div className="px-4 py-3 border-b border-white/[0.06] flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <TrendingDown className="w-3.5 h-3.5 text-red-400" />
                  <h3 className="text-sm font-medium text-zinc-300">Species at Risk</h3>
                </div>
                <Link href="/species" className="text-[10px] text-radar-green hover:underline">
                  View all
                </Link>
              </div>
              <div className="divide-y divide-white/[0.04]">
                {TRACKED_SPECIES.filter((s) => s.conservationStatus === "critically_endangered" || s.population.trend === "decreasing")
                  .slice(0, 6)
                  .map((sp) => {
                    const relatedAlerts = alerts.filter((a) => a.species_name === sp.scientificName);
                    return (
                      <Link
                        key={sp.id}
                        href={`/species/${sp.id}`}
                        className="flex items-center gap-3 px-4 py-2.5 hover:bg-white/[0.03] transition-colors group"
                      >
                        <img
                          src={sp.imageUrl}
                          alt={sp.commonName}
                          className="w-8 h-8 rounded-full object-cover border border-white/[0.08]"
                        />
                        <div className="flex-1 min-w-0">
                          <span className="text-xs font-medium text-white truncate block group-hover:text-radar-green transition-colors">
                            {sp.commonName}
                          </span>
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] text-zinc-600">
                              ~{new Intl.NumberFormat("en-US").format(sp.population.estimated)}
                            </span>
                            {sp.population.trend === "decreasing" ? (
                              <TrendingDown className="w-2.5 h-2.5 text-red-400" />
                            ) : sp.population.trend === "increasing" ? (
                              <TrendingUp className="w-2.5 h-2.5 text-emerald-400" />
                            ) : null}
                            {relatedAlerts.length > 0 && (
                              <span className="text-[10px] text-amber-400 ml-auto">
                                {relatedAlerts.length} alert{relatedAlerts.length > 1 ? "s" : ""}
                              </span>
                            )}
                          </div>
                        </div>
                        <ConservationBadge status={sp.conservationStatus} size="sm" />
                      </Link>
                    );
                  })}
              </div>
            </motion.div>

            {/* Watched species */}
            {watchedSpecies.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6, duration: 0.5 }}
                className="radar-card rounded-xl overflow-hidden"
              >
                <div className="px-4 py-3 border-b border-white/[0.06] flex items-center gap-2">
                  <Heart className="w-3.5 h-3.5 text-red-400 fill-current" />
                  <h3 className="text-sm font-medium text-zinc-300">Watched Species</h3>
                </div>
                <div className="divide-y divide-white/[0.04]">
                  {watchedSpecies.map((sp) => {
                    const relatedAlerts = alerts.filter((a) => a.species_name === sp.scientificName);
                    return (
                      <Link
                        key={sp.id}
                        href={`/species/${sp.id}`}
                        className="flex items-center gap-3 px-4 py-2.5 hover:bg-white/[0.03] transition-colors"
                      >
                        <img
                          src={sp.imageUrl}
                          alt={sp.commonName}
                          className="w-8 h-8 rounded-full object-cover border border-white/[0.08]"
                        />
                        <div className="flex-1 min-w-0">
                          <span className="text-xs font-medium text-white truncate block">{sp.commonName}</span>
                          {relatedAlerts.length > 0 && (
                            <span className="text-[10px] text-amber-400">{relatedAlerts.length} alert{relatedAlerts.length > 1 ? "s" : ""}</span>
                          )}
                        </div>
                        <ConservationBadge status={sp.conservationStatus} size="sm" />
                      </Link>
                    );
                  })}
                </div>
              </motion.div>
            )}

            {/* Active animals panel */}
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.65, duration: 0.5 }}
              className="radar-card rounded-xl overflow-hidden"
            >
              <div className="px-4 py-3 border-b border-white/[0.06] flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Locate className="w-3.5 h-3.5 text-cyan-400" />
                  <h3 className="text-sm font-medium text-zinc-300">GPS Tracking</h3>
                </div>
                <Link href="/animals" className="text-[10px] text-radar-green hover:underline">
                  View all
                </Link>
              </div>
              <div className="px-4 py-3">
                <div className="grid grid-cols-3 gap-3 mb-3">
                  <div className="text-center p-2 rounded-lg bg-white/[0.02]">
                    <p className="text-sm font-bold text-white">{animals.length}</p>
                    <p className="text-[9px] text-zinc-500 uppercase">Total</p>
                  </div>
                  <div className="text-center p-2 rounded-lg bg-white/[0.02]">
                    <p className="text-sm font-bold text-emerald-400">{animals.filter((a) => a.status === "active").length}</p>
                    <p className="text-[9px] text-zinc-500 uppercase">Active</p>
                  </div>
                  <div className="text-center p-2 rounded-lg bg-white/[0.02]">
                    <p className="text-sm font-bold text-zinc-400">{animals.filter((a) => a.status !== "active").length}</p>
                    <p className="text-[9px] text-zinc-500 uppercase">Inactive</p>
                  </div>
                </div>
                {animals.slice(0, 8).map((animal) => (
                  <Link
                    key={animal.id}
                    href={`/animals/${animal.studyId}/${animal.id}`}
                    className="flex items-center gap-2 py-1.5 hover:bg-white/[0.02] rounded px-1 -mx-1 transition-colors group"
                  >
                    <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                      animal.status === "active" ? "bg-emerald-400" : "bg-zinc-600"
                    }`} />
                    <span className="text-xs text-zinc-400 group-hover:text-white transition-colors truncate flex-1">
                      {animal.name || animal.id}
                    </span>
                    <span className="text-[10px] text-zinc-600 shrink-0">
                      {animal.commonName}
                    </span>
                  </Link>
                ))}
                {animals.length > 8 && (
                  <Link
                    href="/animals"
                    className="text-[11px] text-radar-green hover:underline flex items-center gap-1 mt-2"
                  >
                    +{animals.length - 8} more <ArrowRight className="w-3 h-3" />
                  </Link>
                )}
              </div>
            </motion.div>

            {/* Recent alerts */}
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7, duration: 0.5 }}
              className="radar-card rounded-xl overflow-hidden"
            >
              <div className="px-4 py-3 border-b border-white/[0.06] flex items-center justify-between">
                <h3 className="text-sm font-medium text-zinc-300">Recent Alerts</h3>
                <span className="text-[10px] font-mono text-zinc-500 bg-white/[0.04] px-2 py-0.5 rounded-full">
                  {alerts.length}
                </span>
              </div>
              <div className="p-3 max-h-[600px] overflow-y-auto">
                <AnomalyFeed
                  alerts={alerts}
                  maxItems={15}
                  onAlertClick={handleAlertClick}
                  highlightIds={newAlertIds}
                />
              </div>
            </motion.div>
          </div>
        </motion.div>
      </div>

      <InvestigationModal
        alert={investigatingAlert}
        onClose={() => {
          setInvestigatingAlert(null);
          setActiveMapMarker(null);
        }}
        nearbyAlerts={investigatingAlert ? findNearbyAlerts(investigatingAlert, alerts) : []}
      />
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  accent,
}: {
  icon: typeof Activity;
  label: string;
  value: number | string;
  accent: "teal" | "amber" | "red";
}) {
  const colors: Record<string, { icon: string; iconBg: string }> = {
    teal: { icon: "text-radar-green", iconBg: "bg-radar-green/10 border-radar-green/20" },
    amber: { icon: "text-amber-400", iconBg: "bg-amber-500/10 border-amber-500/20" },
    red: { icon: "text-red-400", iconBg: "bg-red-500/10 border-red-500/20" },
  };
  const c = colors[accent] || colors.teal;

  return (
    <div className="radar-card rounded-xl p-5">
      <div className="flex items-center gap-2 mb-2">
        <div className={`w-7 h-7 rounded-md ${c.iconBg} border flex items-center justify-center`}>
          <Icon className={`w-3.5 h-3.5 ${c.icon}`} />
        </div>
      </div>
      <p className="text-xs text-zinc-500 uppercase tracking-wide mb-1">{label}</p>
      <p className="text-2xl font-bold text-white">{value}</p>
    </div>
  );
}
