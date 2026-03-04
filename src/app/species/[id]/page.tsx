"use client";

import { useState, useMemo } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  ArrowRight,
  MapPin,
  Users,
  Clock,
  Utensils,
  Scale,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  Info,
  Heart,
} from "lucide-react";

import Navbar from "@/components/Navbar";
import ConservationBadge from "@/components/ConservationBadge";
import WildlifeMap from "@/components/WildlifeMap";
import SpeciesCard from "@/components/SpeciesCard";
import PopulationChart from "@/components/PopulationChart";
import AnomalyFeed from "@/components/AnomalyFeed";
import InvestigationModal from "@/components/InvestigationModal";
import { getSpeciesById, TRACKED_SPECIES } from "@/lib/species-data";
import { useSentinel } from "@/lib/SentinelContext";
import { useWatchlist } from "@/lib/useWatchlist";
import type { Alert } from "@/lib/types";

function formatPopulation(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`.replace(".0K", "K");
  return new Intl.NumberFormat("en-US").format(n);
}

function zoomFromRadius(radiusDeg: number): number {
  if (radiusDeg <= 3) return 6;
  if (radiusDeg <= 5) return 5;
  if (radiusDeg <= 10) return 4;
  if (radiusDeg <= 20) return 3;
  return 2;
}

const trendConfig = {
  increasing: { icon: TrendingUp, color: "text-emerald-400", label: "Increasing" },
  decreasing: { icon: TrendingDown, color: "text-red-400", label: "Decreasing" },
  stable: { icon: Minus, color: "text-yellow-400", label: "Stable" },
  unknown: { icon: Minus, color: "text-zinc-500", label: "Unknown" },
} as const;

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

export default function SpeciesDetailPage() {
  const params = useParams();
  const species = getSpeciesById(params.id as string);
  const { alerts, isLoading: alertsLoading } = useSentinel();
  const { isWatched, toggle } = useWatchlist();
  const [investigatingAlert, setInvestigatingAlert] = useState<Alert | null>(null);

  const speciesAlerts = useMemo(() => {
    if (!species) return [];
    return alerts.filter((a) => a.species_name === species.scientificName);
  }, [alerts, species]);

  const alertMarkers = useMemo(() => {
    return speciesAlerts.map((a) => ({
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
  }, [speciesAlerts]);

  const relatedSpecies = useMemo(
    () => TRACKED_SPECIES.filter((s) => s.id !== params.id).slice(0, 3),
    [params.id],
  );

  if (!species) {
    return (
      <div className="min-h-screen bg-[#09090b] text-white">
        <Navbar />
        <div className="flex flex-col items-center justify-center pt-40 gap-4">
          <h1 className="text-2xl font-bold">Species not found</h1>
          <p className="text-zinc-400">
            The species you&apos;re looking for doesn&apos;t exist in our database.
          </p>
          <Link href="/species" className="inline-flex items-center gap-2 text-radar-green hover:underline text-sm mt-2">
            <ArrowLeft className="w-4 h-4" />
            Back to Species
          </Link>
        </div>
      </div>
    );
  }

  const trend = trendConfig[species.population.trend];
  const TrendIcon = trend.icon;
  const watched = isWatched(species.id);

  return (
    <div className="min-h-screen bg-[#09090b] text-white">
      <Navbar />

      <div className="max-w-6xl mx-auto px-6 pt-24 pb-16 space-y-10">
        <Link href="/species" className="inline-flex items-center gap-2 text-zinc-400 hover:text-white transition-colors text-sm">
          <ArrowLeft className="w-4 h-4" />
          Back to Species
        </Link>

        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="grid md:grid-cols-2 gap-8 items-start"
        >
          <div className="space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h1 className="text-4xl font-bold">{species.commonName}</h1>
                <p className="text-lg text-zinc-400 italic mt-1">{species.scientificName}</p>
              </div>
              <button
                onClick={() => toggle(species.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  watched
                    ? "bg-red-500/15 text-red-400 border border-red-500/30 hover:bg-red-500/25"
                    : "bg-white/[0.05] text-zinc-400 border border-white/[0.1] hover:text-white hover:border-white/[0.2]"
                }`}
              >
                <Heart className={`w-4 h-4 ${watched ? "fill-current" : ""}`} />
                {watched ? "Following" : "Follow"}
              </button>
            </div>
            <ConservationBadge status={species.conservationStatus} size="lg" />
            <p className="text-zinc-300 leading-relaxed mt-4">{species.habitat}</p>
          </div>

          <div className="relative rounded-xl overflow-hidden aspect-video">
            <img src={species.imageUrl} alt={species.commonName} className="w-full h-full object-cover rounded-xl" />
            <div className="absolute inset-0 bg-gradient-to-t from-[#09090b]/60 via-transparent to-transparent pointer-events-none" />
          </div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4"
        >
          <div className="radar-card rounded-xl p-4 space-y-2">
            <div className="flex items-center gap-2">
              <Users className="w-4 h-4 text-zinc-500" />
              <span className="text-xs text-zinc-500 uppercase">Population</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-white font-semibold text-lg">{formatPopulation(species.population.estimated)}</span>
              <TrendIcon className={`w-4 h-4 ${trend.color}`} />
            </div>
            <span className={`text-xs ${trend.color}`}>{trend.label}</span>
          </div>

          <div className="radar-card rounded-xl p-4 space-y-2">
            <div className="flex items-center gap-2">
              <Scale className="w-4 h-4 text-zinc-500" />
              <span className="text-xs text-zinc-500 uppercase">Weight</span>
            </div>
            <span className="text-white font-semibold">{species.weight}</span>
          </div>

          <div className="radar-card rounded-xl p-4 space-y-2">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-zinc-500" />
              <span className="text-xs text-zinc-500 uppercase">Lifespan</span>
            </div>
            <span className="text-white font-semibold">{species.lifespan}</span>
          </div>

          <div className="radar-card rounded-xl p-4 space-y-2">
            <div className="flex items-center gap-2">
              <Utensils className="w-4 h-4 text-zinc-500" />
              <span className="text-xs text-zinc-500 uppercase">Diet</span>
            </div>
            <span className="text-white font-semibold text-sm">{species.diet}</span>
          </div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.15 }}
        >
          <div className="radar-card rounded-xl p-5">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-radar-green" />
              Population Trend
            </h2>
            <PopulationChart currentPopulation={species.population.estimated} trend={species.population.trend} speciesName={species.id} />
          </div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.2 }}
          className="grid lg:grid-cols-2 gap-8"
        >
          <div className="space-y-8">
            <div>
              <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <MapPin className="w-4 h-4 text-radar-green" />
                Range &amp; Distribution
              </h2>
              <p className="text-zinc-300 text-sm leading-relaxed mb-3">{species.range.description}</p>
              <div className="flex flex-wrap gap-1.5">
                {species.range.regions.map((region) => (
                  <span key={region} className="text-xs px-2 py-1 rounded-md bg-white/5 text-zinc-400 border border-white/[0.06]">
                    {region}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400" />
                Threats
              </h2>
              <ol className="space-y-2">
                {species.threats.map((threat, i) => (
                  <li key={i} className="flex items-start gap-2.5 text-sm text-zinc-300">
                    <span className="flex-shrink-0 mt-0.5 w-5 h-5 rounded bg-red-500/10 border border-red-500/20 flex items-center justify-center text-[10px] text-red-400 font-mono font-bold">
                      {i + 1}
                    </span>
                    {threat}
                  </li>
                ))}
              </ol>
            </div>

            <div>
              <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Info className="w-4 h-4 text-radar-green" />
                Facts
              </h2>
              <ul className="space-y-2">
                {species.facts.map((fact, i) => (
                  <li key={i} className="flex items-start gap-2.5 text-sm text-zinc-300 leading-relaxed">
                    <Info className="w-4 h-4 text-zinc-600 flex-shrink-0 mt-0.5" />
                    {fact}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div>
            <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <MapPin className="w-4 h-4 text-radar-green" />
              Range Map
              {alertMarkers.length > 0 && (
                <span className="text-xs text-zinc-500 font-normal">({alertMarkers.length} observation{alertMarkers.length > 1 ? "s" : ""})</span>
              )}
            </h2>
            <WildlifeMap
              ranges={[{ center: [species.range.center.lat, species.range.center.lng], radiusDeg: species.range.radiusDeg, label: `${species.commonName} range` }]}
              markers={alertMarkers}
              center={[species.range.center.lat, species.range.center.lng]}
              zoom={zoomFromRadius(species.range.radiusDeg)}
              height="400px"
              className="rounded-xl overflow-hidden border border-white/[0.06]"
              onMarkerClick={(id) => {
                const alert = speciesAlerts.find((a) => a.alert_id === id);
                if (alert) setInvestigatingAlert(alert);
              }}
            />
          </div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.25 }}
        >
          <div className="radar-card rounded-xl overflow-hidden">
            <div className="px-5 py-3 border-b border-white/[0.06] flex items-center justify-between">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Recent Anomalies
              </h2>
              {speciesAlerts.length > 0 && (
                <Link href={`/alerts?species=${species.id}`} className="text-xs text-radar-green hover:underline flex items-center gap-1">
                  View all <ArrowRight className="w-3 h-3" />
                </Link>
              )}
            </div>
            <div className="p-4">
              {alertsLoading ? (
                <div className="space-y-3">
                  {[...Array(3)].map((_, i) => (
                    <div key={i} className="h-16 skeleton rounded-lg" style={{ animationDelay: `${i * 80}ms` }} />
                  ))}
                </div>
              ) : speciesAlerts.length > 0 ? (
                <AnomalyFeed alerts={speciesAlerts} maxItems={10} onAlertClick={(a) => setInvestigatingAlert(a)} />
              ) : (
                <div className="text-center py-8">
                  <AlertTriangle className="w-8 h-8 text-zinc-700 mx-auto mb-2" />
                  <p className="text-sm text-zinc-500">No anomalies detected for {species.commonName}</p>
                  <p className="text-xs text-zinc-600 mt-1">This species is currently within expected parameters.</p>
                </div>
              )}
            </div>
          </div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.3 }}
        >
          <h2 className="text-lg font-semibold mb-4">Other Tracked Species</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {relatedSpecies.map((s) => (
              <SpeciesCard key={s.id} species={s} isWatched={isWatched(s.id)} onToggleWatch={toggle} />
            ))}
          </div>
        </motion.section>
      </div>

      <InvestigationModal
        alert={investigatingAlert}
        onClose={() => setInvestigatingAlert(null)}
        nearbyAlerts={
          investigatingAlert
            ? alerts.filter(
                (a) =>
                  a.alert_id !== investigatingAlert.alert_id &&
                  Math.abs(a.location.lat - investigatingAlert.location.lat) < 5 &&
                  Math.abs(a.location.lng - investigatingAlert.location.lng) < 5
              )
            : []
        }
      />
    </div>
  );
}
