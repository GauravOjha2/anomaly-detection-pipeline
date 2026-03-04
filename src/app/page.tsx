"use client";

import { useMemo, useRef } from "react";
import Link from "next/link";
import { motion, useScroll, useTransform, useInView } from "framer-motion";
import {
  ArrowRight,
  PawPrint,
  Globe2,
  Shield,
  Satellite,
  ScanSearch,
  ShieldAlert,
  Locate,
  AlertTriangle,
  Activity,
  TrendingDown,
  MapPin,
  Clock,
  Eye,
  Route,
} from "lucide-react";
import Navbar from "@/components/Navbar";
import FloatingScene from "@/components/FloatingScene";
import SpeciesCard from "@/components/SpeciesCard";
import ConservationBadge from "@/components/ConservationBadge";
import { TRACKED_SPECIES } from "@/lib/species-data";
import { useSentinel } from "@/lib/SentinelContext";
import { useAnimalTracking } from "@/lib/AnimalTrackingContext";
import { getRelativeTime } from "@/lib/time";

// ── Animation variants ──────────────────────────────────────
const sectionFade = {
  hidden: { opacity: 0, y: 50 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.9, ease: "easeOut" as const },
  },
};

const staggerContainer = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.12, delayChildren: 0.1 },
  },
};

const staggerItem = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.7, ease: "easeOut" as const },
  },
};

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

// ── Steps data ───────────────────────────────────────────────
const steps = [
  {
    icon: Satellite,
    number: "01",
    title: "Data streams in\nfrom the field",
    description:
      "Observation data from GPS collars, research stations, and field reports across six continents. Updated every 60 seconds.",
    stat: "60s",
    statLabel: "refresh cycle",
  },
  {
    icon: ScanSearch,
    number: "02",
    title: "Patterns analyzed.\nAnomalies surfaced.",
    description:
      "Eight distinct anomaly types — from range deviations and poaching indicators to habitat mismatches and unusual clustering events.",
    stat: "8",
    statLabel: "anomaly types",
  },
  {
    icon: ShieldAlert,
    number: "03",
    title: "Teams alerted.\nAction taken.",
    description:
      "Prioritized alerts with confidence scores, location data, and one-click investigation tools. Three severity tiers: Info, Warning, Critical.",
    stat: "3",
    statLabel: "severity levels",
  },
];

// ── Explore cards ────────────────────────────────────────────
const exploreCards = [
  {
    href: "/dashboard",
    icon: Activity,
    title: "Live Dashboard",
    description:
      "Real-time map, anomaly detection, activity charts, species-level alert feeds.",
    color: "text-radar-green",
    borderColor: "border-radar-green/20 hover:border-radar-green/40",
    glowColor: "rgba(45, 212, 191, 0.06)",
  },
  {
    href: "/species",
    icon: PawPrint,
    title: "Species Directory",
    description:
      "In-depth profiles for 12 tracked species with population trends and conservation status.",
    color: "text-emerald-400",
    borderColor: "border-emerald-500/20 hover:border-emerald-500/40",
    glowColor: "rgba(16, 185, 129, 0.06)",
  },
  {
    href: "/animals",
    icon: Locate,
    title: "Individual Tracking",
    description:
      "GPS telemetry for individually tracked animals with movement maps and status updates.",
    color: "text-cyan-400",
    borderColor: "border-cyan-500/20 hover:border-cyan-500/40",
    glowColor: "rgba(6, 182, 212, 0.06)",
  },
  {
    href: "/alerts",
    icon: AlertTriangle,
    title: "Anomaly Alerts",
    description:
      "Filterable alert feed with severity levels, confidence scores, and investigation modals.",
    color: "text-amber-400",
    borderColor: "border-amber-500/20 hover:border-amber-500/40",
    glowColor: "rgba(245, 158, 11, 0.06)",
  },
];

// ── Step section component (DuperMemory-style) ───────────────
function StepSection({
  step,
  index,
  isLast,
}: {
  step: (typeof steps)[0];
  index: number;
  isLast: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: "-20% 0px -20% 0px" });

  return (
    <div ref={ref} className={`relative ${isLast ? "" : "mb-0"}`}>
      {/* Horizontal rule with glowing dot (between steps) */}
      {index === 0 && <hr className="glow-hr mb-16 md:mb-20" />}

      <motion.div
        initial={{ opacity: 0 }}
        animate={isInView ? { opacity: 1 } : { opacity: 0.15 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        className="relative grid md:grid-cols-2 gap-8 md:gap-16 items-center py-16 md:py-24"
      >
        {/* Left — text */}
        <div className="relative z-10">
          <motion.p
            initial={{ opacity: 0, x: -20 }}
            animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: -20 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-[11px] font-mono uppercase tracking-[0.2em] text-radar-green/60 mb-4"
          >
            Step {step.number}
          </motion.p>
          <motion.h3
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
            transition={{ duration: 0.7, delay: 0.15 }}
            className="text-3xl md:text-4xl font-light leading-[1.2] text-white whitespace-pre-line mb-6"
            style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
          >
            {step.title}
          </motion.h3>
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={isInView ? { opacity: 0.7, y: 0 } : { opacity: 0, y: 16 }}
            transition={{ duration: 0.6, delay: 0.25 }}
            className="text-sm text-zinc-400 leading-relaxed max-w-md"
          >
            {step.description}
          </motion.p>

          {/* Stat pill */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 10 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="mt-6 inline-flex items-center gap-3"
          >
            <span className="text-2xl font-bold text-white">{step.stat}</span>
            <span className="text-[10px] text-zinc-500 uppercase tracking-wide">
              {step.statLabel}
            </span>
          </motion.div>
        </div>

        {/* Right — big number */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={
            isInView
              ? { opacity: 1, scale: 1 }
              : { opacity: 0, scale: 0.9 }
          }
          transition={{ duration: 0.8, delay: 0.1 }}
          className="flex items-center justify-center md:justify-end"
        >
          <span className={`step-number ${isInView ? "step-number-glow" : ""}`}>{step.number}</span>
        </motion.div>
      </motion.div>

      {!isLast && <hr className="glow-hr" />}
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────
export default function HomePage() {
  const featuredSpecies = TRACKED_SPECIES.slice(0, 6);
  const { getStats, alerts } = useSentinel();
  const { animals } = useAnimalTracking();

  const stats = getStats();
  const heroRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"],
  });
  const heroOpacity = useTransform(scrollYProgress, [0, 0.8], [1, 0]);
  const heroY = useTransform(scrollYProgress, [0, 0.8], [0, -60]);

  // Dynamic hero stats
  const heroStats = useMemo(() => {
    const regions = new Set<string>();
    for (const s of TRACKED_SPECIES) {
      for (const r of s.range.regions) {
        regions.add(r);
      }
    }
    return [
      { icon: PawPrint, value: stats.speciesCount, label: "Species Tracked" },
      { icon: Locate, value: animals.length, label: "Individuals Monitored" },
      { icon: AlertTriangle, value: stats.totalAlerts, label: "Active Alerts" },
      { icon: Globe2, value: regions.size, label: "Regions Covered" },
    ];
  }, [stats, animals]);

  // Conservation impact stats
  const conservationStats = useMemo(() => {
    const criticalSpecies = TRACKED_SPECIES.filter(
      (s) => s.conservationStatus === "critically_endangered"
    );
    const totalPop = TRACKED_SPECIES.reduce(
      (sum, s) => sum + s.population.estimated,
      0
    );
    const decreasingCount = TRACKED_SPECIES.filter(
      (s) => s.population.trend === "decreasing"
    ).length;
    const activeAnimals = animals.filter((a) => a.status === "active").length;

    return { criticalSpecies, totalPop, decreasingCount, activeAnimals };
  }, [animals]);

  // Recent alerts for live feed
  const recentAlerts = useMemo(() => alerts.slice(0, 5), [alerts]);

  return (
    <main className="relative min-h-screen overflow-x-hidden">
      <Navbar />
      <FloatingScene />

      {/* ===== HERO ===== */}
      <section ref={heroRef} className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-20">
        <motion.div
          style={{ opacity: heroOpacity, y: heroY }}
          className="max-w-4xl mx-auto text-center relative z-10"
        >
          {/* Live badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-radar-green/[0.06] border border-radar-green/15 text-radar-green text-[11px] font-medium uppercase tracking-[0.15em] mb-8"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-radar-green live-indicator" />
            Live Monitoring Active
          </motion.div>

          {/* Main heading — serif, cinematic */}
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
            className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-light leading-[1.05] tracking-tight mb-8"
            style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
          >
            Every species.
            <br />
            <span className="text-radar-green">Monitored.</span>
            <br />
            Protected.
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="text-base md:text-lg text-zinc-500 max-w-xl mx-auto mb-12 leading-relaxed"
          >
            Real-time anomaly detection across endangered species habitats.
            Track individual animals via GPS telemetry. Get instant alerts
            when something looks wrong.
          </motion.p>

          {/* CTA buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.5 }}
            className="flex items-center justify-center gap-4 flex-wrap"
          >
            <Link
              href="/dashboard"
              className="group inline-flex items-center gap-2 px-7 py-3.5 bg-radar-green text-[#09090b] text-sm font-semibold rounded-lg hover:bg-radar-green/90 transition-all hover:shadow-lg hover:shadow-radar-green/20"
            >
              Open Dashboard
              <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </Link>
            <Link
              href="#how-it-works"
              className="inline-flex items-center gap-2 px-7 py-3.5 text-zinc-400 text-sm font-medium rounded-lg border border-zinc-800 hover:border-zinc-600 hover:text-white transition-all"
            >
              See how it works
              <ArrowRight className="w-4 h-4 rotate-90" />
            </Link>
          </motion.div>

          {/* Stats row — below CTA */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.9, duration: 0.8 }}
            className="mt-16 md:mt-20"
          >
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-3xl mx-auto">
              {heroStats.map((stat, i) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 1 + i * 0.1, duration: 0.5 }}
                  className="flex flex-col items-center gap-1.5"
                >
                  <stat.icon className="w-3.5 h-3.5 text-radar-green/50" />
                  <span className="text-xl font-semibold text-white">
                    {stat.value}
                  </span>
                  <span className="text-[10px] text-zinc-600 uppercase tracking-wider">
                    {stat.label}
                  </span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      </section>

      {/* ===== CONSERVATION AT A GLANCE ===== */}
      <motion.section
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={sectionFade}
        className="relative py-24 md:py-32 px-6"
      >
        {/* Aura glow behind section */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] bg-radar-green/[0.03] rounded-full blur-[120px] pointer-events-none" />

        <div className="max-w-6xl mx-auto relative z-10">
          <div className="mb-12">
            <p className="text-[11px] font-mono uppercase tracking-[0.2em] text-radar-green/50 mb-3">
              Conservation Overview
            </p>
            <h2
              className="text-3xl md:text-4xl font-light text-white"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
            >
              Conservation at a Glance
            </h2>
          </div>

          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid lg:grid-cols-5 gap-6"
          >
            {/* Left — key numbers */}
            <div className="lg:col-span-2 grid grid-cols-2 gap-4">
              {[
                {
                  icon: Shield,
                  label: "Critically Endangered",
                  value: conservationStats.criticalSpecies.length,
                  sub: "species monitored",
                  color: "text-red-400",
                  bg: "bg-red-500/10 border-red-500/20",
                },
                {
                  icon: TrendingDown,
                  label: "Population Declining",
                  value: conservationStats.decreasingCount,
                  sub: `of ${TRACKED_SPECIES.length} species`,
                  color: "text-amber-400",
                  bg: "bg-amber-500/10 border-amber-500/20",
                },
                {
                  icon: Route,
                  label: "GPS Active",
                  value: conservationStats.activeAnimals,
                  sub: "animals transmitting",
                  color: "text-emerald-400",
                  bg: "bg-emerald-500/10 border-emerald-500/20",
                },
                {
                  icon: Eye,
                  label: "Total Population",
                  value:
                    conservationStats.totalPop > 1000
                      ? `${Math.round(conservationStats.totalPop / 1000)}K`
                      : conservationStats.totalPop,
                  sub: "individuals estimated",
                  color: "text-radar-green",
                  bg: "bg-radar-green/10 border-radar-green/20",
                },
              ].map((card) => (
                <motion.div
                  key={card.label}
                  variants={staggerItem}
                  className="radar-card rounded-xl p-4 group hover:bg-white/[0.03] transition-all duration-300"
                >
                  <div
                    className={`w-8 h-8 rounded-lg ${card.bg} border flex items-center justify-center mb-3`}
                  >
                    <card.icon className={`w-4 h-4 ${card.color}`} />
                  </div>
                  <p className="text-2xl font-bold text-white">{card.value}</p>
                  <p className="text-[10px] text-zinc-500 uppercase tracking-wide mt-0.5">
                    {card.sub}
                  </p>
                </motion.div>
              ))}
            </div>

            {/* Right — live alert feed */}
            <motion.div
              variants={staggerItem}
              className="lg:col-span-3 radar-card rounded-xl overflow-hidden"
            >
              <div className="px-4 py-3 border-b border-white/[0.06] flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="w-3.5 h-3.5 text-radar-green" />
                  <h3 className="text-sm font-medium text-zinc-300">
                    Recent Activity
                  </h3>
                </div>
                <Link
                  href="/alerts"
                  className="text-[11px] text-radar-green hover:underline flex items-center gap-1"
                >
                  View all <ArrowRight className="w-3 h-3" />
                </Link>
              </div>

              {recentAlerts.length === 0 ? (
                <div className="px-4 py-10 text-center text-zinc-600 text-sm">
                  Monitoring active — no recent alerts
                </div>
              ) : (
                <div className="divide-y divide-white/[0.04]">
                  {recentAlerts.map((alert, i) => (
                    <motion.div
                      key={alert.alert_id}
                      initial={{ opacity: 0, x: -10 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: 0.1 + i * 0.06 }}
                      className="flex items-center gap-3 px-4 py-3 hover:bg-white/[0.02] transition-colors"
                    >
                      <div
                        className={`w-2 h-2 rounded-full shrink-0 ${
                          alert.alert_level === "CRITICAL"
                            ? "bg-red-400"
                            : alert.alert_level === "WARNING"
                            ? "bg-amber-400"
                            : "bg-teal-400"
                        }`}
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-white truncate">
                          {alert.common_name}
                          <span className="text-zinc-500 ml-2 text-xs">
                            {ANOMALY_LABELS[alert.anomaly_type] ||
                              alert.anomaly_type}
                          </span>
                        </p>
                        <p className="text-[11px] text-zinc-600 flex items-center gap-2">
                          {alert.place_name && (
                            <span className="flex items-center gap-0.5">
                              <MapPin className="w-2.5 h-2.5" />
                              {alert.place_name}
                            </span>
                          )}
                          <span className="flex items-center gap-0.5">
                            <Clock className="w-2.5 h-2.5" />
                            {getRelativeTime(alert.timestamp)}
                          </span>
                        </p>
                      </div>
                      <span
                        className={`text-[9px] px-1.5 py-0.5 rounded font-mono font-semibold shrink-0 ${
                          alert.alert_level === "CRITICAL"
                            ? "bg-red-500/15 text-red-400"
                            : alert.alert_level === "WARNING"
                            ? "bg-amber-500/15 text-amber-400"
                            : "bg-teal-500/15 text-teal-400"
                        }`}
                      >
                        {alert.alert_level}
                      </span>
                    </motion.div>
                  ))}
                </div>
              )}
            </motion.div>
          </motion.div>
        </div>
      </motion.section>

      {/* ===== HOW IT WORKS (DuperMemory style) ===== */}
      <section id="how-it-works" className="relative py-24 md:py-32 px-6">
        {/* Background aura */}
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-emerald-500/[0.02] rounded-full blur-[150px] pointer-events-none" />

        <div className="max-w-5xl mx-auto relative z-10">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={sectionFade}
            className="mb-6"
          >
            <p className="text-[11px] font-mono uppercase tracking-[0.2em] text-radar-green/50 mb-3">
              How It Works
            </p>
            <h2
              className="text-4xl md:text-5xl font-light text-white leading-[1.1]"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
            >
              Three steps.
              <br />
              <span className="text-zinc-500">No friction.</span>
            </h2>
          </motion.div>

          {/* Steps */}
          <div className="mt-12">
            {steps.map((step, i) => (
              <StepSection
                key={step.number}
                step={step}
                index={i}
                isLast={i === steps.length - 1}
              />
            ))}
          </div>
        </div>
      </section>

      {/* ===== TRACKED SPECIES ===== */}
      <motion.section
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={sectionFade}
        className="relative py-24 md:py-32 px-6"
      >
        {/* Aura */}
        <div className="absolute bottom-0 left-1/4 w-[500px] h-[400px] bg-teal-500/[0.02] rounded-full blur-[120px] pointer-events-none" />

        <div className="max-w-6xl mx-auto relative z-10">
          <div className="flex items-end justify-between gap-4 mb-12">
            <div>
              <p className="text-[11px] font-mono uppercase tracking-[0.2em] text-radar-green/50 mb-3">
                Species Directory
              </p>
              <h2
                className="text-3xl md:text-4xl font-light text-white"
                style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
              >
                Tracked Species
              </h2>
              <p className="text-sm text-zinc-500 mt-2">
                Monitoring endangered and vulnerable species across critical habitats
              </p>
            </div>
            <Link
              href="/species"
              className="text-radar-green text-sm hover:underline inline-flex items-center gap-1 shrink-0"
            >
              All {TRACKED_SPECIES.length} species
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-50px" }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {featuredSpecies.map((species) => (
              <motion.div key={species.id} variants={staggerItem}>
                <SpeciesCard species={species} />
              </motion.div>
            ))}
          </motion.div>
        </div>
      </motion.section>

      {/* ===== CONSERVATION STATUS BREAKDOWN ===== */}
      <motion.section
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={sectionFade}
        className="relative py-24 md:py-32 px-6"
      >
        <div className="max-w-6xl mx-auto relative z-10">
          <div className="mb-12">
            <p className="text-[11px] font-mono uppercase tracking-[0.2em] text-radar-green/50 mb-3">
              IUCN Red List
            </p>
            <h2
              className="text-3xl md:text-4xl font-light text-white"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
            >
              Conservation Status Overview
            </h2>
          </div>

          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
          >
            {TRACKED_SPECIES.map((sp) => (
              <motion.div key={sp.id} variants={staggerItem}>
                <Link
                  href={`/species/${sp.id}`}
                  className="radar-card rounded-xl p-4 flex items-center gap-4 hover:bg-white/[0.04] transition-all group"
                >
                  <img
                    src={sp.imageUrl}
                    alt={sp.commonName}
                    className="w-12 h-12 rounded-lg object-cover border border-white/[0.08] shrink-0"
                  />
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-medium text-white group-hover:text-radar-green transition-colors truncate">
                      {sp.commonName}
                    </h3>
                    <p className="text-[11px] text-zinc-500 italic truncate">
                      {sp.scientificName}
                    </p>
                    <p className="text-[11px] text-zinc-600 mt-0.5">
                      Pop: ~{new Intl.NumberFormat("en-US").format(sp.population.estimated)} ·{" "}
                      {sp.population.trend === "increasing" ? (
                        <span className="text-emerald-400">Increasing</span>
                      ) : sp.population.trend === "decreasing" ? (
                        <span className="text-red-400">Declining</span>
                      ) : (
                        <span className="text-zinc-400">Stable</span>
                      )}
                    </p>
                  </div>
                  <ConservationBadge
                    status={sp.conservationStatus}
                    size="sm"
                  />
                </Link>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </motion.section>

      {/* ===== EXPLORE THE PLATFORM ===== */}
      <section className="relative py-24 md:py-32 px-6">
        {/* Aura */}
        <div className="absolute top-1/3 right-0 w-[400px] h-[400px] bg-radar-green/[0.02] rounded-full blur-[100px] pointer-events-none" />

        <div className="max-w-5xl mx-auto relative z-10">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={sectionFade}
            className="text-center mb-14"
          >
            <p className="text-[11px] font-mono uppercase tracking-[0.2em] text-radar-green/50 mb-3">
              Platform
            </p>
            <h2
              className="text-3xl md:text-4xl font-light text-white mb-4"
              style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
            >
              Explore the Platform
            </h2>
            <p className="text-zinc-500 text-sm max-w-md mx-auto">
              Every page is built with real data, interactive maps, and
              deep cross-linking between species, animals, and alerts.
            </p>
          </motion.div>

          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-50px" }}
            className="grid grid-cols-1 md:grid-cols-2 gap-5"
          >
            {exploreCards.map((card) => (
              <motion.div key={card.href} variants={staggerItem}>
                <Link
                  href={card.href}
                  className={`relative radar-card rounded-xl p-6 flex items-start gap-4 border ${card.borderColor} transition-all duration-300 group overflow-hidden`}
                >
                  {/* Hover glow */}
                  <div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
                    style={{
                      background: `radial-gradient(ellipse at 30% 50%, ${card.glowColor}, transparent 70%)`,
                    }}
                  />

                  <div className="relative z-10 flex items-start gap-4 w-full">
                    <div className="w-10 h-10 rounded-lg bg-white/[0.03] border border-white/[0.06] flex items-center justify-center shrink-0">
                      <card.icon className={`w-5 h-5 ${card.color}`} />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-medium text-white group-hover:text-radar-green transition-colors mb-1.5">
                        {card.title}
                      </h3>
                      <p className="text-sm text-zinc-500 leading-relaxed">
                        {card.description}
                      </p>
                    </div>
                    <ArrowRight className="w-4 h-4 text-zinc-700 group-hover:text-radar-green group-hover:translate-x-1 transition-all shrink-0 mt-1" />
                  </div>
                </Link>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ===== FOOTER ===== */}
      <footer className="relative border-t border-zinc-800/30 py-16 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded-md bg-radar-greenDim flex items-center justify-center border border-radar-green/20">
                <Shield className="w-3.5 h-3.5 text-radar-green" />
              </div>
              <div>
                <span
                  className="text-base font-light text-white"
                  style={{ fontFamily: "Georgia, 'Times New Roman', serif" }}
                >
                  sentinel
                </span>
                <span className="text-xs text-zinc-600 ml-2">
                  Wildlife Conservation Monitoring
                </span>
              </div>
            </div>

            <div className="flex items-center gap-8">
              {[
                { href: "/dashboard", label: "Dashboard" },
                { href: "/species", label: "Species" },
                { href: "/animals", label: "Animals" },
                { href: "/alerts", label: "Alerts" },
              ].map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="text-sm text-zinc-500 hover:text-white transition-colors"
                >
                  {link.label}
                </Link>
              ))}
            </div>
          </div>

          <div className="mt-10 text-center">
            <p className="text-[11px] text-zinc-700">
              &copy; {new Date().getFullYear()} Sentinel. Open source
              conservation technology.
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}
