"use client";

import { useState, useMemo } from "react";
import { useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Search } from "lucide-react";
import Navbar from "@/components/Navbar";
import SpeciesCard from "@/components/SpeciesCard";
import { TRACKED_SPECIES } from "@/lib/species-data";
import { useWatchlist } from "@/lib/useWatchlist";
import type { ConservationStatus } from "@/lib/types";

// ── Filter definitions ──────────────────────────────────────

interface FilterOption {
  label: string;
  value: ConservationStatus | "all";
}

const STATUS_FILTERS: FilterOption[] = [
  { label: "All", value: "all" },
  { label: "Critically Endangered", value: "critically_endangered" },
  { label: "Endangered", value: "endangered" },
  { label: "Vulnerable", value: "vulnerable" },
];

const TAXON_FILTERS = [
  { label: "All Taxa", value: "all" },
  { label: "Mammals", value: "Mammalia" },
  { label: "Birds", value: "Aves" },
  { label: "Reptiles", value: "Reptilia" },
];

// ── Page ─────────────────────────────────────────────────────

function SpeciesCatalogPageContent() {
  const searchParams = useSearchParams();
  
  const [statusFilter, setStatusFilter] = useState<ConservationStatus | "all">(
    (searchParams.get("status") as ConservationStatus) || "all"
  );
  const [taxonFilter, setTaxonFilter] = useState(searchParams.get("taxon") || "all");
  const [searchQuery, setSearchQuery] = useState(searchParams.get("q") || "");
  const [showWatchedOnly, setShowWatchedOnly] = useState(searchParams.get("watched") === "true");
  
  const watchlist = useWatchlist();

  const filtered = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    return TRACKED_SPECIES.filter((s) => {
      if (statusFilter !== "all" && s.conservationStatus !== statusFilter) return false;
      if (taxonFilter !== "all" && s.iconicTaxon !== taxonFilter) return false;
      if (showWatchedOnly && !watchlist.isWatched(s.id)) return false;
      if (q) {
        return (
          s.commonName.toLowerCase().includes(q) ||
          s.scientificName.toLowerCase().includes(q) ||
          s.range.regions.some((r) => r.toLowerCase().includes(q)) ||
          s.habitat.toLowerCase().includes(q)
        );
      }
      return true;
    });
  }, [statusFilter, taxonFilter, searchQuery, showWatchedOnly, watchlist]);

  const criticalCount = TRACKED_SPECIES.filter(
    (s) => s.conservationStatus === "critically_endangered",
  ).length;

  const endangeredCount = TRACKED_SPECIES.filter(
    (s) => s.conservationStatus === "endangered",
  ).length;

  return (
    <div className="min-h-screen bg-[#09090b] text-white">
      <Navbar />

      {/* ── Header ──────────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-6 pt-24 pb-4">
        <motion.h1
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="text-3xl font-bold"
        >
          Species Directory
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.06 }}
          className="text-zinc-400 mt-2"
        >
          Track and monitor endangered species across global habitats
        </motion.p>
      </section>

      {/* ── Search bar ──────────────────────────────────────── */}
      <motion.section
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        className="max-w-6xl mx-auto px-6 pb-4"
      >
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search species, regions, habitats..."
            className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-white/[0.03] border border-white/[0.06] text-sm text-white placeholder-zinc-500 outline-none focus:border-radar-green/30 focus:ring-1 focus:ring-radar-green/20 transition-all"
          />
        </div>
      </motion.section>

      {/* ── Filter bar ──────────────────────────────────────── */}
      <motion.section
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.15 }}
        className="max-w-6xl mx-auto px-6 pb-4"
      >
        <div className="flex flex-wrap items-center gap-2 mb-2">
          {/* Status filters */}
          {STATUS_FILTERS.map((f) => {
            const isActive = statusFilter === f.value;
            return (
              <button
                key={f.value}
                onClick={() => setStatusFilter(f.value)}
                className={
                  isActive
                    ? "px-3 py-1.5 text-sm rounded-lg text-radar-green bg-radar-green/10 border border-radar-green/20"
                    : "px-3 py-1.5 text-sm rounded-lg text-zinc-400 hover:text-white hover:bg-white/5 transition-colors border border-transparent"
                }
              >
                {f.label}
              </button>
            );
          })}

          {/* Divider */}
          <div className="w-px h-6 bg-white/[0.06] mx-1" />

          {/* Taxon filters */}
          {TAXON_FILTERS.map((f) => {
            const isActive = taxonFilter === f.value;
            return (
              <button
                key={f.value}
                onClick={() => setTaxonFilter(f.value)}
                className={
                  isActive
                    ? "px-3 py-1.5 text-sm rounded-lg text-radar-green bg-radar-green/10 border border-radar-green/20"
                    : "px-3 py-1.5 text-sm rounded-lg text-zinc-400 hover:text-white hover:bg-white/5 transition-colors border border-transparent"
                }
              >
                {f.label}
              </button>
            );
          })}

          {/* Divider */}
          <div className="w-px h-6 bg-white/[0.06] mx-1" />

          {/* Watched only toggle */}
          {watchlist.count > 0 && (
            <button
              onClick={() => setShowWatchedOnly(!showWatchedOnly)}
              className={
                showWatchedOnly
                  ? "px-3 py-1.5 text-sm rounded-lg text-red-400 bg-red-500/10 border border-red-500/20"
                  : "px-3 py-1.5 text-sm rounded-lg text-zinc-400 hover:text-white hover:bg-white/5 transition-colors border border-transparent"
              }
            >
              Watched ({watchlist.count})
            </button>
          )}
        </div>

        {/* Summary stats */}
        <p className="text-sm text-zinc-500 mt-3">
          {filtered.length} species
          {" | "}
          {criticalCount} critically endangered
          {" | "}
          {endangeredCount} endangered
        </p>
      </motion.section>

      {/* ── Species grid ────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-6 pb-16">
        <AnimatePresence mode="popLayout">
          {filtered.length === 0 ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-center py-20"
            >
              <p className="text-zinc-500 mb-2">No species found</p>
              <p className="text-zinc-600 text-sm">
                Try adjusting your search or filter criteria.
              </p>
              {(statusFilter !== "all" || taxonFilter !== "all" || searchQuery || showWatchedOnly) && (
                <button
                  onClick={() => {
                    setStatusFilter("all");
                    setTaxonFilter("all");
                    setSearchQuery("");
                    setShowWatchedOnly(false);
                  }}
                  className="text-xs text-radar-green hover:underline mt-3"
                >
                  Clear all filters
                </button>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="grid"
              layout
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
            >
              {filtered.map((species, i) => (
                <motion.div
                  key={species.id}
                  layout
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{
                    duration: 0.3,
                    delay: i * 0.05,
                    layout: { duration: 0.3 },
                  }}
                >
                  <SpeciesCard
                    species={species}
                    isWatched={watchlist.isWatched(species.id)}
                    onToggleWatch={watchlist.toggle}
                  />
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </section>
    </div>
  );
}

import { Suspense } from "react";

export default function SpeciesCatalogPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#09090b]" />}>
      <SpeciesCatalogPageContent />
    </Suspense>
  );
}
