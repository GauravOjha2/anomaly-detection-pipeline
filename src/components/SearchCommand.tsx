"use client";

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  PawPrint,
  AlertTriangle,
  MapPin,
  Command,
  X,
  Locate,
} from "lucide-react";
import { TRACKED_SPECIES } from "@/lib/species-data";
import { useSentinel } from "@/lib/SentinelContext";
import { useAnimalTracking } from "@/lib/AnimalTrackingContext";
import type { AnomalyType } from "@/lib/types";

interface SearchCommandProps {
  isOpen: boolean;
  onClose: () => void;
}

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

interface SearchResult {
  id: string;
  label: string;
  sublabel: string;
  type: "species" | "alert" | "page" | "animal";
  icon: typeof PawPrint;
  href: string;
}

export default function SearchCommand({ isOpen, onClose }: SearchCommandProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();
  const { alerts } = useSentinel();
  const { animals } = useAnimalTracking();

  // Focus input when modal opens
  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  // Build results
  const results: SearchResult[] = useMemo(() => {
    if (!query.trim()) {
      // Show default pages + species when no query
      return [
        { id: "page-dashboard", label: "Dashboard", sublabel: "Wildlife activity overview", type: "page", icon: MapPin, href: "/dashboard" },
        { id: "page-species", label: "Species Directory", sublabel: "Browse tracked species", type: "page", icon: PawPrint, href: "/species" },
        { id: "page-alerts", label: "Anomaly Alerts", sublabel: "View all detected anomalies", type: "page", icon: AlertTriangle, href: "/alerts" },
        { id: "page-animals", label: "Animal Tracking", sublabel: "Individual GPS tracking", type: "page", icon: Locate, href: "/animals" },
        ...TRACKED_SPECIES.slice(0, 4).map((sp) => ({
          id: `sp-${sp.id}`,
          label: sp.commonName,
          sublabel: sp.scientificName,
          type: "species" as const,
          icon: PawPrint,
          href: `/species/${sp.id}`,
        })),
      ];
    }

    const q = query.toLowerCase();
    const out: SearchResult[] = [];

    // Search species
    for (const sp of TRACKED_SPECIES) {
      if (
        sp.commonName.toLowerCase().includes(q) ||
        sp.scientificName.toLowerCase().includes(q) ||
        sp.range.regions.some((r) => r.toLowerCase().includes(q)) ||
        sp.iconicTaxon.toLowerCase().includes(q)
      ) {
        out.push({
          id: `sp-${sp.id}`,
          label: sp.commonName,
          sublabel: sp.scientificName,
          type: "species",
          icon: PawPrint,
          href: `/species/${sp.id}`,
        });
      }
    }

    // Search tracked animals
    for (const animal of animals) {
      if (
        animal.name.toLowerCase().includes(q) ||
        animal.species.toLowerCase().includes(q) ||
        animal.commonName.toLowerCase().includes(q) ||
        animal.id.toLowerCase().includes(q)
      ) {
        out.push({
          id: `animal-${animal.id}`,
          label: animal.name,
          sublabel: `${animal.commonName} — ${animal.status}`,
          type: "animal",
          icon: Locate,
          href: `/animals/${animal.studyId}/${animal.individualId}`,
        });
      }
    }

    // Search alerts
    for (const a of alerts.slice(0, 50)) {
      if (
        a.common_name.toLowerCase().includes(q) ||
        a.species_name.toLowerCase().includes(q) ||
        (a.place_name && a.place_name.toLowerCase().includes(q)) ||
        ANOMALY_LABELS[a.anomaly_type]?.toLowerCase().includes(q)
      ) {
        out.push({
          id: `alert-${a.alert_id}`,
          label: `${a.common_name} — ${ANOMALY_LABELS[a.anomaly_type]}`,
          sublabel: a.place_name || "Unknown location",
          type: "alert",
          icon: AlertTriangle,
          href: "/alerts",
        });
      }
    }

    // Pages
    const pages = [
      { label: "Dashboard", sublabel: "Wildlife activity overview", href: "/dashboard" },
      { label: "Species Directory", sublabel: "Browse tracked species", href: "/species" },
      { label: "Anomaly Alerts", sublabel: "View all detected anomalies", href: "/alerts" },
      { label: "Animal Tracking", sublabel: "Individual GPS tracking", href: "/animals" },
    ];
    for (const p of pages) {
      if (p.label.toLowerCase().includes(q) || p.sublabel.toLowerCase().includes(q)) {
        out.push({
          id: `page-${p.href}`,
          label: p.label,
          sublabel: p.sublabel,
          type: "page",
          icon: p.href === "/animals" ? Locate : MapPin,
          href: p.href,
        });
      }
    }

    return out.slice(0, 12);
  }, [query, alerts, animals]);

  // Keyboard navigation
  const handleSelect = useCallback(
    (result: SearchResult) => {
      router.push(result.href);
      onClose();
    },
    [router, onClose],
  );

  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((i) => Math.min(i + 1, results.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === "Enter" && results[selectedIndex]) {
        e.preventDefault();
        handleSelect(results[selectedIndex]);
      } else if (e.key === "Escape") {
        onClose();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, results, selectedIndex, handleSelect, onClose]);

  // Reset index when results change
  useEffect(() => {
    setSelectedIndex(0);
  }, [results.length]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-[70] bg-black/50 backdrop-blur-sm"
          />

          {/* Command palette */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -20 }}
            transition={{ type: "spring", stiffness: 400, damping: 30 }}
            className="fixed top-[15%] left-1/2 -translate-x-1/2 w-full max-w-lg z-[75] bg-[#0c0c0e] border border-white/[0.1] rounded-2xl shadow-2xl overflow-hidden"
          >
            {/* Search input */}
            <div className="flex items-center gap-3 px-4 py-3 border-b border-white/[0.06]">
              <Search className="w-4 h-4 text-zinc-500 shrink-0" />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search species, animals, alerts..."
                className="flex-1 bg-transparent text-sm text-white placeholder-zinc-500 outline-none"
              />
              <button
                onClick={onClose}
                className="shrink-0 p-1 text-zinc-500 hover:text-white transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Results */}
            <div className="max-h-[360px] overflow-y-auto py-2">
              {results.length === 0 && query.trim() && (
                <div className="px-4 py-8 text-center text-zinc-500 text-sm">
                  No results for &ldquo;{query}&rdquo;
                </div>
              )}

              {results.map((result, i) => {
                const Icon = result.icon;
                const isSelected = i === selectedIndex;

                return (
                  <button
                    key={result.id}
                    onClick={() => handleSelect(result)}
                    onMouseEnter={() => setSelectedIndex(i)}
                    className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                      isSelected ? "bg-white/[0.06]" : "hover:bg-white/[0.03]"
                    }`}
                  >
                    <div
                      className={`w-7 h-7 rounded-md flex items-center justify-center shrink-0 ${
                        result.type === "species"
                          ? "bg-emerald-500/10 text-emerald-400"
                          : result.type === "alert"
                          ? "bg-amber-500/10 text-amber-400"
                          : result.type === "animal"
                          ? "bg-cyan-500/10 text-cyan-400"
                          : "bg-teal-500/10 text-teal-400"
                      }`}
                    >
                      <Icon className="w-3.5 h-3.5" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-white truncate">{result.label}</p>
                      <p className="text-[11px] text-zinc-500 truncate">{result.sublabel}</p>
                    </div>
                    <span className="text-[10px] text-zinc-600 uppercase shrink-0">
                      {result.type}
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Footer */}
            <div className="px-4 py-2 border-t border-white/[0.06] flex items-center gap-4 text-[10px] text-zinc-600">
              <span className="flex items-center gap-1">
                <Command className="w-3 h-3" />K to toggle
              </span>
              <span>&uarr;&darr; navigate</span>
              <span>&crarr; select</span>
              <span>esc close</span>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
