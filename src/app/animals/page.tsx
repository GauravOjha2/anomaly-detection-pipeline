"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import {
  MapPin,
  Clock,
  Route,
  ChevronRight,
  Heart,
  Radio,
  Loader2,
  Search,
  Filter,
} from "lucide-react";

import Navbar from "@/components/Navbar";
import { useAnimalTracking } from "@/lib/AnimalTrackingContext";
import { getTimeSince } from "@/lib/time";

const statusColors = {
  active: { dot: "bg-emerald-400", text: "text-emerald-400", label: "Active" },
  inactive: { dot: "bg-yellow-400", text: "text-yellow-400", label: "Inactive" },
  unknown: { dot: "bg-zinc-500", text: "text-zinc-500", label: "Unknown" },
};

export default function AnimalsPage() {
  const {
    animals,
    isLoading,
    lastFetched,
    studies,
    followedAnimals,
    followAnimal,
    unfollowAnimal,
    isFollowing,
  } = useAnimalTracking();

  const [searchQuery, setSearchQuery] = useState("");
  const [studyFilter, setStudyFilter] = useState<number | "all">("all");
  const [showFollowedOnly, setShowFollowedOnly] = useState(false);

  const filteredAnimals = useMemo(() => {
    let result = animals;

    if (studyFilter !== "all") {
      result = result.filter((a) => a.studyId === studyFilter);
    }

    if (showFollowedOnly) {
      result = result.filter((a) => followedAnimals.has(a.id));
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (a) =>
          a.name.toLowerCase().includes(q) ||
          a.commonName.toLowerCase().includes(q) ||
          a.species.toLowerCase().includes(q)
      );
    }

    return result;
  }, [animals, studyFilter, showFollowedOnly, searchQuery, followedAnimals]);

  const followedCount = useMemo(
    () => animals.filter((a) => followedAnimals.has(a.id)).length,
    [animals, followedAnimals]
  );

  const activeCount = useMemo(
    () => animals.filter((a) => a.status === "active").length,
    [animals]
  );

  return (
    <div className="min-h-screen bg-[#09090b] text-white">
      <Navbar />

      <div className="max-w-6xl mx-auto px-6 pt-24 pb-16">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="mb-8"
        >
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <h1 className="text-3xl font-bold">Individual Tracking</h1>
              <p className="text-zinc-400 text-sm mt-1">
                GPS telemetry for individually tracked animals across{" "}
                {studies.length} studies
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs text-zinc-500">
              {lastFetched && (
                <span className="flex items-center gap-1.5">
                  <Clock className="w-3 h-3" />
                  Updated {getTimeSince(lastFetched)}
                </span>
              )}
              <span className="flex items-center gap-1.5">
                <Radio className="w-3 h-3 text-emerald-400" />
                {activeCount} active
              </span>
            </div>
          </div>
        </motion.div>

        {/* Stats row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {[
            { label: "Total Tracked", value: animals.length, color: "text-white" },
            { label: "Active Now", value: activeCount, color: "text-emerald-400" },
            { label: "Studies", value: studies.length, color: "text-white" },
            { label: "Following", value: followedCount, color: "text-red-400" },
          ].map((card, i) => (
            <motion.div
              key={card.label}
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.05 + i * 0.07 }}
              className="radar-card rounded-xl p-4"
            >
              <div className="text-xs text-zinc-500 uppercase mb-1">
                {card.label}
              </div>
              <div className={`text-2xl font-bold ${card.color}`}>
                {card.value}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Filters */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="flex flex-col sm:flex-row gap-3 mb-6"
        >
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
            <input
              type="text"
              placeholder="Search by name or species..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 bg-white/[0.03] border border-white/[0.08] rounded-lg text-sm text-white placeholder:text-zinc-600 focus:outline-none focus:border-radar-green/40 focus:ring-1 focus:ring-radar-green/20 transition-all"
            />
          </div>

          <div className="flex gap-2">
            <div className="relative">
              <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-500 pointer-events-none" />
              <select
                value={studyFilter === "all" ? "all" : String(studyFilter)}
                onChange={(e) =>
                  setStudyFilter(
                    e.target.value === "all" ? "all" : Number(e.target.value)
                  )
                }
                className="pl-9 pr-8 py-2.5 bg-white/[0.03] border border-white/[0.08] rounded-lg text-sm text-white appearance-none cursor-pointer focus:outline-none focus:border-radar-green/40 transition-all"
              >
                <option value="all">All Studies</option>
                {studies.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={() => setShowFollowedOnly(!showFollowedOnly)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm border transition-all ${
                showFollowedOnly
                  ? "bg-red-500/10 border-red-500/30 text-red-400"
                  : "bg-white/[0.03] border-white/[0.08] text-zinc-400 hover:text-white"
              }`}
            >
              <Heart
                className={`w-3.5 h-3.5 ${showFollowedOnly ? "fill-current" : ""}`}
              />
              Following
            </button>
          </div>
        </motion.div>

        {/* Loading state */}
        {isLoading && animals.length === 0 && (
          <div className="flex flex-col items-center justify-center py-20 gap-3">
            <Loader2 className="w-8 h-8 text-radar-green animate-spin" />
            <p className="text-sm text-zinc-500">
              Fetching GPS telemetry data...
            </p>
          </div>
        )}

        {/* Animal cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <AnimatePresence mode="popLayout">
            {filteredAnimals.map((animal, i) => {
              const status = statusColors[animal.status];
              const followed = isFollowing(animal.id);
              const lastLoc =
                animal.locations[animal.locations.length - 1];

              return (
                <motion.div
                  key={animal.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.3, delay: i * 0.03 }}
                  className="group radar-card rounded-xl overflow-hidden hover:border-white/[0.12] transition-all"
                >
                  <Link
                    href={`/animals/${animal.studyId}/${animal.individualId}`}
                    className="block p-4"
                  >
                    <div className="flex items-start justify-between gap-2 mb-3">
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-white truncate group-hover:text-radar-green transition-colors">
                          {animal.name}
                        </h3>
                        <p className="text-xs text-zinc-500 italic truncate">
                          {animal.commonName || animal.species}
                        </p>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <button
                          onClick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            if (followed) {
                              unfollowAnimal(animal.id);
                            } else {
                              followAnimal(animal.id);
                            }
                          }}
                          className={`p-1.5 rounded-md transition-colors ${
                            followed
                              ? "text-red-400 hover:bg-red-500/10"
                              : "text-zinc-600 hover:text-zinc-400 hover:bg-white/[0.05]"
                          }`}
                        >
                          <Heart
                            className={`w-3.5 h-3.5 ${followed ? "fill-current" : ""}`}
                          />
                        </button>
                        <ChevronRight className="w-4 h-4 text-zinc-600 group-hover:text-zinc-400 transition-colors" />
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-3 text-xs">
                      <div>
                        <span className="text-zinc-600 block mb-0.5">
                          Status
                        </span>
                        <span className={`flex items-center gap-1.5 ${status.text}`}>
                          <span
                            className={`w-1.5 h-1.5 rounded-full ${status.dot}`}
                          />
                          {status.label}
                        </span>
                      </div>
                      <div>
                        <span className="text-zinc-600 block mb-0.5">
                          Distance
                        </span>
                        <span className="text-white flex items-center gap-1">
                          <Route className="w-3 h-3 text-zinc-500" />
                          {animal.totalDistance > 1000
                            ? `${(animal.totalDistance / 1000).toFixed(1)}K`
                            : animal.totalDistance.toFixed(0)}{" "}
                          km
                        </span>
                      </div>
                      <div>
                        <span className="text-zinc-600 block mb-0.5">
                          Last Seen
                        </span>
                        <span className="text-white flex items-center gap-1">
                          <Clock className="w-3 h-3 text-zinc-500" />
                          {getTimeSince(new Date(animal.lastSeen))}
                        </span>
                      </div>
                    </div>

                    {lastLoc && (
                      <div className="mt-3 pt-3 border-t border-white/[0.04] flex items-center gap-1.5 text-[11px] text-zinc-600">
                        <MapPin className="w-3 h-3" />
                        {lastLoc.lat.toFixed(3)}°,{" "}
                        {lastLoc.lng.toFixed(3)}°
                        {lastLoc.altitude !== undefined && (
                          <span className="ml-auto">
                            Alt: {lastLoc.altitude}m
                          </span>
                        )}
                      </div>
                    )}
                  </Link>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>

        {/* Empty state */}
        {!isLoading && filteredAnimals.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center justify-center py-20 gap-3"
          >
            <MapPin className="w-8 h-8 text-zinc-700" />
            <p className="text-sm text-zinc-500">
              {showFollowedOnly
                ? "You're not following any animals yet"
                : "No animals match your search"}
            </p>
            {showFollowedOnly && (
              <button
                onClick={() => setShowFollowedOnly(false)}
                className="text-xs text-radar-green hover:underline"
              >
                View all animals
              </button>
            )}
          </motion.div>
        )}

        {/* Study info cards */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.2 }}
          className="mt-16"
        >
          <h2 className="text-lg font-semibold mb-2">Tracking Studies</h2>
          <p className="text-sm text-zinc-500 mb-4">
            GPS telemetry research programs monitoring individually collared or tagged animals.
            Each study provides real-time location data from field-deployed transmitters.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {studies.map((study) => {
              const studyAnimals = animals.filter(
                (a) => a.studyId === study.id
              );
              return (
                <div
                  key={study.id}
                  className="radar-card rounded-xl p-4 space-y-3"
                >
                  <div className="flex items-start gap-3">
                    {study.thumbnailUrl && (
                      <img
                        src={study.thumbnailUrl}
                        alt={study.commonName}
                        className="w-12 h-12 rounded-lg object-cover"
                      />
                    )}
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-sm text-white truncate">
                        {study.name}
                      </h3>
                      <p className="text-xs text-zinc-500 italic">
                        {study.commonName || study.species}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 text-xs text-zinc-500">
                    <span>{studyAnimals.length} individuals</span>
                    <span>
                      {studyAnimals.filter((a) => a.status === "active").length}{" "}
                      active
                    </span>
                  </div>
                  <button
                    onClick={() => setStudyFilter(study.id)}
                    className="text-xs text-radar-green hover:underline"
                  >
                    View animals →
                  </button>
                </div>
              );
            })}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
