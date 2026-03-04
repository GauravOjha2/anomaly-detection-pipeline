"use client";

import { useMemo } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  MapPin,
  Clock,
  Route,
  Gauge,
  Mountain,
  Heart,
  Navigation,
  Radio,
  Loader2,
} from "lucide-react";

import Navbar from "@/components/Navbar";
import AnimalTrackMap from "@/components/AnimalTrackMap";
import { useAnimalTracking } from "@/lib/AnimalTrackingContext";
import { getTimeSince } from "@/lib/time";

const statusConfig = {
  active: { dot: "bg-emerald-400", text: "text-emerald-400", label: "Active — transmitting" },
  inactive: { dot: "bg-yellow-400", text: "text-yellow-400", label: "Inactive — no recent data" },
  unknown: { dot: "bg-zinc-500", text: "text-zinc-500", label: "Unknown status" },
};

export default function AnimalDetailPage() {
  const params = useParams();
  const studyId = Number(params.studyId);
  const individualId = Number(params.individualId);

  const {
    animals,
    isLoading,
    studies,
    isFollowing,
    followAnimal,
    unfollowAnimal,
  } = useAnimalTracking();

  const animal = useMemo(
    () => animals.find((a) => a.studyId === studyId && a.individualId === individualId),
    [animals, studyId, individualId]
  );

  const study = useMemo(
    () => studies.find((s) => s.id === studyId),
    [studies, studyId]
  );

  const stats = useMemo(() => {
    if (!animal || animal.locations.length === 0) return null;

    const locs = animal.locations;
    const speeds = locs
      .filter((l) => l.groundSpeed !== undefined)
      .map((l) => l.groundSpeed as number);
    const altitudes = locs
      .filter((l) => l.altitude !== undefined)
      .map((l) => l.altitude as number);

    return {
      avgSpeed: speeds.length > 0 ? speeds.reduce((a, b) => a + b, 0) / speeds.length : 0,
      maxSpeed: speeds.length > 0 ? Math.max(...speeds) : 0,
      maxAltitude: altitudes.length > 0 ? Math.max(...altitudes) : 0,
      avgAltitude: altitudes.length > 0 ? altitudes.reduce((a, b) => a + b, 0) / altitudes.length : 0,
      dataPoints: locs.length,
      firstRecord: locs[0].timestamp,
      lastRecord: locs[locs.length - 1].timestamp,
    };
  }, [animal]);

  // Recent locations for the table
  const recentLocations = useMemo(() => {
    if (!animal) return [];
    return [...animal.locations].reverse().slice(0, 20);
  }, [animal]);

  if (isLoading && !animal) {
    return (
      <div className="min-h-screen bg-[#09090b] text-white">
        <Navbar />
        <div className="flex flex-col items-center justify-center pt-40 gap-3">
          <Loader2 className="w-8 h-8 text-radar-green animate-spin" />
          <p className="text-sm text-zinc-500">Loading animal data...</p>
        </div>
      </div>
    );
  }

  if (!animal) {
    return (
      <div className="min-h-screen bg-[#09090b] text-white">
        <Navbar />
        <div className="flex flex-col items-center justify-center pt-40 gap-4">
          <h1 className="text-2xl font-bold">Animal not found</h1>
          <p className="text-zinc-400">
            This individual is not in our tracking database.
          </p>
          <Link
            href="/animals"
            className="inline-flex items-center gap-2 text-radar-green hover:underline text-sm mt-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Animals
          </Link>
        </div>
      </div>
    );
  }

  const status = statusConfig[animal.status];
  const followed = isFollowing(animal.id);
  const lastLoc = animal.locations[animal.locations.length - 1];

  return (
    <div className="min-h-screen bg-[#09090b] text-white">
      <Navbar />

      <div className="max-w-6xl mx-auto px-6 pt-24 pb-16 space-y-8">
        {/* Breadcrumb */}
        <Link
          href="/animals"
          className="inline-flex items-center gap-2 text-zinc-400 hover:text-white transition-colors text-sm"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Animals
        </Link>

        {/* Header */}
        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="flex flex-col md:flex-row md:items-start justify-between gap-4"
        >
          <div className="flex items-start gap-4">
            {animal.thumbnailUrl && (
              <img
                src={animal.thumbnailUrl}
                alt={animal.commonName}
                className="w-16 h-16 rounded-xl object-cover border border-white/[0.08]"
              />
            )}
            <div>
              <h1 className="text-3xl font-bold">{animal.name}</h1>
              <p className="text-zinc-400 italic text-sm mt-0.5">
                {animal.commonName || animal.species}
              </p>
              {study && (
                <p className="text-zinc-600 text-xs mt-1">
                  Study: {study.name}
                </p>
              )}
              <div className="flex items-center gap-2 mt-2">
                <span
                  className={`w-2 h-2 rounded-full ${status.dot}`}
                />
                <span className={`text-xs ${status.text}`}>
                  {status.label}
                </span>
                {lastLoc && (
                  <span className="text-xs text-zinc-600 ml-2">
                    · Last seen {getTimeSince(new Date(animal.lastSeen))}
                  </span>
                )}
              </div>
            </div>
          </div>

          <button
            onClick={() =>
              followed
                ? unfollowAnimal(animal.id)
                : followAnimal(animal.id)
            }
            className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all ${
              followed
                ? "bg-red-500/15 text-red-400 border border-red-500/30 hover:bg-red-500/25"
                : "bg-white/[0.05] text-zinc-400 border border-white/[0.1] hover:text-white hover:border-white/[0.2]"
            }`}
          >
            <Heart
              className={`w-4 h-4 ${followed ? "fill-current" : ""}`}
            />
            {followed ? "Following" : "Follow"}
          </button>
        </motion.section>

        {/* Stats cards */}
        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.05 }}
          className="grid grid-cols-2 md:grid-cols-5 gap-4"
        >
          <div className="radar-card rounded-xl p-4 space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-zinc-500">
              <Route className="w-3.5 h-3.5" />
              Total Distance
            </div>
            <div className="text-lg font-semibold text-white">
              {animal.totalDistance > 1000
                ? `${(animal.totalDistance / 1000).toFixed(1)}K`
                : animal.totalDistance.toFixed(0)}{" "}
              km
            </div>
          </div>

          <div className="radar-card rounded-xl p-4 space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-zinc-500">
              <Gauge className="w-3.5 h-3.5" />
              Avg Speed
            </div>
            <div className="text-lg font-semibold text-white">
              {stats ? `${stats.avgSpeed.toFixed(1)}` : "—"}{" "}
              <span className="text-xs text-zinc-500 font-normal">km/h</span>
            </div>
          </div>

          <div className="radar-card rounded-xl p-4 space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-zinc-500">
              <Gauge className="w-3.5 h-3.5" />
              Max Speed
            </div>
            <div className="text-lg font-semibold text-white">
              {stats ? `${stats.maxSpeed.toFixed(1)}` : "—"}{" "}
              <span className="text-xs text-zinc-500 font-normal">km/h</span>
            </div>
          </div>

          <div className="radar-card rounded-xl p-4 space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-zinc-500">
              <Mountain className="w-3.5 h-3.5" />
              Max Altitude
            </div>
            <div className="text-lg font-semibold text-white">
              {stats ? `${stats.maxAltitude}` : "—"}{" "}
              <span className="text-xs text-zinc-500 font-normal">m</span>
            </div>
          </div>

          <div className="radar-card rounded-xl p-4 space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-zinc-500">
              <Radio className="w-3.5 h-3.5" />
              Data Points
            </div>
            <div className="text-lg font-semibold text-white">
              {stats ? stats.dataPoints : "—"}
            </div>
          </div>
        </motion.section>

        {/* Map */}
        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
        >
          <div className="radar-card rounded-xl overflow-hidden">
            <div className="px-5 py-3 border-b border-white/[0.06] flex items-center justify-between">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <MapPin className="w-4 h-4 text-radar-green" />
                GPS Track
              </h2>
              {lastLoc && (
                <span className="text-xs text-zinc-500">
                  {lastLoc.lat.toFixed(4)}°, {lastLoc.lng.toFixed(4)}°
                </span>
              )}
            </div>
            <AnimalTrackMap
              locations={animal.locations}
              animalName={animal.name}
              height="450px"
            />
          </div>
        </motion.section>

        {/* Recent locations table */}
        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.15 }}
        >
          <div className="radar-card rounded-xl overflow-hidden">
            <div className="px-5 py-3 border-b border-white/[0.06]">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <Clock className="w-4 h-4 text-radar-green" />
                Recent Locations
              </h2>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/[0.04] text-xs text-zinc-500 uppercase">
                    <th className="text-left px-5 py-3 font-medium">Time</th>
                    <th className="text-left px-5 py-3 font-medium">
                      Latitude
                    </th>
                    <th className="text-left px-5 py-3 font-medium">
                      Longitude
                    </th>
                    <th className="text-left px-5 py-3 font-medium">Speed</th>
                    <th className="text-left px-5 py-3 font-medium">
                      Heading
                    </th>
                    <th className="text-left px-5 py-3 font-medium">
                      Altitude
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {recentLocations.map((loc, i) => (
                    <tr
                      key={i}
                      className={`border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors ${
                        i === 0 ? "bg-radar-green/[0.03]" : ""
                      }`}
                    >
                      <td className="px-5 py-2.5 text-zinc-300 whitespace-nowrap">
                        <div className="flex items-center gap-2">
                          {i === 0 && (
                            <span className="w-1.5 h-1.5 rounded-full bg-radar-green" />
                          )}
                          {new Date(loc.timestamp).toLocaleDateString(
                            undefined,
                            {
                              month: "short",
                              day: "numeric",
                              hour: "2-digit",
                              minute: "2-digit",
                            }
                          )}
                        </div>
                      </td>
                      <td className="px-5 py-2.5 font-mono text-xs text-zinc-400">
                        {loc.lat.toFixed(5)}°
                      </td>
                      <td className="px-5 py-2.5 font-mono text-xs text-zinc-400">
                        {loc.lng.toFixed(5)}°
                      </td>
                      <td className="px-5 py-2.5 text-zinc-400">
                        {loc.groundSpeed !== undefined ? (
                          <span className="flex items-center gap-1">
                            <Gauge className="w-3 h-3 text-zinc-600" />
                            {loc.groundSpeed.toFixed(1)} km/h
                          </span>
                        ) : (
                          <span className="text-zinc-600">—</span>
                        )}
                      </td>
                      <td className="px-5 py-2.5 text-zinc-400">
                        {loc.heading !== undefined ? (
                          <span className="flex items-center gap-1">
                            <Navigation
                              className="w-3 h-3 text-zinc-600"
                              style={{
                                transform: `rotate(${loc.heading}deg)`,
                              }}
                            />
                            {loc.heading}°
                          </span>
                        ) : (
                          <span className="text-zinc-600">—</span>
                        )}
                      </td>
                      <td className="px-5 py-2.5 text-zinc-400">
                        {loc.altitude !== undefined ? (
                          `${loc.altitude}m`
                        ) : (
                          <span className="text-zinc-600">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {animal.locations.length > 20 && (
              <div className="px-5 py-3 border-t border-white/[0.04] text-xs text-zinc-600 text-center">
                Showing 20 of {animal.locations.length} locations
              </div>
            )}
          </div>
        </motion.section>
      </div>
    </div>
  );
}
