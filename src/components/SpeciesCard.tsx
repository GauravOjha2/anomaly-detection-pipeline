"use client";

import { useRef, useCallback } from "react";
import Link from "next/link";
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { ArrowUpRight, ArrowDownRight, Minus, Heart } from "lucide-react";
import type { TrackedSpecies } from "@/lib/species-data";
import ConservationBadge from "./ConservationBadge";

interface SpeciesCardProps {
  species: TrackedSpecies;
  isWatched?: boolean;
  onToggleWatch?: (id: string) => void;
}

function formatPopulation(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`.replace(".0K", "K");
  return new Intl.NumberFormat("en-US").format(n);
}

const trendConfig = {
  increasing: {
    icon: ArrowUpRight,
    color: "text-emerald-400",
    label: "Increasing",
  },
  decreasing: {
    icon: ArrowDownRight,
    color: "text-red-400",
    label: "Decreasing",
  },
  stable: {
    icon: Minus,
    color: "text-yellow-400",
    label: "Stable",
  },
  unknown: {
    icon: Minus,
    color: "text-zinc-500",
    label: "Unknown",
  },
} as const;

export default function SpeciesCard({
  species,
  isWatched = false,
  onToggleWatch,
}: SpeciesCardProps) {
  const trend = trendConfig[species.population.trend];
  const TrendIcon = trend.icon;
  const cardRef = useRef<HTMLDivElement>(null);

  // 3D tilt values
  const rotateX = useMotionValue(0);
  const rotateY = useMotionValue(0);
  const glareX = useMotionValue(50);
  const glareY = useMotionValue(50);

  const springConfig = { stiffness: 300, damping: 30 };
  const springRotateX = useSpring(rotateX, springConfig);
  const springRotateY = useSpring(rotateY, springConfig);

  // Subtle glare highlight
  const glareOpacity = useTransform(
    springRotateX,
    [-8, 0, 8],
    [0.06, 0, 0.06]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!cardRef.current) return;
      const rect = cardRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;

      // Tilt max 6 degrees
      rotateX.set(((y - centerY) / centerY) * -6);
      rotateY.set(((x - centerX) / centerX) * 6);
      glareX.set((x / rect.width) * 100);
      glareY.set((y / rect.height) * 100);
    },
    [rotateX, rotateY, glareX, glareY]
  );

  const handleMouseLeave = useCallback(() => {
    rotateX.set(0);
    rotateY.set(0);
  }, [rotateX, rotateY]);

  return (
    <div className="relative group" style={{ perspective: "800px" }}>
      {/* Watchlist toggle */}
      {onToggleWatch && (
        <button
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onToggleWatch(species.id);
          }}
          className={`absolute top-3 left-3 z-10 p-1.5 rounded-lg backdrop-blur-sm transition-all ${
            isWatched
              ? "bg-red-500/20 text-red-400 border border-red-500/30"
              : "bg-black/40 text-zinc-400 border border-white/[0.1] opacity-0 group-hover:opacity-100"
          }`}
          aria-label={isWatched ? "Unfollow species" : "Follow species"}
        >
          <Heart
            className={`w-3.5 h-3.5 ${isWatched ? "fill-current" : ""}`}
          />
        </button>
      )}

      <Link href={`/species/${species.id}`}>
        <motion.div
          ref={cardRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{
            rotateX: springRotateX,
            rotateY: springRotateY,
            transformStyle: "preserve-3d",
          }}
          whileHover={{ y: -4 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
          className="radar-card rounded-xl overflow-hidden cursor-pointer relative"
        >
          {/* Glare overlay */}
          <motion.div
            className="absolute inset-0 z-[1] pointer-events-none rounded-xl"
            style={{
              opacity: glareOpacity,
              background: `radial-gradient(circle at ${glareX}% ${glareY}%, rgba(45, 212, 191, 0.15), transparent 60%)`,
            }}
          />

          {/* Image section */}
          <div className="relative h-[200px] overflow-hidden">
            <img
              src={species.imageUrl}
              alt={species.commonName}
              className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
            />
            {/* Gradient overlay */}
            <div className="absolute inset-0 bg-gradient-to-t from-[#09090b] via-transparent to-transparent" />
            {/* Conservation badge */}
            <div className="absolute top-3 right-3">
              <ConservationBadge status={species.conservationStatus} size="sm" />
            </div>
          </div>

          {/* Content section */}
          <div className="p-4 space-y-3">
            {/* Names */}
            <div>
              <h3 className="font-semibold text-white text-lg leading-tight group-hover:text-radar-green transition-colors duration-300">
                {species.commonName}
              </h3>
              <p className="italic text-zinc-500 text-sm">
                {species.scientificName}
              </p>
            </div>

            {/* Population */}
            <div className="flex items-center justify-between">
              <span className="text-zinc-500 text-sm">Population</span>
              <div className="flex items-center gap-1.5">
                <span className="text-white text-sm font-medium">
                  {formatPopulation(species.population.estimated)}
                </span>
                <TrendIcon className={`w-3.5 h-3.5 ${trend.color}`} />
              </div>
            </div>

            {/* Habitat */}
            <p className="text-zinc-400 text-sm truncate">{species.habitat}</p>

            {/* Threats */}
            {species.threats.length > 0 && (
              <div className="flex items-center gap-1.5 flex-wrap">
                {species.threats.slice(0, 2).map((threat, i) => (
                  <span
                    key={i}
                    className="text-[10px] px-1.5 py-0.5 bg-white/5 rounded text-zinc-500 truncate max-w-[140px]"
                  >
                    {threat}
                  </span>
                ))}
              </div>
            )}
          </div>
        </motion.div>
      </Link>
    </div>
  );
}
