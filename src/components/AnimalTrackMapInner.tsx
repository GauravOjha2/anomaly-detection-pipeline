"use client";

import { useMemo, useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  CircleMarker,
  Popup,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import type { LatLngExpression } from "leaflet";
import "leaflet/dist/leaflet.css";
import type { GPSLocation } from "@/lib/movebank";

interface AnimalTrackMapInnerProps {
  locations: GPSLocation[];
  animalName: string;
  center?: [number, number];
  zoom?: number;
  height?: string;
  className?: string;
}

function FitBounds({ positions }: { positions: LatLngExpression[] }) {
  const map = useMap();

  useEffect(() => {
    if (positions.length < 2) return;
    const bounds = L.latLngBounds(positions);
    map.fitBounds(bounds, { padding: [40, 40], maxZoom: 12 });
  }, [positions, map]);

  return null;
}

export default function AnimalTrackMapInner({
  locations,
  animalName,
  center,
  zoom = 6,
  height = "500px",
  className = "",
}: AnimalTrackMapInnerProps) {
  const positions: LatLngExpression[] = useMemo(
    () => locations.map((loc) => [loc.lat, loc.lng] as LatLngExpression),
    [locations]
  );

  const mapCenter: [number, number] = useMemo(() => {
    if (center) return center;
    if (locations.length === 0) return [0, 0];
    const last = locations[locations.length - 1];
    return [last.lat, last.lng];
  }, [center, locations]);

  // Create gradient segments for the track
  const segments = useMemo(() => {
    if (locations.length === 0) return [];
    const segs: { positions: LatLngExpression[]; opacity: number }[] = [];
    const totalSegments = Math.max(1, Math.ceil(locations.length / 5));

    for (let i = 0; i < totalSegments; i++) {
      const start = i * 5;
      const end = Math.min(start + 6, locations.length);
      if (start >= locations.length) break;

      const segPositions = locations
        .slice(start, end)
        .map((loc) => [loc.lat, loc.lng] as LatLngExpression);

      // Fade from dim (old) to bright (recent)
      const progress = (i + 1) / totalSegments;
      segs.push({
        positions: segPositions,
        opacity: 0.2 + progress * 0.8,
      });
    }

    return segs;
  }, [locations]);

  if (locations.length === 0) {
    return (
      <div
        className={`flex items-center justify-center bg-zinc-900/50 rounded-xl border border-zinc-800 ${className}`}
        style={{ height }}
      >
        <span className="text-zinc-500 text-sm">No GPS data available</span>
      </div>
    );
  }

  const firstLoc = locations[0];
  const lastLoc = locations[locations.length - 1];

  return (
    <div className={className} style={{ height }}>
      <style>{`
        .leaflet-container { background: #09090b; }
        .leaflet-popup-content-wrapper {
          background: #18181b !important;
          color: #fff !important;
          border: 1px solid rgba(255,255,255,0.1) !important;
          border-radius: 12px !important;
          box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important;
        }
        .leaflet-popup-tip {
          background: #18181b !important;
          border: 1px solid rgba(255,255,255,0.1) !important;
        }
        .leaflet-popup-close-button { color: #71717a !important; }
        .leaflet-popup-close-button:hover { color: #fff !important; }
      `}</style>
      <MapContainer
        center={mapCenter}
        zoom={zoom}
        scrollWheelZoom={true}
        style={{ height: "100%", width: "100%", borderRadius: "0.75rem" }}
      >
        <TileLayer
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        <FitBounds positions={positions} />

        {/* Track line with gradient opacity */}
        {segments.map((seg, i) => (
          <Polyline
            key={`track-${i}`}
            positions={seg.positions}
            pathOptions={{
              color: "#2dd4bf",
              weight: 3,
              opacity: seg.opacity,
              lineCap: "round",
              lineJoin: "round",
            }}
          />
        ))}

        {/* Start point */}
        <CircleMarker
          center={[firstLoc.lat, firstLoc.lng]}
          radius={5}
          pathOptions={{
            color: "#71717a",
            fillColor: "#3f3f46",
            fillOpacity: 0.9,
            weight: 2,
          }}
        >
          <Popup>
            <div className="space-y-1 min-w-[140px]">
              <div className="font-semibold text-xs text-zinc-400">
                Track Start
              </div>
              <div className="text-xs text-white">{animalName}</div>
              <div className="text-[10px] text-zinc-500">
                {new Date(firstLoc.timestamp).toLocaleDateString(undefined, {
                  month: "short",
                  day: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </div>
            </div>
          </Popup>
        </CircleMarker>

        {/* Current position (last point) */}
        <CircleMarker
          center={[lastLoc.lat, lastLoc.lng]}
          radius={8}
          pathOptions={{
            color: "#fff",
            fillColor: "#2dd4bf",
            fillOpacity: 0.9,
            weight: 2,
          }}
        >
          <Popup>
            <div className="space-y-1 min-w-[140px]">
              <div className="font-semibold text-xs text-teal-400">
                Current Position
              </div>
              <div className="text-xs text-white">{animalName}</div>
              <div className="text-[10px] text-zinc-500">
                {new Date(lastLoc.timestamp).toLocaleDateString(undefined, {
                  month: "short",
                  day: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </div>
              {lastLoc.groundSpeed !== undefined && (
                <div className="text-[10px] text-zinc-500">
                  Speed: {lastLoc.groundSpeed.toFixed(1)} km/h
                </div>
              )}
              {lastLoc.altitude !== undefined && (
                <div className="text-[10px] text-zinc-500">
                  Alt: {lastLoc.altitude}m
                </div>
              )}
            </div>
          </Popup>
        </CircleMarker>

        {/* Waypoint dots for every Nth point */}
        {locations
          .filter((_, i) => i > 0 && i < locations.length - 1 && i % 10 === 0)
          .map((loc, i) => (
            <CircleMarker
              key={`waypoint-${i}`}
              center={[loc.lat, loc.lng]}
              radius={3}
              pathOptions={{
                color: "#2dd4bf",
                fillColor: "#2dd4bf",
                fillOpacity: 0.5,
                weight: 1,
              }}
            >
              <Popup>
                <div className="space-y-1 min-w-[120px]">
                  <div className="text-[10px] text-zinc-500">
                    {new Date(loc.timestamp).toLocaleDateString(undefined, {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </div>
                  {loc.groundSpeed !== undefined && (
                    <div className="text-[10px] text-zinc-500">
                      {loc.groundSpeed.toFixed(1)} km/h
                    </div>
                  )}
                </div>
              </Popup>
            </CircleMarker>
          ))}
      </MapContainer>
    </div>
  );
}
