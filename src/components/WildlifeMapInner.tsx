"use client";

import { useRef, useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  Circle,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";

// ── Types ────────────────────────────────────────────────────

export interface WildlifeMapMarker {
  id: string;
  lat: number;
  lng: number;
  label: string;
  type: "normal" | "anomaly" | "critical";
  species?: string;
  details?: string;
}

interface WildlifeMapRange {
  center: [number, number];
  radiusDeg: number;
  label: string;
  color?: string;
}

interface WildlifeMapProps {
  markers?: WildlifeMapMarker[];
  ranges?: WildlifeMapRange[];
  center?: [number, number];
  zoom?: number;
  height?: string;
  className?: string;
  onMarkerClick?: (markerId: string) => void;
  activeMarkerId?: string | null;
}

const markerColors: Record<WildlifeMapMarker["type"], string> = {
  normal: "#2dd4bf",
  anomaly: "#f59e0b",
  critical: "#ef4444",
};

// ── FlyTo component — animates map to activeMarkerId ─────────

function FlyToMarker({
  markers,
  activeMarkerId,
}: {
  markers: WildlifeMapMarker[];
  activeMarkerId: string | null;
}) {
  const map = useMap();

  useEffect(() => {
    if (!activeMarkerId) return;
    const marker = markers.find((m) => m.id === activeMarkerId);
    if (marker) {
      map.flyTo([marker.lat, marker.lng], 6, { duration: 1.2 });
    }
  }, [activeMarkerId, markers, map]);

  return null;
}

// ── Main component ───────────────────────────────────────────

export default function WildlifeMapInner({
  markers = [],
  ranges = [],
  center = [20, 0],
  zoom = 2,
  height = "500px",
  className = "",
  onMarkerClick,
  activeMarkerId = null,
}: WildlifeMapProps) {
  const _containerRef = useRef<HTMLDivElement>(null);

  return (
    <div className={className} style={{ height }} ref={_containerRef}>
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
        .leaflet-popup-close-button {
          color: #71717a !important;
        }
        .leaflet-popup-close-button:hover {
          color: #fff !important;
        }
      `}</style>
      <MapContainer
        center={center}
        zoom={zoom}
        scrollWheelZoom={true}
        style={{ height: "100%", width: "100%", borderRadius: "0.75rem" }}
      >
        <TileLayer
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        {/* FlyTo handler */}
        <FlyToMarker markers={markers} activeMarkerId={activeMarkerId} />

        {/* Species ranges */}
        {ranges.map((range, i) => (
          <Circle
            key={`range-${i}`}
            center={range.center}
            radius={range.radiusDeg * 111000}
            pathOptions={{
              color: range.color || "#2dd4bf",
              fillColor: range.color || "#2dd4bf",
              fillOpacity: 0.08,
              weight: 1,
            }}
          >
            <Popup>
              <div className="text-sm font-medium">{range.label}</div>
            </Popup>
          </Circle>
        ))}

        {/* Sighting markers */}
        {markers.map((marker) => {
          const isActive = marker.id === activeMarkerId;
          const isCritical = marker.type === "critical";

          return (
            <CircleMarker
              key={marker.id}
              center={[marker.lat, marker.lng]}
              radius={isActive ? 10 : isCritical ? 8 : 6}
              pathOptions={{
                color: isActive ? "#fff" : markerColors[marker.type],
                fillColor: markerColors[marker.type],
                fillOpacity: isActive ? 0.9 : 0.7,
                weight: isActive ? 2 : 1,
                className: isCritical ? "leaflet-pulsing-marker" : undefined,
              }}
              eventHandlers={{
                click: () => {
                  if (onMarkerClick) onMarkerClick(marker.id);
                },
              }}
            >
              <Popup>
                <div className="space-y-1.5 min-w-[160px]">
                  <div className="font-semibold text-sm text-white">{marker.label}</div>
                  {marker.species && (
                    <div className="text-xs text-zinc-400 italic">{marker.species}</div>
                  )}
                  {marker.details && (
                    <div className="text-xs text-zinc-400">{marker.details}</div>
                  )}
                  {onMarkerClick && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onMarkerClick(marker.id);
                      }}
                      className="text-[10px] text-teal-400 hover:text-teal-300 underline mt-1"
                    >
                      Investigate
                    </button>
                  )}
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
}
