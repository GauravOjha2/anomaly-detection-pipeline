// ============================================================
// MOVEBANK API CLIENT — Individual Animal GPS Telemetry
// Fetches real tracking data from Movebank public API
// Falls back to high-quality simulated data when API unavailable
// ============================================================

// ---- Types ----

export interface TrackedAnimal {
  id: string; // "{studyId}-{individualId}"
  studyId: number;
  individualId: number;
  name: string; // "Albatross 4261" or custom name
  species: string; // Scientific name
  commonName: string;
  locations: GPSLocation[];
  lastSeen: string; // ISO timestamp
  totalDistance: number; // km traveled (from locations)
  status: "active" | "inactive" | "unknown";
  thumbnailUrl: string | null;
}

export interface GPSLocation {
  timestamp: string; // ISO
  lat: number;
  lng: number;
  groundSpeed?: number; // m/s
  heading?: number; // degrees
  altitude?: number; // meters above ellipsoid
}

export interface MovebankStudy {
  id: number;
  name: string;
  species: string;
  commonName: string;
  individualCount: number;
  thumbnailUrl: string | null;
}

// ---- Movebank API Response Types ----

interface MovebankIndividual {
  individual_local_identifier: string;
  individual_taxon_canonical_name: string;
  individual_id: number;
  locations: MovebankLocation[];
}

interface MovebankLocation {
  timestamp: number; // epoch ms
  location_long: number;
  location_lat: number;
  ground_speed?: number;
  heading?: number;
  height_above_ellipsoid?: number;
}

interface MovebankApiResponse {
  individuals: MovebankIndividual[];
}

// ---- Config ----

const MOVEBANK_BASE = "https://www.movebank.org/movebank/service/public/json";
const FETCH_TIMEOUT = 8000; // 8s timeout — Movebank can be slow

// Studies we track — curated for interesting wildlife
export const TRACKED_STUDIES: MovebankStudy[] = [
  {
    id: 2911040,
    name: "Galapagos Albatrosses",
    species: "Phoebastria irrorata",
    commonName: "Waved Albatross",
    individualCount: 28,
    thumbnailUrl: "https://images.unsplash.com/photo-1611689342806-0a0f1ab7efec?w=320&h=320&fit=crop",
  },
  {
    id: 10763606,
    name: "Savanna Elephants — Etosha",
    species: "Loxodonta africana",
    commonName: "African Elephant",
    individualCount: 14,
    thumbnailUrl: "https://images.unsplash.com/photo-1557050543-4d5f4e07ef46?w=320&h=320&fit=crop",
  },
  {
    id: 9651291,
    name: "White Storks — Germany",
    species: "Ciconia ciconia",
    commonName: "White Stork",
    individualCount: 12,
    thumbnailUrl: "https://images.unsplash.com/photo-1591608971362-f08b2a75731a?w=320&h=320&fit=crop",
  },
];

// ---- API Fetch ----

async function fetchFromMovebank(
  studyId: number,
  maxEventsPerIndividual: number = 50
): Promise<TrackedAnimal[] | null> {
  const url = `${MOVEBANK_BASE}?study_id=${studyId}&sensor_type=gps&max_events_per_individual=${maxEventsPerIndividual}&attributes=timestamp,location_long,location_lat,ground_speed,heading,height_above_ellipsoid`;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT);

  try {
    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(timeout);

    if (!res.ok) return null;

    const text = await res.text();
    if (!text || text.trim().length === 0) return null;

    const data: MovebankApiResponse = JSON.parse(text);
    if (!data.individuals || data.individuals.length === 0) return null;

    const study = TRACKED_STUDIES.find((s) => s.id === studyId);

    return data.individuals.map((ind) => {
      const locations: GPSLocation[] = (ind.locations || [])
        .filter((loc) => loc.location_lat && loc.location_long)
        .map((loc) => ({
          timestamp: new Date(loc.timestamp).toISOString(),
          lat: loc.location_lat,
          lng: loc.location_long,
          groundSpeed: loc.ground_speed,
          heading: loc.heading,
          altitude: loc.height_above_ellipsoid,
        }))
        .sort(
          (a, b) =>
            new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
        );

      const lastLoc = locations[locations.length - 1];

      return {
        id: `${studyId}-${ind.individual_id}`,
        studyId,
        individualId: ind.individual_id,
        name: formatAnimalName(
          ind.individual_local_identifier,
          study?.commonName || ""
        ),
        species: ind.individual_taxon_canonical_name || study?.species || "Unknown",
        commonName: study?.commonName || "",
        locations,
        lastSeen: lastLoc?.timestamp || new Date().toISOString(),
        totalDistance: calculateTotalDistance(locations),
        status: getAnimalStatus(lastLoc?.timestamp),
        thumbnailUrl: study?.thumbnailUrl || null,
      };
    });
  } catch {
    clearTimeout(timeout);
    return null;
  }
}

// ---- Public API ----

/**
 * Fetches tracked animals for a study. Uses Movebank API with fallback.
 */
export async function getTrackedAnimals(
  studyId: number,
  maxEvents: number = 50
): Promise<TrackedAnimal[]> {
  // Try real API first
  const real = await fetchFromMovebank(studyId, maxEvents);
  if (real && real.length > 0) return real;

  // Fallback: generate realistic simulated data
  return generateFallbackAnimals(studyId, maxEvents);
}

/**
 * Fetches a single tracked animal by study and individual ID.
 */
export async function getTrackedAnimal(
  studyId: number,
  individualId: number,
  maxEvents: number = 200
): Promise<TrackedAnimal | null> {
  const animals = await getTrackedAnimals(studyId, maxEvents);
  return animals.find((a) => a.individualId === individualId) || null;
}

/**
 * Fetches all animals across all tracked studies.
 */
export async function getAllTrackedAnimals(): Promise<TrackedAnimal[]> {
  // Fetch all studies in parallel — but Movebank has 1 concurrent request limit
  // so we chain them with small delays
  const results: TrackedAnimal[] = [];

  for (const study of TRACKED_STUDIES) {
    const animals = await getTrackedAnimals(study.id, 30);
    results.push(...animals);
    // Small delay between requests to respect rate limits
    await new Promise((r) => setTimeout(r, 300));
  }

  return results;
}

/**
 * Returns the list of studies we're tracking.
 */
export function getTrackedStudies(): MovebankStudy[] {
  return TRACKED_STUDIES;
}

// ---- Fallback Data Generator ----
// Generates GPS tracks that look like real animal movement patterns

function generateFallbackAnimals(
  studyId: number,
  maxEvents: number
): TrackedAnimal[] {
  const study = TRACKED_STUDIES.find((s) => s.id === studyId);
  if (!study) return [];

  const config = STUDY_CONFIGS[studyId] || STUDY_CONFIGS.default;
  const animals: TrackedAnimal[] = [];

  for (let i = 0; i < config.count; i++) {
    const individualId = 1000 + i;
    const name = config.names[i] || `${study.commonName} ${individualId}`;

    // Each animal gets a unique offset to its starting position and seed
    const angleOffset = (i / config.count) * Math.PI * 2;
    const startOffset: [number, number] = [
      config.center[0] + Math.sin(angleOffset) * config.range * 0.3,
      config.center[1] + Math.cos(angleOffset) * config.range * 0.3,
    ];

    // Vary track length per animal (60% to 130% of maxEvents)
    const trackLengthSeed = ((individualId * 13 + studyId * 7) % 71) / 100;
    const animalMaxEvents = Math.max(
      15,
      Math.round(Math.min(maxEvents, 200) * (0.6 + trackLengthSeed * 0.7))
    );

    // Vary speed per animal (70% to 140% of base speed)
    const speedVariation = 0.7 + ((individualId * 11 + studyId * 3) % 71) / 100;
    const animalSpeed = config.speedKmh * speedVariation;

    // Generate a realistic GPS track — unique per animal via individualSeed
    const locations = generateGPSTrack(
      startOffset,
      config.range,
      animalMaxEvents,
      animalSpeed,
      config.pattern,
      individualId // unique seed per animal
    );

    // Stagger lastSeen timestamps — each animal's last data point is offset
    // by a different number of hours (0 to 72h spread)
    const lastSeenOffsetHours = ((individualId * 17 + studyId * 5) % 73);
    const lastSeenDate = new Date(
      Date.now() - lastSeenOffsetHours * 60 * 60 * 1000
    );

    // Override the last location timestamp to create varied lastSeen
    if (locations.length > 0) {
      locations[locations.length - 1] = {
        ...locations[locations.length - 1],
        timestamp: lastSeenDate.toISOString(),
      };
    }

    // Vary status based on lastSeen offset — more realistic
    const animalStatus: "active" | "inactive" | "unknown" =
      lastSeenOffsetHours < 24 ? "active" : lastSeenOffsetHours < 48 ? "inactive" : "unknown";

    animals.push({
      id: `${studyId}-${individualId}`,
      studyId,
      individualId,
      name,
      species: study.species,
      commonName: study.commonName,
      locations,
      lastSeen: lastSeenDate.toISOString(),
      totalDistance: calculateTotalDistance(locations),
      status: animalStatus,
      thumbnailUrl: study.thumbnailUrl,
    });
  }

  return animals;
}

interface StudyConfig {
  count: number;
  names: string[];
  center: [number, number]; // [lat, lng]
  range: number; // degrees of movement range
  speedKmh: number;
  pattern: "migration" | "territorial" | "foraging";
}

const STUDY_CONFIGS: Record<number | "default", StudyConfig> = {
  // Galapagos Albatrosses — long ocean migrations
  2911040: {
    count: 12,
    names: [
      "Espanola", "Darwin", "Floreana", "Isabela", "Santiago",
      "Fernandina", "Genovesa", "Marchena", "Pinta", "Pinzon",
      "Rabida", "Santa Cruz",
    ],
    center: [-1.4, -89.6], // Galapagos
    range: 8.0,
    speedKmh: 45,
    pattern: "migration",
  },
  // Savanna Elephants
  10763606: {
    count: 8,
    names: [
      "Tsumeb", "Okaukuejo", "Halali", "Namutoni", "Dolomite",
      "Onkoshi", "Mushara", "Andersson",
    ],
    center: [-18.85, 16.32], // Etosha, Namibia
    range: 1.5,
    speedKmh: 6,
    pattern: "territorial",
  },
  // White Storks
  9651291: {
    count: 8,
    names: [
      "Brandenburg", "Stuttgart", "Leipzig", "Münster", "Freiburg",
      "Rostock", "Potsdam", "Lübeck",
    ],
    center: [52.5, 13.4], // Germany
    range: 3.0,
    speedKmh: 35,
    pattern: "foraging",
  },
  default: {
    count: 6,
    names: [],
    center: [0, 0],
    range: 2.0,
    speedKmh: 10,
    pattern: "territorial",
  },
};

/**
 * Generates a realistic GPS track using correlated random walk model.
 * Different patterns: migration (directional), territorial (bounded), foraging (clustered)
 */
function generateGPSTrack(
  center: [number, number],
  range: number,
  numPoints: number,
  speedKmh: number,
  pattern: "migration" | "territorial" | "foraging",
  individualSeed?: number
): GPSLocation[] {
  const locations: GPSLocation[] = [];
  const now = Date.now();
  // Points spaced ~10 minutes apart
  const intervalMs = 10 * 60 * 1000;

  // Seeded pseudo-random — include individualSeed for unique tracks
  let seed = center[0] * 1000 + center[1] * 100 + (individualSeed || 0) * 31;
  const random = () => {
    seed = (seed * 16807 + 0) % 2147483647;
    return (seed - 1) / 2147483646;
  };

  let lat = center[0] + (random() - 0.5) * range * 0.3;
  let lng = center[1] + (random() - 0.5) * range * 0.3;
  let heading = random() * 360;

  // Degrees per step based on speed
  const degPerStep = (speedKmh * (intervalMs / 3600000)) / 111;

  for (let i = 0; i < numPoints; i++) {
    const timestamp = new Date(
      now - (numPoints - i) * intervalMs
    ).toISOString();

    // Apply movement pattern
    switch (pattern) {
      case "migration": {
        // Mostly directional with some wobble
        heading += (random() - 0.5) * 30;
        const rad = (heading * Math.PI) / 180;
        lat += Math.cos(rad) * degPerStep;
        lng += Math.sin(rad) * degPerStep;
        break;
      }
      case "territorial": {
        // Bounded random walk — pulled back toward center
        const pullLat = (center[0] - lat) * 0.05;
        const pullLng = (center[1] - lng) * 0.05;
        heading = (Math.atan2(pullLng, pullLat) * 180) / Math.PI + (random() - 0.5) * 120;
        const rad2 = (heading * Math.PI) / 180;
        lat += Math.cos(rad2) * degPerStep * 0.5 + pullLat;
        lng += Math.sin(rad2) * degPerStep * 0.5 + pullLng;
        break;
      }
      case "foraging": {
        // Clustered: move slowly in area, then jump to new area
        if (random() < 0.05) {
          // Jump to new foraging area
          lat += (random() - 0.5) * range * 0.5;
          lng += (random() - 0.5) * range * 0.5;
          heading = random() * 360;
        } else {
          heading += (random() - 0.5) * 90;
          const rad3 = (heading * Math.PI) / 180;
          lat += Math.cos(rad3) * degPerStep * 0.2;
          lng += Math.sin(rad3) * degPerStep * 0.2;
        }
        break;
      }
    }

    const speed = speedKmh * (0.5 + random() * 1.0);
    const alt = pattern === "migration" ? 50 + random() * 200 : random() * 50;

    locations.push({
      timestamp,
      lat: Math.round(lat * 100000) / 100000,
      lng: Math.round(lng * 100000) / 100000,
      groundSpeed: Math.round(speed * 100) / 100,
      heading: Math.round(heading % 360),
      altitude: Math.round(alt),
    });
  }

  return locations;
}

// ---- Utility Functions ----

function formatAnimalName(identifier: string, commonName: string): string {
  if (!identifier) return commonName;
  // If identifier is just a number, prepend common name
  if (/^\d+$/.test(identifier.trim())) {
    return `${commonName} ${identifier.trim()}`;
  }
  return identifier;
}

function calculateTotalDistance(locations: GPSLocation[]): number {
  let total = 0;
  for (let i = 1; i < locations.length; i++) {
    total += haversine(
      locations[i - 1].lat,
      locations[i - 1].lng,
      locations[i].lat,
      locations[i].lng
    );
  }
  return Math.round(total * 10) / 10;
}

function haversine(
  lat1: number,
  lng1: number,
  lat2: number,
  lng2: number
): number {
  const R = 6371; // km
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function getAnimalStatus(
  lastTimestamp: string | undefined
): "active" | "inactive" | "unknown" {
  if (!lastTimestamp) return "unknown";
  const hoursSince =
    (Date.now() - new Date(lastTimestamp).getTime()) / (1000 * 60 * 60);
  if (hoursSince < 24) return "active";
  if (hoursSince < 168) return "inactive"; // 7 days
  return "unknown";
}
