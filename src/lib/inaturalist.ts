// iNaturalist API Client
// Fetches real-time endangered wildlife observations from the iNaturalist API
// No API key required — public API with rate limits

import {
  WildlifeSighting,
  ConservationStatus,
  INatApiResponse,
  INatObservation,
} from "./types";

const INATURALIST_API = "https://api.inaturalist.org/v1";

// Map iNaturalist status codes to our conservation status
function mapConservationStatus(
  status?: string | null,
  iucn?: number | null
): { status: ConservationStatus; level: number } {
  if (iucn) {
    if (iucn >= 50) return { status: "critically_endangered", level: 50 };
    if (iucn >= 40) return { status: "endangered", level: 40 };
    if (iucn >= 30) return { status: "vulnerable", level: 30 };
    if (iucn >= 20) return { status: "near_threatened", level: 20 };
    return { status: "least_concern", level: 10 };
  }

  if (status) {
    const s = status.toLowerCase();
    if (s === "cr" || s === "critically_endangered") return { status: "critically_endangered", level: 50 };
    if (s === "en" || s === "endangered") return { status: "endangered", level: 40 };
    if (s === "vu" || s === "vulnerable") return { status: "vulnerable", level: 30 };
    if (s === "nt" || s === "near_threatened") return { status: "near_threatened", level: 20 };
    if (s === "lc" || s === "least_concern") return { status: "least_concern", level: 10 };
  }

  return { status: "unknown", level: 0 };
}

// Get photo URL in medium size
function getPhotoUrl(obs: INatObservation): string | null {
  if (!obs.photos || obs.photos.length === 0) return null;
  const url = obs.photos[0].url;
  if (!url) return null;
  // iNaturalist returns square thumbnails, convert to medium
  return url.replace("square", "medium");
}

// Parse lat/lng from iNaturalist observation
function parseLocation(obs: INatObservation): { lat: number; lng: number } | null {
  if (obs.geojson && obs.geojson.coordinates) {
    return { lat: obs.geojson.coordinates[1], lng: obs.geojson.coordinates[0] };
  }
  if (obs.location) {
    const parts = obs.location.split(",");
    if (parts.length === 2) {
      return { lat: parseFloat(parts[0]), lng: parseFloat(parts[1]) };
    }
  }
  return null;
}

// Convert iNaturalist observation to our WildlifeSighting type
function toWildlifeSighting(obs: INatObservation): WildlifeSighting | null {
  const location = parseLocation(obs);
  if (!location) return null;

  const taxon = obs.taxon;
  if (!taxon) return null;

  const cs = mapConservationStatus(
    taxon.conservation_status?.status,
    taxon.conservation_status?.iucn
  );

  return {
    id: `inat_${obs.id}`,
    species_name: taxon.name || "Unknown",
    common_name: taxon.preferred_common_name || obs.species_guess || taxon.name || "Unknown",
    lat: location.lat,
    lng: location.lng,
    observed_at: obs.time_observed_at || obs.observed_on || new Date().toISOString(),
    place_name: obs.place_guess || "Unknown Location",
    conservation_status: cs.status,
    iucn_level: cs.level,
    photo_url: getPhotoUrl(obs),
    observer: obs.user?.login || "anonymous",
    quality_grade: obs.quality_grade,
    iconic_taxon: taxon.iconic_taxon_name || "Unknown",
    taxon_id: taxon.id,
    positional_accuracy: obs.positional_accuracy,
    captive: obs.captive || false,
  };
}

// ============================================================
// MAIN FETCH FUNCTIONS
// ============================================================

export type TaxonGroup = "Mammalia" | "Aves" | "Reptilia" | "Amphibia" | "Actinopterygii" | "all";

export interface FetchOptions {
  taxon_group?: TaxonGroup;
  threatened_only?: boolean;
  per_page?: number;
  quality_grade?: "research" | "needs_id" | "any";
  place_id?: number | null;  // iNaturalist place ID for region filtering
}

/**
 * Fetch recent wildlife observations from iNaturalist.
 * Returns real-time data — new observations appear as people around the world log sightings.
 */
export async function fetchWildlifeSightings(
  options: FetchOptions = {}
): Promise<WildlifeSighting[]> {
  const {
    taxon_group = "all",
    threatened_only = true,
    per_page = 30,
    quality_grade = "research",
    place_id = null,
  } = options;

  const params = new URLSearchParams({
    per_page: String(Math.min(per_page, 50)),
    order: "desc",
    order_by: "created_at",
    quality_grade: quality_grade === "any" ? "" : quality_grade,
  });

  if (threatened_only) {
    params.set("threatened", "true");
  }

  if (taxon_group !== "all") {
    params.set("iconic_taxa", taxon_group);
  }

  if (place_id) {
    params.set("place_id", String(place_id));
  }

  // Remove empty params
  for (const [key, value] of Array.from(params.entries())) {
    if (!value) params.delete(key);
  }

  const url = `${INATURALIST_API}/observations?${params.toString()}`;

  const response = await fetch(url, {
    headers: {
      "Accept": "application/json",
    },
    // Revalidate every 60 seconds on the server side
    next: { revalidate: 60 },
  });

  if (!response.ok) {
    throw new Error(`iNaturalist API error: ${response.status} ${response.statusText}`);
  }

  const data: INatApiResponse = await response.json();

  const sightings: WildlifeSighting[] = [];
  for (const obs of data.results) {
    const sighting = toWildlifeSighting(obs);
    if (sighting) {
      sightings.push(sighting);
    }
  }

  return sightings;
}

/**
 * Fetch sightings for specific taxon groups that are commonly endangered.
 * Returns a diverse mix for demo purposes.
 */
export async function fetchDiverseSightings(
  per_page: number = 30
): Promise<WildlifeSighting[]> {
  // Fetch mammals and birds in parallel for diversity
  const perGroup = Math.ceil(per_page / 2);

  try {
    const [mammals, birds] = await Promise.all([
      fetchWildlifeSightings({
        taxon_group: "Mammalia",
        threatened_only: true,
        per_page: perGroup,
      }),
      fetchWildlifeSightings({
        taxon_group: "Aves",
        threatened_only: true,
        per_page: perGroup,
      }),
    ]);

    const combined = [...mammals, ...birds];
    // Sort by observation time, most recent first
    combined.sort(
      (a, b) => new Date(b.observed_at).getTime() - new Date(a.observed_at).getTime()
    );

    return combined.slice(0, per_page);
  } catch (error) {
    // If parallel fetch fails, try a single request
    console.error("Diverse fetch failed, falling back to single request:", error);
    return fetchWildlifeSightings({ per_page, threatened_only: true });
  }
}

// ============================================================
// REGION PRESETS (iNaturalist place IDs)
// ============================================================

export const REGIONS: Record<string, { label: string; place_id: number; description: string }> = {
  global: { label: "Global", place_id: 0, description: "Worldwide observations" },
  africa: { label: "Africa", place_id: 97392, description: "African wildlife" },
  south_asia: { label: "South Asia", place_id: 97395, description: "India, Nepal, Sri Lanka" },
  southeast_asia: { label: "SE Asia", place_id: 97394, description: "Indonesia, Malaysia, Thailand" },
  south_america: { label: "South America", place_id: 97389, description: "Amazon, Andes, Pantanal" },
  australia: { label: "Australia", place_id: 6744, description: "Australian wildlife" },
  north_america: { label: "North America", place_id: 97394, description: "US, Canada, Mexico" },
};
