// Fallback Wildlife Sighting Generator
// Generates realistic sighting data when iNaturalist API is unavailable
// Includes: normal sightings, range anomalies, temporal anomalies, rarity spikes,
//           cluster anomalies, captive escapes, and habitat mismatches

import { WildlifeSighting, ConservationStatus } from './types';

// ============================================================
// SPECIES DATABASE — real species with approximate range data
// ============================================================

interface SpeciesProfile {
  scientific: string;
  common: string;
  status: ConservationStatus;
  iucn: number;
  taxon: string;
  taxon_id: number;
  // Approximate range center + radius
  range: { lat: number; lng: number; radius_deg: number };
  // Photo URL (iNaturalist static assets, CC-licensed)
  photo: string | null;
}

const SPECIES: SpeciesProfile[] = [
  // Mammals — Critically Endangered
  {
    scientific: 'Panthera tigris',
    common: 'Tiger',
    status: 'endangered',
    iucn: 40,
    taxon: 'Mammalia',
    taxon_id: 41944,
    range: { lat: 22, lng: 80, radius_deg: 15 },
    photo: null,
  },
  {
    scientific: 'Diceros bicornis',
    common: 'Black Rhinoceros',
    status: 'critically_endangered',
    iucn: 50,
    taxon: 'Mammalia',
    taxon_id: 43352,
    range: { lat: -5, lng: 30, radius_deg: 15 },
    photo: null,
  },
  {
    scientific: 'Gorilla beringei',
    common: 'Eastern Gorilla',
    status: 'critically_endangered',
    iucn: 50,
    taxon: 'Mammalia',
    taxon_id: 43390,
    range: { lat: -1.5, lng: 29.5, radius_deg: 3 },
    photo: null,
  },
  {
    scientific: 'Pongo pygmaeus',
    common: 'Bornean Orangutan',
    status: 'critically_endangered',
    iucn: 50,
    taxon: 'Mammalia',
    taxon_id: 43578,
    range: { lat: 1, lng: 112, radius_deg: 5 },
    photo: null,
  },
  {
    scientific: 'Ailuropoda melanoleuca',
    common: 'Giant Panda',
    status: 'vulnerable',
    iucn: 30,
    taxon: 'Mammalia',
    taxon_id: 41918,
    range: { lat: 31, lng: 104, radius_deg: 4 },
    photo: null,
  },
  // Mammals — Endangered
  {
    scientific: 'Elephas maximus',
    common: 'Asian Elephant',
    status: 'endangered',
    iucn: 40,
    taxon: 'Mammalia',
    taxon_id: 43353,
    range: { lat: 15, lng: 80, radius_deg: 20 },
    photo: null,
  },
  {
    scientific: 'Panthera uncia',
    common: 'Snow Leopard',
    status: 'vulnerable',
    iucn: 30,
    taxon: 'Mammalia',
    taxon_id: 41970,
    range: { lat: 38, lng: 75, radius_deg: 15 },
    photo: null,
  },
  {
    scientific: 'Lycaon pictus',
    common: 'African Wild Dog',
    status: 'endangered',
    iucn: 40,
    taxon: 'Mammalia',
    taxon_id: 42096,
    range: { lat: -10, lng: 30, radius_deg: 20 },
    photo: null,
  },
  // Birds
  {
    scientific: 'Gymnogyps californianus',
    common: 'California Condor',
    status: 'critically_endangered',
    iucn: 50,
    taxon: 'Aves',
    taxon_id: 4856,
    range: { lat: 35, lng: -118, radius_deg: 5 },
    photo: null,
  },
  {
    scientific: 'Spheniscus demersus',
    common: 'African Penguin',
    status: 'endangered',
    iucn: 40,
    taxon: 'Aves',
    taxon_id: 4039,
    range: { lat: -33, lng: 18, radius_deg: 5 },
    photo: null,
  },
  {
    scientific: 'Leucopsar rothschildi',
    common: 'Bali Myna',
    status: 'critically_endangered',
    iucn: 50,
    taxon: 'Aves',
    taxon_id: 19867,
    range: { lat: -8.2, lng: 114.5, radius_deg: 1 },
    photo: null,
  },
  // Reptiles
  {
    scientific: 'Chelonia mydas',
    common: 'Green Sea Turtle',
    status: 'endangered',
    iucn: 40,
    taxon: 'Reptilia',
    taxon_id: 39681,
    range: { lat: 10, lng: -60, radius_deg: 40 },
    photo: null,
  },
  {
    scientific: 'Gavialis gangeticus',
    common: 'Gharial',
    status: 'critically_endangered',
    iucn: 50,
    taxon: 'Reptilia',
    taxon_id: 39773,
    range: { lat: 26, lng: 82, radius_deg: 5 },
    photo: null,
  },
  // Vulnerable
  {
    scientific: 'Hippopotamus amphibius',
    common: 'Hippopotamus',
    status: 'vulnerable',
    iucn: 30,
    taxon: 'Mammalia',
    taxon_id: 43359,
    range: { lat: -5, lng: 30, radius_deg: 20 },
    photo: null,
  },
  {
    scientific: 'Ursus maritimus',
    common: 'Polar Bear',
    status: 'vulnerable',
    iucn: 30,
    taxon: 'Mammalia',
    taxon_id: 41955,
    range: { lat: 75, lng: -30, radius_deg: 20 },
    photo: null,
  },
];

// Observer usernames for realism
const OBSERVERS = [
  'wildlife_watcher', 'field_bio_22', 'naturalist_k', 'safari_guide_jm',
  'birdnerd99', 'eco_ranger', 'conservation_intern', 'jungle_jay',
  'marine_biologist', 'expedition_lead', 'park_ranger_ns', 'photonaturalist',
];

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

function randomBetween(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

function randomPick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function generateSightingId(): string {
  return 'gen_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now().toString(36);
}

function generateTimestamp(hoursAgo: number): string {
  const now = new Date();
  now.setHours(now.getHours() - hoursAgo);
  now.setMinutes(now.getMinutes() - Math.floor(Math.random() * 60));
  return now.toISOString();
}

// ============================================================
// SIGHTING GENERATORS BY SCENARIO
// ============================================================

/**
 * Normal sighting: species observed within its expected range with good data quality.
 */
function generateNormalSighting(species: SpeciesProfile, hoursAgo: number): WildlifeSighting {
  const { range } = species;
  const lat = range.lat + randomBetween(-range.radius_deg * 0.5, range.radius_deg * 0.5);
  const lng = range.lng + randomBetween(-range.radius_deg * 0.5, range.radius_deg * 0.5);

  return {
    id: generateSightingId(),
    species_name: species.scientific,
    common_name: species.common,
    lat,
    lng,
    observed_at: generateTimestamp(hoursAgo),
    place_name: `Near ${species.common} habitat`,
    conservation_status: species.status,
    iucn_level: species.iucn,
    photo_url: species.photo,
    observer: randomPick(OBSERVERS),
    quality_grade: 'research',
    iconic_taxon: species.taxon,
    taxon_id: species.taxon_id,
    positional_accuracy: Math.round(randomBetween(5, 30)),
    captive: false,
  };
}

/**
 * Range anomaly: species observed far outside its expected range.
 * Could indicate escaped captive, range expansion, or misidentification.
 */
function generateRangeAnomaly(species: SpeciesProfile, hoursAgo: number): WildlifeSighting {
  const { range } = species;
  // Place sighting 2-4x outside expected range
  const offsetLat = randomBetween(range.radius_deg * 2, range.radius_deg * 4) * (Math.random() > 0.5 ? 1 : -1);
  const offsetLng = randomBetween(range.radius_deg * 2, range.radius_deg * 4) * (Math.random() > 0.5 ? 1 : -1);
  const lat = Math.max(-85, Math.min(85, range.lat + offsetLat));
  const lng = ((range.lng + offsetLng + 180) % 360) - 180;

  const places = [
    'Unexpected location — outside known range',
    'Urban park — far from natural habitat',
    'Agricultural area — no prior records',
    'Remote coastal zone',
    'Highway wildlife corridor',
  ];

  return {
    id: generateSightingId(),
    species_name: species.scientific,
    common_name: species.common,
    lat,
    lng,
    observed_at: generateTimestamp(hoursAgo),
    place_name: randomPick(places),
    conservation_status: species.status,
    iucn_level: species.iucn,
    photo_url: species.photo,
    observer: randomPick(OBSERVERS),
    quality_grade: Math.random() > 0.5 ? 'research' : 'needs_id',
    iconic_taxon: species.taxon,
    taxon_id: species.taxon_id,
    positional_accuracy: Math.round(randomBetween(15, 200)),
    captive: false,
  };
}

/**
 * Cluster anomaly: multiple sightings of same species in very tight area.
 * Could indicate feeding event, breeding, or data artifacts.
 */
function generateClusterSightings(species: SpeciesProfile, count: number, hoursAgo: number): WildlifeSighting[] {
  const { range } = species;
  const clusterLat = range.lat + randomBetween(-range.radius_deg * 0.3, range.radius_deg * 0.3);
  const clusterLng = range.lng + randomBetween(-range.radius_deg * 0.3, range.radius_deg * 0.3);

  const sightings: WildlifeSighting[] = [];
  for (let i = 0; i < count; i++) {
    sightings.push({
      id: generateSightingId(),
      species_name: species.scientific,
      common_name: species.common,
      lat: clusterLat + randomBetween(-0.02, 0.02),
      lng: clusterLng + randomBetween(-0.02, 0.02),
      observed_at: generateTimestamp(hoursAgo + randomBetween(0, 2)),
      place_name: `${species.common} observation cluster`,
      conservation_status: species.status,
      iucn_level: species.iucn,
      photo_url: species.photo,
      observer: randomPick(OBSERVERS),
      quality_grade: 'research',
      iconic_taxon: species.taxon,
      taxon_id: species.taxon_id,
      positional_accuracy: Math.round(randomBetween(5, 20)),
      captive: false,
    });
  }
  return sightings;
}

/**
 * Captive escape scenario: captive animal spotted in unexpected wild area.
 */
function generateCaptiveEscape(species: SpeciesProfile, hoursAgo: number): WildlifeSighting {
  // Captive animals are often found near urban areas, zoos, etc.
  const urbanLocations = [
    { lat: 40.78, lng: -73.97, name: 'Central Park, New York' },
    { lat: 51.53, lng: -0.15, name: 'Regent\'s Park, London' },
    { lat: -33.86, lng: 151.21, name: 'Taronga area, Sydney' },
    { lat: 48.84, lng: 2.35, name: 'Jardin des Plantes, Paris' },
  ];
  const loc = randomPick(urbanLocations);

  return {
    id: generateSightingId(),
    species_name: species.scientific,
    common_name: species.common,
    lat: loc.lat + randomBetween(-0.05, 0.05),
    lng: loc.lng + randomBetween(-0.05, 0.05),
    observed_at: generateTimestamp(hoursAgo),
    place_name: loc.name,
    conservation_status: species.status,
    iucn_level: species.iucn,
    photo_url: species.photo,
    observer: randomPick(OBSERVERS),
    quality_grade: 'casual',
    iconic_taxon: species.taxon,
    taxon_id: species.taxon_id,
    positional_accuracy: Math.round(randomBetween(20, 150)),
    captive: true,
  };
}

/**
 * Temporal anomaly: sighting with unusual timing gaps from other sightings.
 */
function generateTemporalAnomaly(species: SpeciesProfile): WildlifeSighting {
  const { range } = species;
  const lat = range.lat + randomBetween(-range.radius_deg * 0.4, range.radius_deg * 0.4);
  const lng = range.lng + randomBetween(-range.radius_deg * 0.4, range.radius_deg * 0.4);

  // Set observation far in the past relative to others (30-60 days ago)
  return {
    id: generateSightingId(),
    species_name: species.scientific,
    common_name: species.common,
    lat,
    lng,
    observed_at: generateTimestamp(randomBetween(720, 1440)), // 30-60 days ago
    place_name: `${species.common} — temporally isolated sighting`,
    conservation_status: species.status,
    iucn_level: species.iucn,
    photo_url: species.photo,
    observer: randomPick(OBSERVERS),
    quality_grade: 'research',
    iconic_taxon: species.taxon,
    taxon_id: species.taxon_id,
    positional_accuracy: Math.round(randomBetween(5, 50)),
    captive: false,
  };
}

// ============================================================
// MAIN GENERATOR
// ============================================================

export type FallbackScenario =
  | 'normal'
  | 'range_anomalies'
  | 'cluster_event'
  | 'captive_escapes'
  | 'mixed';

/**
 * Generate a batch of realistic wildlife sighting data.
 * Used as fallback when iNaturalist API is unavailable.
 */
export function generateWildlifeBatch(
  scenario: FallbackScenario = 'mixed',
  count: number = 30,
): WildlifeSighting[] {
  const sightings: WildlifeSighting[] = [];

  if (scenario === 'normal') {
    for (let i = 0; i < count; i++) {
      const species = randomPick(SPECIES);
      sightings.push(generateNormalSighting(species, randomBetween(0, 48)));
    }
  } else if (scenario === 'range_anomalies') {
    // 60% normal, 40% range anomalies
    const normalCount = Math.floor(count * 0.6);
    const anomalyCount = count - normalCount;
    for (let i = 0; i < normalCount; i++) {
      sightings.push(generateNormalSighting(randomPick(SPECIES), randomBetween(0, 48)));
    }
    for (let i = 0; i < anomalyCount; i++) {
      sightings.push(generateRangeAnomaly(randomPick(SPECIES), randomBetween(0, 24)));
    }
  } else if (scenario === 'cluster_event') {
    // 50% normal, 50% tight cluster
    const normalCount = Math.floor(count * 0.5);
    for (let i = 0; i < normalCount; i++) {
      sightings.push(generateNormalSighting(randomPick(SPECIES), randomBetween(0, 48)));
    }
    const clusterSpecies = randomPick(SPECIES);
    sightings.push(...generateClusterSightings(clusterSpecies, count - normalCount, randomBetween(0, 6)));
  } else if (scenario === 'captive_escapes') {
    // 70% normal, 30% captive escapes
    const normalCount = Math.floor(count * 0.7);
    for (let i = 0; i < normalCount; i++) {
      sightings.push(generateNormalSighting(randomPick(SPECIES), randomBetween(0, 48)));
    }
    for (let i = 0; i < count - normalCount; i++) {
      const species = SPECIES.filter(s => s.iucn >= 40);
      sightings.push(generateCaptiveEscape(randomPick(species), randomBetween(0, 24)));
    }
  } else {
    // Mixed scenario — the default demo mode
    // Good balance of normal + various anomaly types
    const normalCount = Math.floor(count * 0.45);
    const rangeAnomalyCount = Math.floor(count * 0.15);
    const clusterCount = Math.floor(count * 0.15);
    const captiveCount = Math.floor(count * 0.1);
    const temporalCount = count - normalCount - rangeAnomalyCount - clusterCount - captiveCount;

    // Normal sightings
    for (let i = 0; i < normalCount; i++) {
      sightings.push(generateNormalSighting(randomPick(SPECIES), randomBetween(0, 48)));
    }

    // Range anomalies
    for (let i = 0; i < rangeAnomalyCount; i++) {
      const rareSpecies = SPECIES.filter(s => s.iucn >= 40);
      sightings.push(generateRangeAnomaly(randomPick(rareSpecies), randomBetween(0, 24)));
    }

    // Cluster event
    const clusterSpecies = randomPick(SPECIES);
    sightings.push(...generateClusterSightings(clusterSpecies, clusterCount, randomBetween(0, 6)));

    // Captive escapes
    for (let i = 0; i < captiveCount; i++) {
      const crSpecies = SPECIES.filter(s => s.iucn >= 40);
      sightings.push(generateCaptiveEscape(randomPick(crSpecies), randomBetween(0, 12)));
    }

    // Temporal anomalies
    for (let i = 0; i < temporalCount; i++) {
      sightings.push(generateTemporalAnomaly(randomPick(SPECIES)));
    }
  }

  // Sort by observation time, most recent first
  sightings.sort((a, b) => new Date(b.observed_at).getTime() - new Date(a.observed_at).getTime());

  return sightings;
}
