// Types for the Sentinel Wildlife Anomaly Detection Pipeline
// Data source: iNaturalist API (real-time endangered species observations)

// ============================================================
// WILDLIFE SIGHTING (input data)
// ============================================================

export interface WildlifeSighting {
  id: string;                           // Unique sighting ID
  species_name: string;                 // Scientific name (e.g. "Panthera tigris")
  common_name: string;                  // Common name (e.g. "Tiger")
  lat: number;                          // Latitude
  lng: number;                          // Longitude
  observed_at: string;                  // ISO timestamp of observation
  place_name: string;                   // Human-readable location (e.g. "Ranthambore NP, India")
  conservation_status: ConservationStatus;
  iucn_level: number;                   // IUCN numeric code (10=LC, 20=NT, 30=VU, 40=EN, 50=CR)
  photo_url: string | null;             // Observation photo URL
  observer: string;                     // Username who observed
  quality_grade: "research" | "needs_id" | "casual";
  iconic_taxon: string;                 // Mammalia, Aves, Reptilia, etc.
  taxon_id: number;                     // iNaturalist taxon ID
  positional_accuracy: number | null;   // GPS accuracy in meters
  captive: boolean;                     // Whether the animal is captive
}

export type ConservationStatus =
  | "critically_endangered"   // CR — IUCN 50
  | "endangered"              // EN — IUCN 40
  | "vulnerable"              // VU — IUCN 30
  | "near_threatened"         // NT — IUCN 20
  | "least_concern"           // LC — IUCN 10
  | "unknown";

// ============================================================
// ANOMALY TYPES (wildlife-specific)
// ============================================================

export type AnomalyType =
  | "RANGE_ANOMALY"           // Species observed far outside known range
  | "TEMPORAL_ANOMALY"        // Sighting at unusual time (nocturnal animal seen at noon)
  | "CLUSTER_ANOMALY"         // Unusual density of sightings in area
  | "RARITY_ANOMALY"          // Extremely rare species (CR) in unexpected area
  | "CAPTIVE_ESCAPE"          // Captive species sighted in wild area
  | "MISIDENTIFICATION"       // Statistical outlier suggesting ID error
  | "HABITAT_MISMATCH"        // Species in wrong habitat type
  | "POACHING_INDICATOR"      // Pattern suggests potential poaching activity
  | "NORMAL";

export type AlertLevel = "CRITICAL" | "WARNING" | "INFO" | "NORMAL";

// ============================================================
// ALERT (output from detection pipeline)
// ============================================================

export interface Alert {
  alert_id: string;
  sighting_id: string;
  species_name: string;
  common_name: string;
  anomaly_type: AnomalyType;
  alert_level: AlertLevel;
  confidence_score: number;
  location: { lat: number; lng: number };
  place_name: string;
  timestamp: string;
  conservation_status: ConservationStatus;
  photo_url: string | null;
  raw_evidence: Record<string, unknown>;
  model_version: string;
  models_used: string[];
  features_extracted: number;
}

// ============================================================
// PIPELINE RESULT
// ============================================================

export interface DetectionResult {
  status: string;
  anomaly_count: number;
  alerts: Alert[];
  sightings_processed: number;
  processing_time_ms: number;
  pipeline_stages: PipelineStage[];
  threshold_used?: number;
  data_source: "inaturalist_live" | "fallback_generated";
  species_analyzed: number;
  region?: string;
}

export interface PipelineStage {
  name: string;
  status: "completed" | "running" | "pending";
  duration_ms: number;
  details?: string;
}

// ============================================================
// DASHBOARD STATE
// ============================================================

export interface DashboardStats {
  total_sightings_processed: number;
  total_anomalies: number;
  critical_alerts: number;
  warning_alerts: number;
  detection_rate: number;
  avg_processing_time: number;
  species_count: number;
}

// ============================================================
// iNATURALIST API RESPONSE TYPES
// ============================================================

export interface INatObservation {
  id: number;
  species_guess: string | null;
  taxon: INatTaxon | null;
  location: string | null;             // "lat,lng" string
  geojson: { type: string; coordinates: [number, number] } | null;
  observed_on: string | null;
  time_observed_at: string | null;
  place_guess: string | null;
  quality_grade: "research" | "needs_id" | "casual";
  photos: INatPhoto[];
  user: { login: string } | null;
  captive: boolean;
  positional_accuracy: number | null;
}

export interface INatTaxon {
  id: number;
  name: string;
  preferred_common_name?: string;
  iconic_taxon_name?: string;
  threatened?: boolean;
  conservation_status?: {
    status: string;
    status_name: string;
    authority: string;
    iucn: number;
  } | null;
}

export interface INatPhoto {
  url: string;
  attribution?: string;
}

export interface INatApiResponse {
  total_results: number;
  page: number;
  per_page: number;
  results: INatObservation[];
}
