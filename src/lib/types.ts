// Types for the anomaly detection pipeline

export interface TelemetryEvent {
  tourist_id: string;
  lat: number;
  lng: number;
  timestamp: string;
  heart_rate?: number;
  battery_level?: number;
  network_status?: string;
  panic_button?: boolean;
  accuracy?: number;
  altitude?: number;
}

export interface Alert {
  alert_id: string;
  tourist_id: string;
  anomaly_type: AnomalyType;
  alert_level: AlertLevel;
  confidence_score: number;
  location: { lat: number; lng: number };
  timestamp: string;
  raw_evidence: Record<string, unknown>;
  model_version: string;
  models_used: string[];
  features_extracted: number;
}

export type AnomalyType =
  | "PANIC"
  | "HEART_RATE_HIGH"
  | "HEART_RATE_LOW"
  | "BATTERY_CRITICAL"
  | "VELOCITY_CRITICAL"
  | "VELOCITY_HIGH"
  | "PROLONGED_INACTIVITY"
  | "ROUTE_DEVIATION"
  | "GEOFENCE_BREACH"
  | "NORMAL";

export type AlertLevel = "CRITICAL" | "WARNING" | "INFO" | "NORMAL";

export interface DetectionResult {
  status: string;
  anomaly_count: number;
  alerts: Alert[];
  telemetry_processed: number;
  processing_time_ms: number;
  pipeline_stages: PipelineStage[];
}

export interface PipelineStage {
  name: string;
  status: "completed" | "running" | "pending";
  duration_ms: number;
  details?: string;
}

export interface DashboardStats {
  total_events_processed: number;
  total_anomalies: number;
  critical_alerts: number;
  warning_alerts: number;
  detection_rate: number;
  avg_processing_time: number;
}
