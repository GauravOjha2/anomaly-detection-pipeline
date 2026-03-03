"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Brain,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Code2,
  Download,
  Gauge,
  History,
  Loader2,
  MapPin,
  Pause,
  Play,
  Radio,
  RefreshCw,
  Settings2,
  Shield,
  Syringe,
  Timer,
  Zap,
} from "lucide-react";
import { DetectionResult, Alert, TelemetryEvent, AnomalyType, AlertLevel } from "@/lib/types";
import { Scenario } from "@/lib/mock-data";

// ============================================================
// CONSTANTS
// ============================================================

const SCENARIOS: { value: Scenario; label: string; description: string }[] = [
  { value: "mixed", label: "Mixed Scenarios", description: "Realistic mix of normal + anomalous patterns" },
  { value: "normal", label: "Normal Activity", description: "Standard tourist walking patterns" },
  { value: "emergency", label: "Emergency", description: "Escalating speed, panic button, high heart rate" },
  { value: "health_anomaly", label: "Health Anomaly", description: "Gradual heart rate spike to critical" },
  { value: "device_failure", label: "Device Failure", description: "Battery drain, GPS degradation" },
  { value: "extreme", label: "Extreme", description: "All sensors in anomalous ranges" },
];

const SEVERITY_CONFIG: Record<AlertLevel, { color: string; bg: string; border: string; icon: typeof AlertTriangle }> = {
  CRITICAL: { color: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/20", icon: AlertTriangle },
  WARNING: { color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20", icon: Shield },
  INFO: { color: "text-blue-400", bg: "bg-blue-500/10", border: "border-blue-500/20", icon: Radio },
  NORMAL: { color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/20", icon: CheckCircle2 },
};

const ANOMALY_LABELS: Record<AnomalyType, string> = {
  PANIC: "Panic Button Activated",
  HEART_RATE_HIGH: "Critical Heart Rate (High)",
  HEART_RATE_LOW: "Critical Heart Rate (Low)",
  BATTERY_CRITICAL: "Battery Critical",
  VELOCITY_CRITICAL: "Extreme Velocity",
  VELOCITY_HIGH: "High Velocity",
  PROLONGED_INACTIVITY: "Prolonged Inactivity",
  ROUTE_DEVIATION: "Route Deviation",
  GEOFENCE_BREACH: "Geofence Breach",
  NORMAL: "Normal",
};

const STREAMING_INTERVALS = [
  { value: 3000, label: "3s" },
  { value: 5000, label: "5s" },
  { value: 10000, label: "10s" },
];

// ============================================================
// INJECTOR TEMPLATE
// ============================================================

function generateInjectorTemplate(): TelemetryEvent[] {
  const now = new Date();
  return [
    {
      tourist_id: "INJECT-001",
      lat: 39.9042, lng: 116.4074,
      timestamp: new Date(now.getTime() - 8 * 60000).toISOString(),
      heart_rate: 72, battery_level: 85, network_status: "connected",
      panic_button: false, accuracy: 5, altitude: 45,
    },
    {
      tourist_id: "INJECT-001",
      lat: 39.9045, lng: 116.4078,
      timestamp: new Date(now.getTime() - 6 * 60000).toISOString(),
      heart_rate: 75, battery_level: 83, network_status: "connected",
      panic_button: false, accuracy: 4, altitude: 46,
    },
    {
      tourist_id: "INJECT-001",
      lat: 39.9048, lng: 116.4082,
      timestamp: new Date(now.getTime() - 4 * 60000).toISOString(),
      heart_rate: 110, battery_level: 80, network_status: "connected",
      panic_button: false, accuracy: 6, altitude: 45,
    },
    {
      tourist_id: "INJECT-001",
      lat: 39.9120, lng: 116.4200,
      timestamp: new Date(now.getTime() - 2 * 60000).toISOString(),
      heart_rate: 155, battery_level: 78, network_status: "weak",
      panic_button: false, accuracy: 15, altitude: 44,
    },
    {
      tourist_id: "INJECT-001",
      lat: 39.9250, lng: 116.4350,
      timestamp: new Date(now.getTime()).toISOString(),
      heart_rate: 180, battery_level: 5, network_status: "disconnected",
      panic_button: true, accuracy: 50, altitude: 42,
    },
  ];
}

// ============================================================
// HISTORY ENTRY TYPE
// ============================================================

interface HistoryEntry {
  id: string;
  timestamp: Date;
  scenario: string;
  source: string;
  anomaly_count: number;
  processing_time_ms: number;
  alerts_critical: number;
  alerts_warning: number;
  events_processed: number;
}

// ============================================================
// MAIN DASHBOARD COMPONENT
// ============================================================

type InputMode = "scenario" | "injector";

export default function DashboardPage() {
  const [mode, setMode] = useState<InputMode>("scenario");
  const [scenario, setScenario] = useState<Scenario>("mixed");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Injector state
  const [injectorOpen, setInjectorOpen] = useState(false);
  const [injectorJson, setInjectorJson] = useState(() =>
    JSON.stringify(generateInjectorTemplate(), null, 2)
  );
  const [injectorError, setInjectorError] = useState<string | null>(null);

  // Threshold slider
  const [threshold, setThreshold] = useState(0.45);

  // Streaming mode
  const [streaming, setStreaming] = useState(false);
  const [streamInterval, setStreamInterval] = useState(5000);
  const streamRef = useRef<NodeJS.Timeout | null>(null);
  const [streamCount, setStreamCount] = useState(0);

  // Detection history
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  // Settings panel
  const [showSettings, setShowSettings] = useState(false);

  // Run detection
  const runDetection = useCallback(async (isStream = false) => {
    if (!isStream) setLoading(true);
    setSelectedAlert(null);
    setInjectorError(null);
    setError(null);

    try {
      let body: Record<string, unknown>;

      if (mode === "injector") {
        let parsed: unknown;
        try {
          parsed = JSON.parse(injectorJson);
        } catch {
          setInjectorError("Invalid JSON. Check your syntax.");
          setLoading(false);
          return;
        }

        if (!Array.isArray(parsed) || parsed.length < 2) {
          setInjectorError("Telemetry must be an array with at least 2 events.");
          setLoading(false);
          return;
        }

        for (let i = 0; i < parsed.length; i++) {
          const evt = parsed[i] as Record<string, unknown>;
          if (typeof evt.lat !== "number" || typeof evt.lng !== "number" || !evt.timestamp || !evt.tourist_id) {
            setInjectorError(`Event ${i}: must have tourist_id (string), lat (number), lng (number), timestamp (string).`);
            setLoading(false);
            return;
          }
        }

        body = { telemetry: parsed, threshold };
      } else {
        body = { scenario, count: 30, tourist_count: 5, threshold };
      }

      const res = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const errData = await res.json();
        setError(errData.error || "Detection failed");
        setLoading(false);
        return;
      }

      const data = await res.json();
      setResult(data);

      // Add to history
      const entry: HistoryEntry = {
        id: `run_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
        timestamp: new Date(),
        scenario: mode === "injector" ? "Custom Injection" : scenario,
        source: mode === "injector" ? "injected" : "generated",
        anomaly_count: data.anomaly_count,
        processing_time_ms: data.processing_time_ms,
        alerts_critical: data.alerts.filter((a: Alert) => a.alert_level === "CRITICAL").length,
        alerts_warning: data.alerts.filter((a: Alert) => a.alert_level === "WARNING").length,
        events_processed: data.telemetry_processed,
      };
      setHistory(prev => [entry, ...prev].slice(0, 50));

      if (isStream) setStreamCount(prev => prev + 1);
    } catch (err) {
      setError(`Pipeline error: ${String(err)}`);
    } finally {
      if (!isStream) setLoading(false);
    }
  }, [scenario, mode, injectorJson, threshold]);

  // Streaming toggle
  const toggleStreaming = useCallback(() => {
    if (streaming) {
      if (streamRef.current) clearInterval(streamRef.current);
      streamRef.current = null;
      setStreaming(false);
    } else {
      setStreaming(true);
      setStreamCount(0);
      // Run immediately
      runDetection(true);
      streamRef.current = setInterval(() => runDetection(true), streamInterval);
    }
  }, [streaming, streamInterval, runDetection]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) clearInterval(streamRef.current);
    };
  }, []);

  // Update interval while streaming
  useEffect(() => {
    if (streaming && streamRef.current) {
      clearInterval(streamRef.current);
      streamRef.current = setInterval(() => runDetection(true), streamInterval);
    }
  }, [streamInterval, streaming, runDetection]);

  // Export functions
  const exportJSON = useCallback(() => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `sentinel-detection-${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result]);

  const exportCSV = useCallback(() => {
    if (!result || result.alerts.length === 0) return;
    const headers = ["alert_id", "tourist_id", "anomaly_type", "alert_level", "confidence_score", "lat", "lng", "timestamp", "models_used", "features_extracted"];
    const rows = result.alerts.map(a => [
      a.alert_id, a.tourist_id, a.anomaly_type, a.alert_level,
      a.confidence_score, a.location.lat, a.location.lng,
      a.timestamp, a.models_used.join(";"), a.features_extracted,
    ]);
    const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `sentinel-alerts-${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result]);

  const criticalCount = result?.alerts.filter((a) => a.alert_level === "CRITICAL").length || 0;
  const warningCount = result?.alerts.filter((a) => a.alert_level === "WARNING").length || 0;
  const infoCount = result?.alerts.filter((a) => a.alert_level === "INFO").length || 0;

  return (
    <div className="min-h-screen bg-[#09090b]">
      {/* ===== TOP BAR ===== */}
      <div className="border-b border-white/[0.06] bg-[#09090b]/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 md:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors">
              <ArrowLeft className="w-4 h-4" />
              <span className="text-sm hidden sm:inline">Back</span>
            </Link>
            <div className="w-px h-5 bg-white/[0.08]" />
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-md bg-blue-500/15 border border-blue-500/30 flex items-center justify-center">
                <Activity className="w-3 h-3 text-blue-400" />
              </div>
              <span className="text-sm font-medium text-white">Dashboard</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* Streaming indicator */}
            {streaming && (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-blue-500/10 border border-blue-500/20 animate-pulse">
                <RefreshCw className="w-3 h-3 text-blue-400 animate-spin" />
                <span className="text-[11px] text-blue-400 font-mono">
                  Streaming #{streamCount}
                </span>
              </div>
            )}
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white/[0.03] border border-white/[0.06]">
              <span className={`w-1.5 h-1.5 rounded-full ${
                streaming ? "bg-blue-500 animate-pulse" : result ? "bg-emerald-500" : "bg-zinc-600"
              }`} />
              <span className="text-[11px] text-zinc-400 font-mono">
                {streaming ? "Live" : result ? "Pipeline Active" : "Idle"}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 md:px-6 py-6">
        {/* ===== CONTROLS ROW ===== */}
        <div className="flex flex-wrap items-center gap-2 mb-4">
          {/* Mode tabs */}
          <div className="flex items-center gap-1 p-1 radar-card rounded-lg">
            <button
              onClick={() => setMode("scenario")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                mode === "scenario"
                  ? "bg-blue-500/15 text-blue-400 border border-blue-500/30"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              <Play className="w-3 h-3" />
              Scenario
            </button>
            <button
              onClick={() => { setMode("injector"); setInjectorOpen(true); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                mode === "injector"
                  ? "bg-blue-500/15 text-blue-400 border border-blue-500/30"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              <Syringe className="w-3 h-3" />
              Injector
            </button>
          </div>

          <div className="flex-1" />

          {/* Action buttons */}
          <div className="flex items-center gap-2">
            {/* Settings toggle */}
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all border ${
                showSettings
                  ? "bg-blue-500/15 text-blue-400 border-blue-500/30"
                  : "text-zinc-500 hover:text-zinc-300 border-white/[0.06] hover:border-white/[0.1]"
              }`}
            >
              <Settings2 className="w-3 h-3" />
              <span className="hidden sm:inline">Settings</span>
            </button>

            {/* History toggle */}
            <button
              onClick={() => setShowHistory(!showHistory)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all border ${
                showHistory
                  ? "bg-blue-500/15 text-blue-400 border-blue-500/30"
                  : "text-zinc-500 hover:text-zinc-300 border-white/[0.06] hover:border-white/[0.1]"
              }`}
            >
              <History className="w-3 h-3" />
              <span className="hidden sm:inline">History</span>
              {history.length > 0 && (
                <span className="text-[10px] text-zinc-600 font-mono">{history.length}</span>
              )}
            </button>

            {/* Export dropdown */}
            {result && (
              <div className="flex items-center gap-1">
                <button
                  onClick={exportJSON}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium text-zinc-500 hover:text-zinc-300 border border-white/[0.06] hover:border-white/[0.1] transition-all"
                >
                  <Download className="w-3 h-3" />
                  <span className="hidden sm:inline">JSON</span>
                </button>
                <button
                  onClick={exportCSV}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium text-zinc-500 hover:text-zinc-300 border border-white/[0.06] hover:border-white/[0.1] transition-all"
                >
                  <Download className="w-3 h-3" />
                  <span className="hidden sm:inline">CSV</span>
                </button>
              </div>
            )}
          </div>
        </div>

        {/* ===== SETTINGS PANEL ===== */}
        <AnimatePresence>
          {showSettings && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <div className="radar-card rounded-xl p-4 md:p-6 mb-4">
                <h3 className="text-xs text-zinc-500 uppercase tracking-wider mb-4">Pipeline Configuration</h3>
                <div className="grid md:grid-cols-2 gap-6">
                  {/* Threshold slider */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="text-xs text-zinc-400">Anomaly Sensitivity Threshold</label>
                      <span className="text-xs text-blue-400 font-mono">{threshold.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.05"
                      value={threshold}
                      onChange={(e) => setThreshold(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-white/[0.06] rounded-full appearance-none cursor-pointer accent-blue-500 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-lg [&::-webkit-slider-thumb]:shadow-blue-500/30"
                    />
                    <div className="flex justify-between text-[10px] text-zinc-600 mt-1">
                      <span>More Sensitive</span>
                      <span>Less Sensitive</span>
                    </div>
                  </div>

                  {/* Streaming controls */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="text-xs text-zinc-400">Live Streaming Mode</label>
                      <span className="text-xs text-zinc-600 font-mono">
                        {streaming ? "Active" : "Off"}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={toggleStreaming}
                        className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-medium transition-all ${
                          streaming
                            ? "bg-red-500/15 text-red-400 border border-red-500/30 hover:bg-red-500/25"
                            : "bg-blue-500/15 text-blue-400 border border-blue-500/30 hover:bg-blue-500/25"
                        }`}
                      >
                        {streaming ? (
                          <><Pause className="w-3 h-3" /> Stop</>
                        ) : (
                          <><RefreshCw className="w-3 h-3" /> Start Stream</>
                        )}
                      </button>
                      <div className="flex items-center gap-1">
                        {STREAMING_INTERVALS.map((si) => (
                          <button
                            key={si.value}
                            onClick={() => setStreamInterval(si.value)}
                            className={`px-2 py-1 rounded text-[10px] font-mono transition-all ${
                              streamInterval === si.value
                                ? "bg-blue-500/15 text-blue-400 border border-blue-500/30"
                                : "text-zinc-600 hover:text-zinc-400 border border-white/[0.06]"
                            }`}
                          >
                            {si.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ===== HISTORY PANEL ===== */}
        <AnimatePresence>
          {showHistory && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <div className="radar-card rounded-xl mb-4 overflow-hidden">
                <div className="px-4 md:px-6 py-3 border-b border-white/[0.06] flex items-center justify-between">
                  <h3 className="text-xs text-zinc-500 uppercase tracking-wider">Detection History</h3>
                  {history.length > 0 && (
                    <button
                      onClick={() => setHistory([])}
                      className="text-[10px] text-zinc-600 hover:text-zinc-400 transition-colors"
                    >
                      Clear
                    </button>
                  )}
                </div>
                {history.length === 0 ? (
                  <div className="p-6 text-center">
                    <History className="w-6 h-6 text-zinc-700 mx-auto mb-2" />
                    <p className="text-xs text-zinc-600">No detection runs yet. Run the pipeline to build history.</p>
                  </div>
                ) : (
                  <div className="max-h-48 overflow-y-auto divide-y divide-white/[0.04]">
                    {history.map((entry) => (
                      <div key={entry.id} className="px-4 md:px-6 py-2.5 flex items-center gap-4 text-xs hover:bg-white/[0.02]">
                        <span className="text-zinc-600 font-mono w-16 flex-shrink-0">
                          {entry.timestamp.toLocaleTimeString()}
                        </span>
                        <span className="text-zinc-400 truncate flex-1">{entry.scenario}</span>
                        <span className="text-zinc-600 font-mono">{entry.events_processed} evts</span>
                        <div className="flex items-center gap-2">
                          {entry.alerts_critical > 0 && (
                            <span className="text-red-400 font-mono">{entry.alerts_critical}C</span>
                          )}
                          {entry.alerts_warning > 0 && (
                            <span className="text-amber-400 font-mono">{entry.alerts_warning}W</span>
                          )}
                          {entry.anomaly_count === 0 && (
                            <span className="text-emerald-400 font-mono">Clear</span>
                          )}
                        </div>
                        <span className="text-zinc-600 font-mono">{entry.processing_time_ms}ms</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ===== SCENARIO CONTROLS ===== */}
        {mode === "scenario" && (
          <div className="radar-card rounded-xl p-4 md:p-6 mb-6">
            <div className="flex flex-col md:flex-row md:items-end gap-4">
              <div className="flex-1">
                <label className="text-xs text-zinc-500 uppercase tracking-wider mb-2 block">
                  Telemetry Scenario
                </label>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
                  {SCENARIOS.map((s) => (
                    <button
                      key={s.value}
                      onClick={() => setScenario(s.value)}
                      className={`px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                        scenario === s.value
                          ? "bg-blue-500/15 text-blue-400 border border-blue-500/30"
                          : "bg-white/[0.03] text-zinc-500 border border-white/[0.06] hover:bg-white/[0.06] hover:text-zinc-300"
                      }`}
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>
              <button
                onClick={() => runDetection(false)}
                disabled={loading || streaming}
                className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 border border-blue-500/30 hover:border-blue-500/50 disabled:opacity-50 text-sm font-medium rounded-lg transition-all hover:shadow-lg hover:shadow-blue-500/10 disabled:cursor-not-allowed min-w-[160px]"
              >
                {loading ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Running...</>
                ) : (
                  <><Play className="w-4 h-4" /> Run Detection</>
                )}
              </button>
            </div>
            {!result && !loading && (
              <p className="text-xs text-zinc-600 mt-3">
                Select a scenario and click &quot;Run Detection&quot; to generate telemetry and run the ML pipeline.
              </p>
            )}
          </div>
        )}

        {/* ===== DATA INJECTOR ===== */}
        {mode === "injector" && (
          <div className="radar-card rounded-xl mb-6 overflow-hidden">
            <button
              onClick={() => setInjectorOpen(!injectorOpen)}
              className="w-full px-4 md:px-6 py-4 flex items-center justify-between hover:bg-white/[0.02] transition-colors"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                  <Syringe className="w-4 h-4 text-blue-400" />
                </div>
                <div className="text-left">
                  <h3 className="text-sm font-medium text-white">Data Injector</h3>
                  <p className="text-[11px] text-zinc-500">
                    Inject custom telemetry JSON into the detection pipeline
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-blue-400/60 font-mono hidden sm:inline">
                  max 100 events
                </span>
                <ChevronDown className={`w-4 h-4 text-zinc-600 transition-transform ${injectorOpen ? "rotate-180" : ""}`} />
              </div>
            </button>

            <AnimatePresence>
              {injectorOpen && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <div className="px-4 md:px-6 pb-4 md:pb-6 border-t border-white/[0.06] pt-4">
                    {/* Schema hint */}
                    <div className="flex items-start gap-2 mb-3 p-3 rounded-lg bg-blue-500/5 border border-blue-500/10">
                      <Code2 className="w-3.5 h-3.5 text-blue-400 mt-0.5 flex-shrink-0" />
                      <div className="text-[11px] text-zinc-500 leading-relaxed">
                        <span className="text-blue-400 font-medium">Schema:</span>{" "}
                        Each event needs <code className="text-zinc-300">tourist_id</code>, <code className="text-zinc-300">lat</code>, <code className="text-zinc-300">lng</code>, <code className="text-zinc-300">timestamp</code>.
                        Optional: <code className="text-zinc-300">heart_rate</code>, <code className="text-zinc-300">battery_level</code>, <code className="text-zinc-300">panic_button</code>, <code className="text-zinc-300">network_status</code>, <code className="text-zinc-300">accuracy</code>, <code className="text-zinc-300">altitude</code>.
                      </div>
                    </div>

                    {/* JSON editor */}
                    <textarea
                      value={injectorJson}
                      onChange={(e) => { setInjectorJson(e.target.value); setInjectorError(null); }}
                      spellCheck={false}
                      className="w-full h-64 bg-[#0c0c0e] border border-white/[0.06] rounded-lg p-4 text-xs font-mono text-blue-400/80 leading-relaxed resize-y focus:outline-none focus:border-blue-500/30 focus:ring-1 focus:ring-blue-500/20 placeholder-zinc-700"
                      placeholder="Paste your TelemetryEvent[] JSON here..."
                    />

                    {/* Error */}
                    {injectorError && (
                      <div className="mt-2 flex items-center gap-2 text-xs text-red-400">
                        <AlertTriangle className="w-3 h-3 flex-shrink-0" />
                        {injectorError}
                      </div>
                    )}

                    {/* Actions */}
                    <div className="flex items-center gap-3 mt-3">
                      <button
                        onClick={() => runDetection(false)}
                        disabled={loading || streaming}
                        className="inline-flex items-center gap-2 px-5 py-2.5 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 border border-blue-500/30 hover:border-blue-500/50 disabled:opacity-50 text-sm font-medium rounded-lg transition-all hover:shadow-lg hover:shadow-blue-500/10 disabled:cursor-not-allowed"
                      >
                        {loading ? (
                          <><Loader2 className="w-4 h-4 animate-spin" /> Injecting...</>
                        ) : (
                          <><Syringe className="w-4 h-4" /> Inject &amp; Detect</>
                        )}
                      </button>
                      <button
                        onClick={() => { setInjectorJson(JSON.stringify(generateInjectorTemplate(), null, 2)); setInjectorError(null); }}
                        className="px-4 py-2.5 text-xs text-zinc-500 hover:text-zinc-300 border border-white/[0.06] hover:border-white/[0.1] rounded-lg transition-all"
                      >
                        Reset Template
                      </button>
                      <div className="flex-1" />
                      <span className="text-[10px] text-zinc-600 font-mono hidden md:inline">
                        {(() => { try { return `${JSON.parse(injectorJson).length} events`; } catch { return "invalid"; } })()}
                      </span>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}

        {/* ===== ERROR DISPLAY ===== */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-4 p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center gap-3"
            >
              <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />
              <p className="text-sm text-red-400">{error}</p>
              <button onClick={() => setError(null)} className="ml-auto text-red-400/60 hover:text-red-400">
                <span className="text-xs">Dismiss</span>
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ===== LOADING SKELETON ===== */}
        {loading && !result && (
          <div className="space-y-4 mb-6">
            <div className="radar-card rounded-xl p-4 md:p-6">
              <div className="h-4 w-32 bg-white/[0.05] rounded animate-pulse mb-4" />
              <div className="flex gap-2">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="flex-1 h-8 bg-white/[0.03] rounded-md animate-pulse" style={{ animationDelay: `${i * 100}ms` }} />
                ))}
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="radar-card rounded-xl p-4">
                  <div className="h-3 w-20 bg-white/[0.05] rounded animate-pulse mb-3" />
                  <div className="h-8 w-16 bg-white/[0.05] rounded animate-pulse" />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ===== PIPELINE STAGES ===== */}
        <AnimatePresence>
          {(loading || result) && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="radar-card rounded-xl p-4 md:p-6 mb-6"
            >
              <h3 className="text-xs text-zinc-500 uppercase tracking-wider mb-4">Pipeline Stages</h3>
              <div className="flex flex-wrap items-center gap-2">
                {(result?.pipeline_stages || [
                  { name: "Data Ingestion", status: loading ? "running" : "pending", duration_ms: 0 },
                  { name: "Feature Extraction", status: "pending", duration_ms: 0 },
                  { name: "Ensemble Prediction", status: "pending", duration_ms: 0 },
                  { name: "Alert Classification", status: "pending", duration_ms: 0 },
                  { name: "Alert Dispatch", status: "pending", duration_ms: 0 },
                ] as const).map((stage, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <div
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-md border text-xs font-mono ${
                        stage.status === "completed"
                          ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                          : stage.status === "running"
                          ? "bg-blue-500/10 border-blue-500/20 text-blue-400"
                          : "bg-white/[0.02] border-white/[0.06] text-zinc-600"
                      }`}
                    >
                      {stage.status === "completed" ? (
                        <CheckCircle2 className="w-3 h-3" />
                      ) : stage.status === "running" ? (
                        <Loader2 className="w-3 h-3 animate-spin" />
                      ) : (
                        <Clock className="w-3 h-3" />
                      )}
                      {stage.name}
                      {stage.status === "completed" && (
                        <span className="text-emerald-500/60">{stage.duration_ms}ms</span>
                      )}
                    </div>
                    {i < 4 && <ChevronRight className="w-3 h-3 text-zinc-700" />}
                  </div>
                ))}
              </div>
              {result && (
                <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-zinc-500">
                  <span>Total: <span className="text-white font-mono">{result.processing_time_ms}ms</span></span>
                  <span>Events: <span className="text-white font-mono">{result.telemetry_processed}</span></span>
                  <span>Features: <span className="text-white font-mono">20 x {result.telemetry_processed}</span></span>
                  <span>Threshold: <span className="text-blue-400 font-mono">{result.threshold_used?.toFixed(2) ?? "0.45"}</span></span>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* ===== RESULTS ===== */}
        <AnimatePresence>
          {result && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
              {/* Stats cards */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
                <StatCard label="Total Anomalies" value={result.anomaly_count} icon={Brain} accent="blue" />
                <StatCard label="Critical" value={criticalCount} icon={AlertTriangle} accent="red" />
                <StatCard label="Warning" value={warningCount} icon={Shield} accent="amber" />
                <StatCard label="Info" value={infoCount} icon={Radio} accent="cyan" />
                <StatCard label="Latency" value={`${result.processing_time_ms}ms`} icon={Zap} accent="emerald" />
              </div>

              {/* ===== PERFORMANCE METRICS ===== */}
              <div className="radar-card rounded-xl p-4 md:p-6 mb-6">
                <h3 className="text-xs text-zinc-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                  <Gauge className="w-3 h-3" /> Performance Metrics
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {result.pipeline_stages.map((stage, i) => (
                    <div key={i} className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-zinc-500 truncate">{stage.name}</span>
                        <span className="text-[10px] text-zinc-400 font-mono">{stage.duration_ms}ms</span>
                      </div>
                      <div className="h-1.5 bg-white/[0.05] rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.min((stage.duration_ms / result.processing_time_ms) * 100, 100)}%` }}
                          transition={{ duration: 0.6, delay: i * 0.1 }}
                          className="h-full bg-blue-500/50 rounded-full"
                        />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-4 pt-3 border-t border-white/[0.04] grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                  <div>
                    <span className="text-zinc-600">Throughput</span>
                    <p className="text-white font-mono">
                      {(result.telemetry_processed / (result.processing_time_ms / 1000)).toFixed(0)} events/s
                    </p>
                  </div>
                  <div>
                    <span className="text-zinc-600">Avg per Event</span>
                    <p className="text-white font-mono">
                      {(result.processing_time_ms / result.telemetry_processed).toFixed(1)}ms
                    </p>
                  </div>
                  <div>
                    <span className="text-zinc-600">Detection Rate</span>
                    <p className="text-white font-mono">
                      {((result.anomaly_count / result.telemetry_processed) * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <span className="text-zinc-600">Models</span>
                    <p className="text-white font-mono">4 / 4 active</p>
                  </div>
                </div>
              </div>

              {/* ===== ALERT FEED + DETAIL ===== */}
              <div className="grid lg:grid-cols-3 gap-6">
                {/* Alert list */}
                <div className="lg:col-span-2 radar-card rounded-xl overflow-hidden">
                  <div className="px-4 md:px-6 py-4 border-b border-white/[0.06] flex items-center justify-between">
                    <h3 className="text-sm font-medium text-white">Alert Feed</h3>
                    <span className="text-xs text-zinc-500 font-mono">{result.alerts.length} alerts</span>
                  </div>
                  <div className="divide-y divide-white/[0.04] max-h-[500px] overflow-y-auto">
                    {result.alerts.length === 0 ? (
                      <div className="p-8 text-center">
                        <CheckCircle2 className="w-8 h-8 text-emerald-500/40 mx-auto mb-3" />
                        <p className="text-sm text-zinc-400">All clear. No anomalies detected.</p>
                        <p className="text-xs text-zinc-600 mt-1">Try a different scenario or lower the threshold.</p>
                      </div>
                    ) : (
                      result.alerts.map((alert, i) => {
                        const config = SEVERITY_CONFIG[alert.alert_level];
                        const Icon = config.icon;
                        return (
                          <motion.button
                            key={alert.alert_id}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.02 }}
                            onClick={() => setSelectedAlert(alert)}
                            className={`w-full px-4 md:px-6 py-3 flex items-center gap-4 hover:bg-white/[0.03] transition-colors text-left ${
                              selectedAlert?.alert_id === alert.alert_id ? "bg-white/[0.04]" : ""
                            }`}
                          >
                            <div className={`w-8 h-8 rounded-lg ${config.bg} border ${config.border} flex items-center justify-center flex-shrink-0`}>
                              <Icon className={`w-4 h-4 ${config.color}`} />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                <span className="text-sm text-white font-medium truncate">
                                  {ANOMALY_LABELS[alert.anomaly_type]}
                                </span>
                                <span className={`text-[10px] px-1.5 py-0.5 rounded ${config.bg} ${config.color} border ${config.border} font-mono`}>
                                  {alert.alert_level}
                                </span>
                              </div>
                              <div className="flex items-center gap-3 mt-0.5">
                                <span className="text-[11px] text-zinc-600 font-mono">{alert.tourist_id}</span>
                                <span className="text-[11px] text-zinc-600">
                                  {Math.round(alert.confidence_score * 100)}% confidence
                                </span>
                              </div>
                            </div>
                            <ChevronRight className="w-4 h-4 text-zinc-700 flex-shrink-0" />
                          </motion.button>
                        );
                      })
                    )}
                  </div>
                </div>

                {/* Alert detail */}
                <div className="radar-card rounded-xl overflow-hidden">
                  <div className="px-4 md:px-6 py-4 border-b border-white/[0.06]">
                    <h3 className="text-sm font-medium text-white">Alert Detail</h3>
                  </div>
                  {selectedAlert ? (
                    <div className="p-4 md:p-6 space-y-4">
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <span className={`text-[10px] px-1.5 py-0.5 rounded ${SEVERITY_CONFIG[selectedAlert.alert_level].bg} ${SEVERITY_CONFIG[selectedAlert.alert_level].color} border ${SEVERITY_CONFIG[selectedAlert.alert_level].border} font-mono`}>
                            {selectedAlert.alert_level}
                          </span>
                        </div>
                        <h4 className="text-lg font-semibold text-white">
                          {ANOMALY_LABELS[selectedAlert.anomaly_type]}
                        </h4>
                        <p className="text-xs text-zinc-500 font-mono mt-1">{selectedAlert.alert_id}</p>
                      </div>

                      <div className="space-y-3">
                        <DetailRow icon={MapPin} label="Tourist" value={selectedAlert.tourist_id} />
                        <DetailRow
                          icon={MapPin}
                          label="Location"
                          value={`${selectedAlert.location.lat.toFixed(4)}, ${selectedAlert.location.lng.toFixed(4)}`}
                        />
                        <DetailRow icon={Brain} label="Confidence" value={`${Math.round(selectedAlert.confidence_score * 100)}%`} />
                        <DetailRow icon={Clock} label="Timestamp" value={new Date(selectedAlert.timestamp).toLocaleTimeString()} />
                        <DetailRow icon={Zap} label="Models Used" value={selectedAlert.models_used.length.toString()} />
                        <DetailRow icon={Activity} label="Features" value={`${selectedAlert.features_extracted}`} />
                      </div>

                      {/* Model scores */}
                      {selectedAlert.raw_evidence.model_scores != null && (
                        <div>
                          <p className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Model Scores</p>
                          <div className="space-y-2">
                            {Object.entries(selectedAlert.raw_evidence.model_scores as Record<string, number>).map(([model, score]) => (
                              <div key={model} className="flex items-center gap-2">
                                <span className="text-[11px] text-zinc-500 w-28 truncate font-mono">{model}</span>
                                <div className="flex-1 h-1.5 bg-white/[0.05] rounded-full overflow-hidden">
                                  <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${Math.min(score * 100, 100)}%` }}
                                    transition={{ duration: 0.5 }}
                                    className={`h-full rounded-full ${
                                      score > 0.7 ? "bg-red-500" : score > 0.4 ? "bg-amber-500" : "bg-emerald-500"
                                    }`}
                                  />
                                </div>
                                <span className="text-[11px] text-zinc-400 font-mono w-10 text-right">
                                  {(score as number).toFixed(2)}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Evidence */}
                      <div>
                        <p className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Raw Evidence</p>
                        <div className="radar-card rounded-lg p-3">
                          <pre className="text-[10px] text-zinc-400 font-mono whitespace-pre-wrap overflow-x-auto">
                            {JSON.stringify(
                              {
                                velocity_kmh: selectedAlert.raw_evidence.velocity_kmh,
                                heart_rate: selectedAlert.raw_evidence.heart_rate,
                                battery_level: selectedAlert.raw_evidence.battery_level,
                                bearing_change: selectedAlert.raw_evidence.bearing_change,
                                movement_efficiency: selectedAlert.raw_evidence.movement_efficiency,
                              },
                              null,
                              2
                            )}
                          </pre>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="p-8 text-center">
                      <Radio className="w-8 h-8 text-zinc-700 mx-auto mb-3" />
                      <p className="text-sm text-zinc-500">Select an alert to view details</p>
                    </div>
                  )}
                </div>
              </div>

              {/* ===== ANOMALY DISTRIBUTION ===== */}
              {result.alerts.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="radar-card rounded-xl p-4 md:p-6 mt-6"
                >
                  <h3 className="text-sm font-medium text-white mb-4">Anomaly Distribution</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3">
                    {Object.entries(
                      result.alerts.reduce<Record<string, number>>((acc, alert) => {
                        acc[alert.anomaly_type] = (acc[alert.anomaly_type] || 0) + 1;
                        return acc;
                      }, {})
                    )
                      .sort(([, a], [, b]) => b - a)
                      .map(([type, count]) => {
                        const total = result.alerts.length;
                        const pct = (count / total) * 100;
                        return (
                          <div key={type} className="radar-card rounded-lg p-3">
                            <div className="text-xl font-bold text-white font-mono">{count}</div>
                            <div className="text-[10px] text-zinc-500 mt-1 uppercase tracking-wider">
                              {ANOMALY_LABELS[type as AnomalyType] || type}
                            </div>
                            <div className="mt-2 h-1 bg-white/[0.05] rounded-full overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${pct}%` }}
                                transition={{ duration: 0.5 }}
                                className="h-full bg-blue-500/50 rounded-full"
                              />
                            </div>
                            <div className="text-[9px] text-zinc-600 font-mono mt-1">{pct.toFixed(0)}%</div>
                          </div>
                        );
                      })}
                  </div>
                </motion.div>
              )}

              {/* ===== SESSION SUMMARY ===== */}
              {history.length > 1 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="radar-card rounded-xl p-4 md:p-6 mt-6"
                >
                  <h3 className="text-sm font-medium text-white mb-4 flex items-center gap-2">
                    <Timer className="w-4 h-4 text-blue-400" /> Session Summary
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                    <div>
                      <span className="text-zinc-600">Total Runs</span>
                      <p className="text-white font-mono text-lg">{history.length}</p>
                    </div>
                    <div>
                      <span className="text-zinc-600">Total Events</span>
                      <p className="text-white font-mono text-lg">
                        {history.reduce((sum, h) => sum + h.events_processed, 0)}
                      </p>
                    </div>
                    <div>
                      <span className="text-zinc-600">Total Anomalies</span>
                      <p className="text-white font-mono text-lg">
                        {history.reduce((sum, h) => sum + h.anomaly_count, 0)}
                      </p>
                    </div>
                    <div>
                      <span className="text-zinc-600">Avg Latency</span>
                      <p className="text-white font-mono text-lg">
                        {Math.round(history.reduce((sum, h) => sum + h.processing_time_ms, 0) / history.length)}ms
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

// ============================================================
// SUB-COMPONENTS
// ============================================================

function StatCard({
  label,
  value,
  icon: Icon,
  accent,
}: {
  label: string;
  value: number | string;
  icon: typeof Activity;
  accent: string;
}) {
  const colors: Record<string, string> = {
    blue: "text-blue-400 bg-blue-500/10 border-blue-500/20",
    red: "text-red-400 bg-red-500/10 border-red-500/20",
    amber: "text-amber-400 bg-amber-500/10 border-amber-500/20",
    emerald: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
    cyan: "text-sky-400 bg-sky-500/10 border-sky-500/20",
  };
  const c = colors[accent] || colors.blue;
  const textColor = c.split(" ")[0];

  return (
    <div className="radar-card rounded-xl p-4">
      <div className="flex items-center gap-2 mb-2">
        <div className={`w-6 h-6 rounded-md ${c.split(" ").slice(1).join(" ")} flex items-center justify-center border`}>
          <Icon className={`w-3 h-3 ${textColor}`} />
        </div>
        <span className="text-[11px] text-zinc-500 uppercase tracking-wider">{label}</span>
      </div>
      <div className="text-2xl font-bold text-white font-mono">{value}</div>
    </div>
  );
}

function DetailRow({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Activity;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-3">
      <Icon className="w-3.5 h-3.5 text-zinc-600 flex-shrink-0" />
      <span className="text-xs text-zinc-500 w-20">{label}</span>
      <span className="text-xs text-white font-mono">{value}</span>
    </div>
  );
}
