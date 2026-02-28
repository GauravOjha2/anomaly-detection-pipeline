"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Brain,
  CheckCircle2,
  ChevronRight,
  Clock,
  Loader2,
  MapPin,
  Play,
  Radio,
  Shield,
  Zap,
} from "lucide-react";
import { DetectionResult, Alert, AnomalyType, AlertLevel } from "@/lib/types";
import { Scenario } from "@/lib/mock-data";

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

export default function DashboardPage() {
  const [scenario, setScenario] = useState<Scenario>("mixed");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [_runCount, setRunCount] = useState(0);

  const runDetection = useCallback(async () => {
    setLoading(true);
    setSelectedAlert(null);
    try {
      const res = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario, count: 30, tourist_count: 5 }),
      });
      const data = await res.json();
      setResult(data);
      setRunCount((c) => c + 1);
    } catch (err) {
      console.error("Detection failed:", err);
    } finally {
      setLoading(false);
    }
  }, [scenario]);

  const criticalCount = result?.alerts.filter((a) => a.alert_level === "CRITICAL").length || 0;
  const warningCount = result?.alerts.filter((a) => a.alert_level === "WARNING").length || 0;
  const _infoCount = result?.alerts.filter((a) => a.alert_level === "INFO").length || 0;

  return (
    <div className="min-h-screen bg-[#09090b]">
      {/* Top bar */}
      <div className="border-b border-white/[0.06] bg-[#09090b]/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 md:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors">
              <ArrowLeft className="w-4 h-4" />
              <span className="text-sm hidden sm:inline">Back</span>
            </Link>
            <div className="w-px h-5 bg-white/[0.08]" />
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-md bg-gradient-to-br from-indigo-500 to-indigo-700 flex items-center justify-center">
                <Activity className="w-3 h-3 text-white" />
              </div>
              <span className="text-sm font-medium text-white">Dashboard</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white/[0.03] border border-white/[0.06]">
              <span className={`w-1.5 h-1.5 rounded-full ${result ? "bg-emerald-500" : "bg-zinc-600"} ${result ? "animate-pulse" : ""}`} />
              <span className="text-[11px] text-zinc-400 font-mono">
                {result ? "Pipeline Active" : "Idle"}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 md:px-6 py-6">
        {/* Controls */}
        <div className="glass rounded-xl p-4 md:p-6 mb-6">
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
                        ? "bg-indigo-500/15 text-indigo-400 border border-indigo-500/30"
                        : "bg-white/[0.03] text-zinc-500 border border-white/[0.06] hover:bg-white/[0.06] hover:text-zinc-300"
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
            <button
              onClick={runDetection}
              disabled={loading}
              className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-indigo-500 hover:bg-indigo-400 disabled:bg-indigo-500/50 text-white text-sm font-medium rounded-lg transition-all hover:shadow-lg hover:shadow-indigo-500/25 disabled:cursor-not-allowed min-w-[160px]"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run Detection
                </>
              )}
            </button>
          </div>
          {!result && !loading && (
            <p className="text-xs text-zinc-600 mt-3">
              Select a scenario and click &quot;Run Detection&quot; to generate telemetry and run the ML pipeline.
            </p>
          )}
        </div>

        {/* Pipeline stages */}
        <AnimatePresence>
          {(loading || result) && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass rounded-xl p-4 md:p-6 mb-6"
            >
              <h3 className="text-xs text-zinc-500 uppercase tracking-wider mb-4">Pipeline Stages</h3>
              <div className="flex flex-wrap items-center gap-2">
                {(result?.pipeline_stages || [
                  { name: "Data Ingestion", status: loading ? "running" : "pending", duration_ms: 0 },
                  { name: "Feature Extraction", status: "pending", duration_ms: 0 },
                  { name: "Ensemble Prediction", status: "pending", duration_ms: 0 },
                  { name: "Alert Classification", status: "pending", duration_ms: 0 },
                  { name: "Alert Dispatch", status: "pending", duration_ms: 0 },
                ]).map((stage, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <div
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-md border text-xs font-mono ${
                        stage.status === "completed"
                          ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                          : stage.status === "running"
                          ? "bg-indigo-500/10 border-indigo-500/20 text-indigo-400"
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
                <div className="mt-3 flex items-center gap-4 text-xs text-zinc-500">
                  <span>Total: <span className="text-white font-mono">{result.processing_time_ms}ms</span></span>
                  <span>Events: <span className="text-white font-mono">{result.telemetry_processed}</span></span>
                  <span>Features: <span className="text-white font-mono">20 x {result.telemetry_processed}</span></span>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results */}
        <AnimatePresence>
          {result && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
              {/* Stats cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                <StatCard label="Total Anomalies" value={result.anomaly_count} icon={Brain} accent="indigo" />
                <StatCard label="Critical" value={criticalCount} icon={AlertTriangle} accent="red" />
                <StatCard label="Warning" value={warningCount} icon={Shield} accent="amber" />
                <StatCard label="Processing" value={`${result.processing_time_ms}ms`} icon={Zap} accent="emerald" />
              </div>

              {/* Alert feed + detail */}
              <div className="grid lg:grid-cols-3 gap-6">
                {/* Alert list */}
                <div className="lg:col-span-2 glass rounded-xl overflow-hidden">
                  <div className="px-4 md:px-6 py-4 border-b border-white/[0.06] flex items-center justify-between">
                    <h3 className="text-sm font-medium text-white">Alert Feed</h3>
                    <span className="text-xs text-zinc-500 font-mono">{result.alerts.length} alerts</span>
                  </div>
                  <div className="divide-y divide-white/[0.04] max-h-[500px] overflow-y-auto">
                    {result.alerts.length === 0 ? (
                      <div className="p-8 text-center">
                        <CheckCircle2 className="w-8 h-8 text-emerald-500/40 mx-auto mb-3" />
                        <p className="text-sm text-zinc-400">All clear. No anomalies detected.</p>
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
                            transition={{ delay: i * 0.03 }}
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
                <div className="glass rounded-xl overflow-hidden">
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
                                  <div
                                    className={`h-full rounded-full transition-all duration-500 ${
                                      score > 0.7 ? "bg-red-500" : score > 0.4 ? "bg-amber-500" : "bg-emerald-500"
                                    }`}
                                    style={{ width: `${Math.min(score * 100, 100)}%` }}
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
                        <div className="glass rounded-lg p-3">
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

              {/* Anomaly distribution chart */}
              {result.alerts.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="glass rounded-xl p-4 md:p-6 mt-6"
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
                      .map(([type, count]) => (
                        <div
                          key={type}
                          className="glass rounded-lg p-3 text-center"
                        >
                          <div className="text-xl font-bold text-white font-mono">{count}</div>
                          <div className="text-[10px] text-zinc-500 mt-1 uppercase tracking-wider">
                            {ANOMALY_LABELS[type as AnomalyType]?.replace(/ /g, "\n") || type}
                          </div>
                        </div>
                      ))}
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
    indigo: "text-indigo-400 bg-indigo-500/10 border-indigo-500/20",
    red: "text-red-400 bg-red-500/10 border-red-500/20",
    amber: "text-amber-400 bg-amber-500/10 border-amber-500/20",
    emerald: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
  };
  const c = colors[accent] || colors.indigo;
  const textColor = c.split(" ")[0];

  return (
    <div className="glass rounded-xl p-4">
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
