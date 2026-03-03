"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  Activity,
  ArrowRight,
  Brain,
  ChevronDown,
  Database,
  Gauge,
  GitBranch,
  Layers,
  Radio,
  Server,
  Shield,
  Crosshair,
  Terminal,
  Zap,
} from "lucide-react";
import Navbar from "@/components/Navbar";

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.6, ease: "easeOut" as const },
  }),
};

const stats = [
  { value: "4", label: "ML Models", suffix: "" },
  { value: "20", label: "Features Extracted", suffix: "" },
  { value: "30+", label: "Events / Batch", suffix: "" },
  { value: "<1s", label: "Inference Latency", suffix: "" },
];

const pipelineSteps = [
  {
    num: "01",
    tag: "input",
    title: "Telemetry Ingestion",
    description:
      "IoT sensor data streams in from GPS devices carried by tourists. Each event contains coordinates, heart rate, battery level, network status, and panic button state. Data is validated and normalized for the pipeline.",
    checks: [
      "GPS coordinates + altitude",
      "Heart rate + battery monitoring",
      "Panic button + network status",
    ],
    terminal: `> ingest_telemetry(batch_size=30)
> validating schema... OK
> 30 events received
> normalizing coordinates...
> ready for feature extraction_`,
  },
  {
    num: "02",
    tag: "processing",
    title: "Feature Engineering",
    description:
      "Raw telemetry is transformed into a 20-dimensional feature vector per data point. Features include haversine distances, instantaneous velocity, acceleration, bearing changes, rolling statistics, z-scores, temporal patterns, and movement efficiency metrics.",
    checks: [
      "Haversine distance + velocity",
      "Rolling mean/std (window=5)",
      "Z-score outlier detection",
    ],
    terminal: `> extract_features(n=20)
> computing haversine distances...
> velocity: 4.2 km/h (normal)
> bearing_change: 12.3 deg
> z_score: 0.3 (within bounds)
> 20-dim vector ready_`,
  },
  {
    num: "03",
    tag: "inference",
    title: "Ensemble ML Detection",
    description:
      "Feature vectors are passed through a 4-model ensemble: Isolation Forest for tree-based outlier isolation, Elliptic Envelope for Mahalanobis distance, One-Class SVM with RBF kernel for boundary detection, and an Autoencoder for reconstruction-error-based anomaly scoring.",
    checks: [
      "Weighted ensemble (IF + EE + SVM + AE)",
      "Confidence scoring per model",
      "Rule-based classification overlay",
    ],
    terminal: `> ensemble_predict(models=4)
> isolation_forest:  0.23
> elliptic_envelope: 0.18
> one_class_svm:     0.31
> autoencoder:       0.27
> ensemble_score:    0.245
> classification: NORMAL_`,
  },
];

const features = [
  {
    icon: Brain,
    title: "Ensemble Architecture",
    description:
      "Four complementary models vote on each data point. No single model dominates \u2014 the ensemble captures different anomaly geometries that individual models miss.",
  },
  {
    icon: Layers,
    title: "20-Feature Pipeline",
    description:
      "Every telemetry event produces a rich feature vector: spatial, temporal, kinematic, and physiological dimensions. Rolling statistics and z-scores add context.",
  },
  {
    icon: Zap,
    title: "Real-Time Processing",
    description:
      "Sub-second inference on batches of 30+ events. The pipeline runs entirely client-side with no external API calls \u2014 zero cold starts, instant results.",
  },
  {
    icon: Shield,
    title: "Rule-Based Failsafe",
    description:
      "Critical conditions (panic button, extreme heart rate, velocity) are caught by deterministic rules regardless of ML scores. Safety-first classification.",
  },
  {
    icon: Radio,
    title: "Live Alert Dispatch",
    description:
      "Anomalies are classified by type and severity, then dispatched to the dashboard in real-time. Each alert includes model scores, evidence, and confidence.",
  },
  {
    icon: Terminal,
    title: "Pipeline Transparency",
    description:
      "Every stage is timed and logged. See exactly how long feature extraction, model inference, and classification took for each batch.",
  },
];

const techStack = [
  {
    icon: Brain,
    category: "ML Engine",
    name: "Ensemble Models",
    description:
      "Isolation Forest, Elliptic Envelope, One-Class SVM, and Autoencoder running as a weighted ensemble.",
  },
  {
    icon: Server,
    category: "Frontend + API",
    name: "Next.js 14",
    description:
      "App Router with TypeScript, API routes for detection pipeline, server-side rendering.",
  },
  {
    icon: Database,
    category: "Data Pipeline",
    name: "TypeScript ML",
    description:
      "Full ML pipeline in TypeScript. 20-feature extraction, haversine distances, z-scores, rolling stats.",
  },
  {
    icon: Gauge,
    category: "Infrastructure",
    name: "Vercel",
    description:
      "Edge deployment with automatic scaling, zero config, and instant global distribution.",
  },
];

function RadarBackground() {
  return (
    <div className="fixed inset-0 flex items-center justify-center pointer-events-none -z-10 overflow-hidden">
      <div className="relative w-[600px] h-[600px] opacity-30">
        {/* Concentric rings */}
        <div className="radar-ring radar-ring-4" />
        <div className="radar-ring radar-ring-3" />
        <div className="radar-ring radar-ring-2" />
        <div className="radar-ring radar-ring-1" />
        
        {/* Cross lines */}
        <div className="absolute top-1/2 left-0 right-0 h-px bg-radar-ring" />
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-radar-ring" />
        
        {/* Sweep */}
        <div className="radar-sweep" />
        
        {/* Center dot */}
        <div className="radar-center" />
        
        {/* Random blips */}
        <div className="radar-blip" style={{ top: '25%', left: '60%', animationDelay: '0s' }} />
        <div className="radar-blip" style={{ top: '70%', left: '35%', animationDelay: '0.5s' }} />
        <div className="radar-blip warning" style={{ top: '45%', left: '75%', animationDelay: '1s' }} />
        <div className="radar-blip critical" style={{ top: '80%', left: '70%', animationDelay: '1.5s' }} />
      </div>
    </div>
  );
}

export default function HomePage() {
  return (
    <main className="relative min-h-screen">
      <Navbar />

      {/* ===== HERO ===== */}
      <section className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-20 dot-grid">
        <RadarBackground />
        
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 px-3 py-1.5 mb-8 rounded-full border border-radar-green/20 bg-radar-greenDim text-xs text-radar-green"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-radar-green animate-pulse" />
            Real-Time ML Pipeline &middot; v2.0
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1 }}
            className="text-5xl md:text-7xl font-bold tracking-tight leading-[1.1] mb-6"
          >
            <span className="radar-gradient-text">Sentinel</span>
          </motion.h1>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-lg md:text-xl text-zinc-400 max-w-2xl mx-auto mb-10 leading-relaxed"
          >
            Real-time tourist safety monitoring through GPS trajectory analysis
            and IoT sensor telemetry. Ensemble ML models detect anomalies and
            dispatch intelligent alerts.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex items-center justify-center gap-4"
          >
            <Link
              href="/dashboard"
              className="group inline-flex items-center gap-2 px-6 py-3 bg-radar-green/20 hover:bg-radar-green/30 text-radar-green border border-radar-green/30 hover:border-radar-green/50 text-sm font-medium rounded-lg transition-all"
            >
              Live Demo
              <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </Link>
            <Link
              href="#how-it-works"
              className="inline-flex items-center gap-2 px-6 py-3 text-zinc-400 hover:text-white text-sm font-medium rounded-lg border border-white/[0.08] hover:border-white/[0.15] hover:bg-white/[0.03] transition-all"
            >
              How It Works
            </Link>
          </motion.div>
        </div>

        {/* Stats bar */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.5 }}
          className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-16"
        >
          {stats.map((stat, i) => (
            <div key={i} className="text-center">
              <div className="text-2xl md:text-3xl font-bold text-radar-green font-mono radar-glow">
                {stat.value}
              </div>
              <div className="text-xs text-zinc-500 mt-1 uppercase tracking-wider">
                {stat.label}
              </div>
            </div>
          ))}
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="absolute bottom-8 flex flex-col items-center gap-2 text-zinc-600"
        >
          <span className="text-xs">Scroll to explore</span>
          <ChevronDown className="w-4 h-4 animate-bounce" />
        </motion.div>
      </section>

      {/* ===== HOW IT WORKS ===== */}
      <section id="how-it-works" className="py-32 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            className="mb-20"
          >
            <motion.p
              variants={fadeUp}
              custom={0}
              className="text-xs text-radar-green uppercase tracking-widest mb-3"
            >
              Architecture
            </motion.p>
            <motion.h2
              variants={fadeUp}
              custom={1}
              className="text-3xl md:text-4xl font-bold radar-gradient-text mb-4"
            >
              How It Works
            </motion.h2>
            <motion.p
              variants={fadeUp}
              custom={2}
              className="text-zinc-400 max-w-xl"
            >
              A three-stage processing pipeline that transforms raw IoT
              telemetry into real-time anomaly alerts.
            </motion.p>
          </motion.div>

          <div className="space-y-16">
            {pipelineSteps.map((step, i) => (
              <motion.div
                key={i}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-50px" }}
                className="grid md:grid-cols-2 gap-8 items-start"
              >
                {/* Left: description */}
                <div>
                  <motion.div
                    variants={fadeUp}
                    custom={0}
                    className="flex items-center gap-3 mb-4"
                  >
                    <span className="text-3xl font-mono font-bold text-radar-green/10">
                      {step.num}
                    </span>
                    <span className="px-2 py-0.5 text-[10px] uppercase tracking-wider text-radar-green bg-radar-greenDim rounded border border-radar-green/20 font-mono">
                      {step.tag}
                    </span>
                  </motion.div>
                  <motion.h3
                    variants={fadeUp}
                    custom={1}
                    className="text-xl font-semibold text-white mb-3"
                  >
                    {step.title}
                  </motion.h3>
                  <motion.p
                    variants={fadeUp}
                    custom={2}
                    className="text-sm text-zinc-400 leading-relaxed mb-5"
                  >
                    {step.description}
                  </motion.p>
                  <motion.ul variants={fadeUp} custom={3} className="space-y-2">
                    {step.checks.map((check, j) => (
                      <li
                        key={j}
                        className="flex items-center gap-2 text-sm text-zinc-500"
                      >
                        <Crosshair className="w-3 h-3 text-radar-green flex-shrink-0" />
                        {check}
                      </li>
                    ))}
                  </motion.ul>
                </div>

                {/* Right: terminal */}
                <motion.div
                  variants={fadeUp}
                  custom={2}
                  className="radar-card rounded-xl overflow-hidden radar-border-glow"
                >
                  <div className="flex items-center gap-1.5 px-4 py-3 border-b border-radar-green/10">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/60" />
                    <div className="w-2.5 h-2.5 rounded-full bg-radar-green/60" />
                    <span className="ml-2 text-[10px] text-zinc-600 font-mono">
                      pipeline_output
                    </span>
                  </div>
                  <pre className="p-4 text-xs text-radar-green/80 font-mono leading-relaxed whitespace-pre-wrap">
                    {step.terminal}
                  </pre>
                </motion.div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== FEATURES ===== */}
      <section id="features" className="py-32 px-6 border-t border-radar-green/10">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            className="mb-16"
          >
            <motion.p
              variants={fadeUp}
              custom={0}
              className="text-xs text-radar-green uppercase tracking-widest mb-3"
            >
              Features
            </motion.p>
            <motion.h2
              variants={fadeUp}
              custom={1}
              className="text-3xl md:text-4xl font-bold radar-gradient-text mb-4"
            >
              Engineered for Depth
            </motion.h2>
            <motion.p
              variants={fadeUp}
              custom={2}
              className="text-zinc-400 max-w-xl"
            >
              Every design decision serves a purpose &mdash; from how we extract
              features to how we ensemble models.
            </motion.p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map((feature, i) => (
              <motion.div
                key={i}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-50px" }}
                variants={fadeUp}
                custom={i}
                className="radar-card glass-hover rounded-xl p-6 transition-all duration-300"
              >
                <feature.icon className="w-5 h-5 text-radar-green mb-4" />
                <h3 className="text-sm font-semibold text-white mb-2">
                  {feature.title}
                </h3>
                <p className="text-xs text-zinc-500 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== TECH STACK ===== */}
      <section id="stack" className="py-32 px-6 border-t border-radar-green/10">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            className="mb-16"
          >
            <motion.p
              variants={fadeUp}
              custom={0}
              className="text-xs text-radar-green uppercase tracking-widest mb-3"
            >
              Stack
            </motion.p>
            <motion.h2
              variants={fadeUp}
              custom={1}
              className="text-3xl md:text-4xl font-bold radar-gradient-text mb-4"
            >
              Built With Modern Tools
            </motion.h2>
            <motion.p
              variants={fadeUp}
              custom={2}
              className="text-zinc-400 max-w-xl"
            >
              Production-grade architecture, deployed on Vercel as a single
              application.
            </motion.p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-4">
            {techStack.map((tech, i) => (
              <motion.div
                key={i}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-50px" }}
                variants={fadeUp}
                custom={i}
                className="radar-card glass-hover rounded-xl p-6 transition-all duration-300"
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-10 h-10 rounded-lg bg-radar-greenDim flex items-center justify-center border border-radar-green/20">
                    <tech.icon className="w-5 h-5 text-radar-green" />
                  </div>
                  <div>
                    <p className="text-[10px] text-zinc-600 uppercase tracking-wider">
                      {tech.category}
                    </p>
                    <p className="text-sm font-semibold text-white">
                      {tech.name}
                    </p>
                  </div>
                </div>
                <p className="text-xs text-zinc-500 leading-relaxed">
                  {tech.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== CTA ===== */}
      <section className="py-32 px-6 border-t border-radar-green/10">
        <div className="max-w-3xl mx-auto text-center">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-3xl md:text-4xl font-bold radar-gradient-text mb-6"
          >
            Ready to see anomaly
            <br />
            detection in action?
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-zinc-400 mb-8 max-w-lg mx-auto"
          >
            Generate dummy telemetry data, run it through the 4-model ensemble
            pipeline, and watch anomalies appear on the live dashboard in
            real-time.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="flex items-center justify-center gap-4"
          >
            <Link
              href="/dashboard"
              className="group inline-flex items-center gap-2 px-6 py-3 bg-radar-green/20 hover:bg-radar-green/30 text-radar-green border border-radar-green/30 hover:border-radar-green/50 text-sm font-medium rounded-lg transition-all"
            >
              Try It Now
              <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </Link>
            <a
              href="https://github.com/GauravOjha2/anomaly-detection-pipeline"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 text-zinc-400 hover:text-white text-sm font-medium rounded-lg border border-white/[0.08] hover:border-white/[0.15] hover:bg-white/[0.03] transition-all"
            >
              View on GitHub
              <GitBranch className="w-4 h-4" />
            </a>
          </motion.div>
        </div>
      </section>

      {/* ===== FOOTER ===== */}
      <footer className="border-t border-radar-green/10 py-12 px-6">
        <div className="max-w-5xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md bg-radar-greenDim flex items-center justify-center border border-radar-green/30">
              <Activity className="w-3 h-3 text-radar-green" />
            </div>
            <span className="text-sm font-medium text-zinc-400">sentinel</span>
          </div>
          <p className="text-xs text-zinc-600">
            Anomaly Detection Pipeline &middot; Ensemble ML &middot; Next.js 14
            &middot; Vercel
          </p>
        </div>
      </footer>
    </main>
  );
}
