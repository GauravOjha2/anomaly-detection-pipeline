"use client";

import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import {
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
  Clock,
  Cpu,
  Globe,
  TrendingUp,
  BarChart3,
  BookOpen,
  Copy,
  Check,
} from "lucide-react";
import { useState } from "react";
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
  { value: "4", label: "ML Models", suffix: "", icon: Brain },
  { value: "16", label: "Features Extracted", suffix: "", icon: Layers },
  { value: "50+", label: "Species / Batch", suffix: "", icon: Database },
  { value: "<1s", label: "Inference Latency", suffix: "", icon: Clock },
  { value: "8", label: "Anomaly Categories", suffix: "", icon: TrendingUp },
  { value: "Live", label: "iNaturalist Data", suffix: "", icon: Globe },
];

const pipelineSteps = [
  {
    num: "01",
    tag: "input",
    title: "Data Ingestion",
    description:
      "Real-time observations stream in from the iNaturalist API — the world's largest citizen science biodiversity platform. Each sighting includes species identification, GPS coordinates, timestamps, conservation status, and observation photos. Data is validated and normalized for the pipeline.",
    checks: [
      "Species ID + IUCN conservation status",
      "GPS coordinates + positional accuracy",
      "Observation photos + quality grade",
    ],
    terminal: `> fetch_sightings(source="iNaturalist")
> filtering threatened species... OK
> 30 observations received
> mapping conservation status...
> ready for feature extraction_`,
  },
  {
    num: "02",
    tag: "processing",
    title: "Feature Engineering",
    description:
      "Raw sightings are transformed into a 16-dimensional feature vector per observation. Features include spatial isolation metrics, species range deviation scores, temporal patterns, IUCN rarity indices, geographic clustering density, and data quality signals — all computed relative to the batch distribution.",
    checks: [
      "Haversine distance + spatial density",
      "Range deviation + peer isolation",
      "IUCN rarity + quality scoring",
    ],
    terminal: `> extract_features(n=16)
> computing spatial isolation...
> range_deviation: 0.12 (in range)
> iucn_rarity: 0.80 (endangered)
> species_frequency: 0.33
> 16-dim vector ready_`,
  },
  {
    num: "03",
    tag: "inference",
    title: "Ensemble ML Detection",
    description:
      "Feature vectors pass through a 4-model ensemble: Isolation Forest for outlier isolation, Elliptic Envelope for Mahalanobis distance, One-Class SVM with RBF kernel for boundary detection, and an Autoencoder for reconstruction-error scoring. Weighted ensemble produces a final anomaly score per sighting.",
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
      "Four complementary models vote on each sighting. No single model dominates — the ensemble captures different anomaly geometries that individual models miss.",
  },
  {
    icon: Layers,
    title: "16-Feature Pipeline",
    description:
      "Every sighting produces a rich feature vector: spatial isolation, range deviation, species rarity, temporal patterns, geographic density, and data quality signals.",
  },
  {
    icon: Globe,
    title: "Live iNaturalist Data",
    description:
      "Pulls real-time endangered species observations from the world's largest citizen science platform. No API key required — real data, real species, real locations.",
  },
  {
    icon: Shield,
    title: "Conservation-Aware",
    description:
      "IUCN conservation status is a core feature. Critically endangered species in unexpected locations are flagged with higher severity — prioritizing what matters most.",
  },
  {
    icon: Radio,
    title: "Live Alert Dispatch",
    description:
      "Anomalies are classified into 8 categories (range, temporal, cluster, rarity, captive escape, misidentification, habitat mismatch, poaching indicator) with severity levels.",
  },
  {
    icon: Terminal,
    title: "Pipeline Transparency",
    description:
      "Every stage is timed and logged. See exactly how long data ingestion, feature extraction, model inference, and classification took for each batch.",
  },
  {
    icon: BarChart3,
    title: "Anomaly Visualization",
    description:
      "Interactive dashboard shows anomaly distribution across categories, species cards with photos, model score breakdowns, and detailed evidence for every flagged sighting.",
  },
  {
    icon: Cpu,
    title: "TypeScript ML Engine",
    description:
      "Full machine learning pipeline implemented in TypeScript — no Python dependencies. Runs entirely server-side in Next.js API routes, deployed on Vercel.",
  },
];

const techStack = [
  {
    icon: Brain,
    category: "ML Engine",
    name: "Ensemble Models",
    description:
      "Isolation Forest, Elliptic Envelope, One-Class SVM, and Autoencoder running as a weighted ensemble on 16 wildlife-specific features.",
  },
  {
    icon: Server,
    category: "Frontend + API",
    name: "Next.js 14",
    description:
      "App Router with TypeScript, API routes for detection pipeline, ISR caching, server-side rendering.",
  },
  {
    icon: Database,
    category: "Data Source",
    name: "iNaturalist API",
    description:
      "Real-time endangered species observations from citizen scientists worldwide. No API key. ISR-cached on Vercel.",
  },
  {
    icon: Gauge,
    category: "Infrastructure",
    name: "Vercel",
    description:
      "Edge deployment with automatic scaling, zero config, and instant global distribution.",
  },
];

const faqs = [
  {
    question: "What real problem does Sentinel solve?",
    answer:
      "Sentinel detects anomalous wildlife sightings — endangered animals appearing in unusual locations, at unusual times, or in unusual patterns. This can flag potential poaching displacement, habitat disruption, illegal wildlife trade, captive escapes, or species misidentification.",
  },
  {
    question: "How does the ensemble scoring work?",
    answer:
      "Each model produces an anomaly score between 0 and 1. We weight these scores (IF: 30%, EE: 25%, SVM: 20%, AE: 25%) and combine them into a final ensemble score. Isolation Forest gets the highest weight due to its robustness with high-dimensional data. Rule-based overrides handle conservation-critical edge cases.",
  },
  {
    question: "Where does the data come from?",
    answer:
      "Sentinel uses the iNaturalist API — the world's largest citizen science biodiversity platform with over 150 million observations. It filters for threatened species (IUCN Red List) and returns research-grade observations with verified identifications, GPS coordinates, timestamps, and photos.",
  },
  {
    question: "What wildlife features does Sentinel analyze?",
    answer:
      "Sentinel extracts 16 features per sighting: spatial isolation metrics, distance to species centroid, nearest-neighbor distance, geographic density, range deviation from peer sightings, IUCN rarity score, species frequency, temporal isolation, positional accuracy, quality grade, and captive status.",
  },
  {
    question: "Is this ready for production use?",
    answer:
      "Sentinel is a portfolio project demonstrating real ML engineering applied to a real conservation problem. For production deployment, you'd need species-specific trained models, historical range databases, integration with conservation alert systems, and data persistence.",
  },
];

const apiEndpoints = [
  {
    method: "POST",
    path: "/api/detect",
    description: "Run anomaly detection on wildlife sighting data",
    params: [
      { name: "region", type: "string", optional: true, description: "Region key (global, africa, south_asia, etc.)" },
      { name: "taxon", type: "string", optional: true, description: "Taxon group (Mammalia, Aves, Reptilia, all)" },
      { name: "count", type: "number", optional: true, description: "Number of sightings to fetch (max 50)" },
      { name: "threshold", type: "number", optional: true, description: "Anomaly sensitivity threshold (0.1 - 0.9)" },
      { name: "scenario", type: "string", optional: true, description: "Use fallback data: mixed, normal, range_anomalies, cluster_event, captive_escapes" },
    ],
    example: `curl -X POST https://your-app.vercel.app/api/detect \\
  -H "Content-Type: application/json" \\
  -d '{
    "region": "africa",
    "taxon": "Mammalia",
    "count": 30
  }'`,
  },
  {
    method: "GET",
    path: "/api/detect",
    description: "Get API status, available regions, taxon groups, and anomaly types",
    params: [],
    example: `curl https://your-app.vercel.app/api/detect`,
  },
];

const modelDetails = [
  {
    name: "Isolation Forest",
    abbreviation: "IF",
    description: "Tree-based model that isolates anomalies by random partitioning. Sightings with extreme spatial isolation, rare species, or high range deviation are easy to isolate — producing high anomaly scores.",
    strength: "Fast, handles high-dimensional data well",
    weight: "30%",
  },
  {
    name: "Elliptic Envelope",
    abbreviation: "EE",
    description: "Fits an ellipse around \"normal\" wildlife observations using Mahalanobis distance. Sightings that fall far from the normal centroid across multiple feature dimensions are flagged.",
    strength: "Good for normally distributed data",
    weight: "25%",
  },
  {
    name: "One-Class SVM",
    abbreviation: "SVM",
    description: "Kernel-based method using RBF kernel. Learns a decision boundary around normal sighting patterns — support vectors represent typical observations.",
    strength: "Captures complex non-linear boundaries",
    weight: "20%",
  },
  {
    name: "Autoencoder",
    abbreviation: "AE",
    description: "Neural network that compresses sighting features through a 3D bottleneck and attempts reconstruction. Normal sightings reconstruct well; anomalous ones produce high reconstruction error.",
    strength: "Captures complex patterns, good for mixed data",
    weight: "25%",
  },
];

function FloatingParticles() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <div className="particle particle-1" />
      <div className="particle particle-2" />
      <div className="particle particle-3" />
      <div className="particle particle-4" />
      <div className="particle particle-5" />
      <div className="particle particle-6" />
      <div className="particle particle-large particle-1" style={{ top: '15%', left: '85%' }} />
      <div className="particle particle-large particle-2" style={{ top: '75%', left: '5%' }} />
    </div>
  );
}

function GlowOrbs() {
  return (
    <>
      <div className="glow-orb glow-orb-1" />
      <div className="glow-orb glow-orb-2" />
    </>
  );
}

function RadarBackground() {
  return (
    <div className="fixed inset-0 flex items-center justify-center pointer-events-none -z-10 overflow-hidden">
      <div className="relative w-[400px] h-[400px] opacity-20">
        <div className="radar-ring radar-ring-4" />
        <div className="radar-ring radar-ring-3" />
        <div className="radar-ring radar-ring-2" />
        <div className="radar-ring radar-ring-1" />
        <div className="absolute top-1/2 left-0 right-0 h-px bg-radar-ring" />
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-radar-ring" />
        <div className="radar-sweep" />
        <div className="radar-center" />
        <div className="radar-blip" style={{ top: '25%', left: '60%', animationDelay: '0s' }} />
        <div className="radar-blip" style={{ top: '70%', left: '35%', animationDelay: '0.5s' }} />
        <div className="radar-blip warning" style={{ top: '45%', left: '75%', animationDelay: '1s' }} />
        <div className="radar-blip critical" style={{ top: '80%', left: '70%', animationDelay: '1.5s' }} />
      </div>
    </div>
  );
}

function FAQItem({ question, answer, index }: { question: string; answer: string; index: number }) {
  const [open, setOpen] = useState(false);

  return (
    <motion.div
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      variants={fadeUp}
      custom={index}
      className="radar-card rounded-lg overflow-hidden"
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-6 py-4 flex items-center justify-between text-left"
      >
        <span className="text-sm font-medium text-white pr-4">{question}</span>
        {open ? (
          <ChevronDown className="w-4 h-4 text-zinc-500 flex-shrink-0 rotate-180 transition-transform" />
        ) : (
          <ChevronDown className="w-4 h-4 text-zinc-500 flex-shrink-0 transition-transform" />
        )}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="px-6 pb-4"
          >
            <p className="text-sm text-zinc-400 leading-relaxed">{answer}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function ApiEndpointCard({ endpoint, index }: { endpoint: typeof apiEndpoints[0]; index: number }) {
  const [copied, setCopied] = useState(false);

  const copyCode = () => {
    navigator.clipboard.writeText(endpoint.example);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const methodColors: Record<string, string> = {
    POST: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    GET: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    PUT: "bg-amber-500/20 text-amber-400 border-amber-500/30",
    DELETE: "bg-red-500/20 text-red-400 border-red-500/30",
  };

  return (
    <motion.div
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      variants={fadeUp}
      custom={index}
      className="radar-card rounded-xl overflow-hidden"
    >
      <div className="px-6 py-4 border-b border-radar-green/10">
        <div className="flex items-center gap-3">
          <span className={`px-2 py-0.5 text-xs font-mono font-medium rounded border ${methodColors[endpoint.method]}`}>
            {endpoint.method}
          </span>
          <code className="text-sm text-zinc-300 font-mono">{endpoint.path}</code>
        </div>
        <p className="text-xs text-zinc-500 mt-2">{endpoint.description}</p>
      </div>
      
      {endpoint.params.length > 0 && (
        <div className="px-6 py-4 border-b border-radar-green/10">
          <p className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Parameters</p>
          <div className="space-y-2">
            {endpoint.params.map((param, i) => (
              <div key={i} className="flex items-start gap-2 text-xs">
                <code className="text-radar-green font-mono">{param.name}</code>
                <span className="text-zinc-600">:</span>
                <span className="text-zinc-400 font-mono">{param.type}</span>
                {param.optional && <span className="text-zinc-600">(optional)</span>}
                <span className="text-zinc-500">- {param.description}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="px-6 py-4">
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs text-zinc-500 uppercase tracking-wider">Example</p>
          <button
            onClick={copyCode}
            className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>
        <pre className="text-xs text-zinc-400 font-mono bg-black/20 rounded-lg p-3 overflow-x-auto">
          {endpoint.example}
        </pre>
      </div>
    </motion.div>
  );
}

function ModelCard({ model, index }: { model: typeof modelDetails[0]; index: number }) {
  return (
    <motion.div
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      variants={fadeUp}
      custom={index}
      className="radar-card rounded-xl p-5"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold text-radar-green font-mono">{model.abbreviation}</span>
          <span className="text-sm font-medium text-white">{model.name}</span>
        </div>
        <span className="text-xs text-zinc-500 font-mono">Weight: {model.weight}</span>
      </div>
      <p className="text-xs text-zinc-400 leading-relaxed mb-3">{model.description}</p>
      <div className="flex items-center gap-2 text-xs text-zinc-500">
        <Zap className="w-3 h-3 text-radar-green" />
        <span>{model.strength}</span>
      </div>
    </motion.div>
  );
}

export default function HomePage() {
  return (
    <main className="relative min-h-screen">
      <Navbar />
      <RadarBackground />

      {/* ===== HERO ===== */}
      <section className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-20 dot-grid">
        <FloatingParticles />
        <GlowOrbs />
        
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 px-3 py-1.5 mb-8 rounded-full border border-radar-green/20 bg-radar-greenDim text-xs text-radar-green"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-radar-green animate-pulse" />
            Live iNaturalist Data &middot; ML Pipeline v2.0
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1 }}
            className="text-5xl md:text-7xl font-bold tracking-tight leading-[1.1] mb-6"
          >
            <span className="radar-gradient-text">Sentinel</span>
            <br />
            <span className="text-2xl md:text-3xl font-medium text-zinc-500">Wildlife Guardian</span>
          </motion.h1>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-lg md:text-xl text-zinc-400 max-w-2xl mx-auto mb-10 leading-relaxed"
          >
            Detecting anomalous endangered species sightings using real-time
            data from iNaturalist and a 4-model ML ensemble. Flags range anomalies,
            poaching displacement, and habitat disruption.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex items-center justify-center gap-4 flex-wrap"
          >
            <Link
              href="/dashboard"
              className="group inline-flex items-center gap-2 px-6 py-3 bg-radar-green/20 hover:bg-radar-green/30 text-radar-green border border-radar-green/30 hover:border-radar-green/50 text-sm font-medium rounded-lg transition-all"
            >
              Live Demo
              <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </Link>
            <Link
              href="#docs"
              className="inline-flex items-center gap-2 px-6 py-3 text-zinc-400 hover:text-white text-sm font-medium rounded-lg border border-white/[0.08] hover:border-white/[0.15] hover:bg-white/[0.03] transition-all"
            >
              <BookOpen className="w-4 h-4" />
              API Docs
            </Link>
          </motion.div>
        </div>

        {/* Stats bar */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.5 }}
          className="mt-20 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6 md:gap-8"
        >
          {stats.map((stat, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 + i * 0.1 }}
              className="text-center"
            >
              <div className="flex items-center justify-center gap-2 mb-2">
                <stat.icon className="w-4 h-4 text-radar-green/60" />
              </div>
              <div className="text-xl md:text-2xl font-bold text-white font-mono radar-glow">
                {stat.value}
              </div>
              <div className="text-xs text-zinc-500 mt-1 uppercase tracking-wider">
                {stat.label}
              </div>
            </motion.div>
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
              A three-stage pipeline that transforms live wildlife observations
              into real-time anomaly alerts for conservation monitoring.
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

      {/* ===== MODELS SECTION ===== */}
      <section id="models" className="py-32 px-6 border-t border-radar-green/10">
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
              Ensemble Models
            </motion.p>
            <motion.h2
              variants={fadeUp}
              custom={1}
              className="text-3xl md:text-4xl font-bold radar-gradient-text mb-4"
            >
              Four Complementary Models
            </motion.h2>
            <motion.p
              variants={fadeUp}
              custom={2}
              className="text-zinc-400 max-w-xl"
            >
              Each model captures different anomaly patterns in wildlife data. Combined together,
              they detect range violations, temporal outliers, and rarity spikes that single models miss.
            </motion.p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-4">
            {modelDetails.map((model, i) => (
              <ModelCard key={i} model={model} index={i} />
            ))}
          </div>
        </div>
      </section>

      {/* ===== API DOCS ===== */}
      <section id="docs" className="py-32 px-6 border-t border-radar-green/10">
        <div className="max-w-4xl mx-auto">
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
              API Reference
            </motion.p>
            <motion.h2
              variants={fadeUp}
              custom={1}
              className="text-3xl md:text-4xl font-bold radar-gradient-text mb-4"
            >
              Developer Documentation
            </motion.h2>
            <motion.p
              variants={fadeUp}
              custom={2}
              className="text-zinc-400 max-w-xl"
            >
              Integrate Sentinel&apos;s wildlife anomaly detection into your own
              conservation tools using our REST API.
            </motion.p>
          </motion.div>

          <div className="space-y-6">
            {apiEndpoints.map((endpoint, i) => (
              <ApiEndpointCard key={i} endpoint={endpoint} index={i} />
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
              Engineered for Conservation
            </motion.h2>
            <motion.p
              variants={fadeUp}
              custom={2}
              className="text-zinc-400 max-w-xl"
            >
              Every design decision serves conservation monitoring — from how we
              extract features to how we classify anomaly severity.
            </motion.p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {features.map((feature, i) => (
              <motion.div
                key={i}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-50px" }}
                variants={fadeUp}
                custom={i}
                className="radar-card glass-hover rounded-xl p-5 transition-all duration-300"
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
              application with live data from iNaturalist.
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

      {/* ===== FAQ ===== */}
      <section id="faq" className="py-32 px-6 border-t border-radar-green/10">
        <div className="max-w-3xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            className="mb-16 text-center"
          >
            <motion.p
              variants={fadeUp}
              custom={0}
              className="text-xs text-radar-green uppercase tracking-widest mb-3"
            >
              FAQ
            </motion.p>
            <motion.h2
              variants={fadeUp}
              custom={1}
              className="text-3xl md:text-4xl font-bold radar-gradient-text mb-4"
            >
              Common Questions
            </motion.h2>
          </motion.div>

          <div className="space-y-3">
            {faqs.map((faq, i) => (
              <FAQItem key={i} {...faq} index={i} />
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
            Ready to see wildlife anomaly
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
            Fetch live endangered species data from iNaturalist, run it through
            the 4-model ensemble pipeline, and watch anomalies appear on the
            dashboard in real-time.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="flex items-center justify-center gap-4 flex-wrap"
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
              <Crosshair className="w-3 h-3 text-radar-green" />
            </div>
            <span className="text-sm font-medium text-zinc-400">sentinel wildlife</span>
          </div>
          <p className="text-xs text-zinc-600">
            Wildlife Anomaly Detection &middot; Ensemble ML &middot; iNaturalist
            &middot; Next.js 14 &middot; Vercel
          </p>
        </div>
      </footer>
    </main>
  );
}
