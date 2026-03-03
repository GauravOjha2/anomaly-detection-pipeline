import { NextRequest, NextResponse } from 'next/server';
import { generateWildlifeBatch, FallbackScenario } from '@/lib/mock-data';
import { runDetectionPipeline } from '@/lib/detection-engine';
import { fetchWildlifeSightings, fetchDiverseSightings, REGIONS, TaxonGroup } from '@/lib/inaturalist';
import { WildlifeSighting } from '@/lib/types';

const MAX_SIGHTINGS = 50;
const VALID_SCENARIOS: FallbackScenario[] = ['normal', 'range_anomalies', 'cluster_event', 'captive_escapes', 'mixed'];
const VALID_TAXONS: TaxonGroup[] = ['Mammalia', 'Aves', 'Reptilia', 'Amphibia', 'Actinopterygii', 'all'];

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    let sightings: WildlifeSighting[];
    let dataSource: 'inaturalist_live' | 'fallback_generated' = 'inaturalist_live';
    let region: string | undefined;

    // Mode 1: Direct sighting injection (for custom data / testing)
    if (body.sightings && Array.isArray(body.sightings)) {
      if (body.sightings.length > MAX_SIGHTINGS) {
        return NextResponse.json(
          { error: `Maximum ${MAX_SIGHTINGS} sightings per request. Received ${body.sightings.length}.` },
          { status: 400 }
        );
      }
      if (body.sightings.length < 2) {
        return NextResponse.json(
          { error: 'At least 2 sightings required for anomaly analysis.' },
          { status: 400 }
        );
      }
      sightings = body.sightings as WildlifeSighting[];
      dataSource = 'fallback_generated';
    }
    // Mode 2: Fallback scenario (generated data)
    else if (body.scenario) {
      const scenario = body.scenario as FallbackScenario;
      if (!VALID_SCENARIOS.includes(scenario)) {
        return NextResponse.json(
          { error: `Invalid scenario "${scenario}". Valid: ${VALID_SCENARIOS.join(', ')}` },
          { status: 400 }
        );
      }
      const count = Math.min(Math.max(body.count || 30, 5), MAX_SIGHTINGS);
      sightings = generateWildlifeBatch(scenario, count);
      dataSource = 'fallback_generated';
    }
    // Mode 3: Live iNaturalist data (default)
    else {
      const count = Math.min(Math.max(body.count || 30, 5), MAX_SIGHTINGS);
      const taxon = (body.taxon as TaxonGroup) || 'all';
      const regionKey = (body.region as string) || 'global';

      if (!VALID_TAXONS.includes(taxon)) {
        return NextResponse.json(
          { error: `Invalid taxon "${taxon}". Valid: ${VALID_TAXONS.join(', ')}` },
          { status: 400 }
        );
      }

      const regionConfig = REGIONS[regionKey] || REGIONS.global;
      region = regionConfig.label;

      try {
        if (taxon === 'all') {
          sightings = await fetchDiverseSightings(count);
        } else {
          sightings = await fetchWildlifeSightings({
            taxon_group: taxon,
            threatened_only: true,
            per_page: count,
            place_id: regionConfig.place_id || null,
          });
        }

        // If API returned too few results, supplement with fallback data
        if (sightings.length < 5) {
          const fallback = generateWildlifeBatch('mixed', count);
          sightings = [...sightings, ...fallback].slice(0, count);
          dataSource = sightings.length > 0 ? 'inaturalist_live' : 'fallback_generated';
        }
      } catch (_apiError) {
        // iNaturalist API unavailable — fall back to generated data
        console.warn('iNaturalist API unavailable, using fallback data:', _apiError);
        sightings = generateWildlifeBatch('mixed', count);
        dataSource = 'fallback_generated';
      }
    }

    // Extract threshold override
    const threshold = typeof body.threshold === 'number'
      ? Math.min(Math.max(body.threshold, 0.1), 0.9)
      : undefined;

    // Run detection pipeline
    const result = await runDetectionPipeline(sightings, threshold, dataSource, region);

    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: 'Detection pipeline failed', details: String(error) },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    status: 'operational',
    pipeline: 'sentinel-wildlife-v2.0',
    description: 'Wildlife anomaly detection pipeline — detects anomalous endangered species sightings using a 4-model ML ensemble.',
    data_source: 'iNaturalist API (real-time, no API key required)',
    models: ['Isolation Forest', 'Elliptic Envelope', 'One-Class SVM', 'Autoencoder'],
    ensemble_weights: { isolation_forest: 0.30, elliptic_envelope: 0.25, svm: 0.20, autoencoder: 0.25 },
    features: 16,
    anomaly_types: [
      'RANGE_ANOMALY', 'TEMPORAL_ANOMALY', 'CLUSTER_ANOMALY', 'RARITY_ANOMALY',
      'CAPTIVE_ESCAPE', 'MISIDENTIFICATION', 'HABITAT_MISMATCH', 'POACHING_INDICATOR',
    ],
    limits: { max_sightings: MAX_SIGHTINGS, min_sightings: 2 },
    scenarios: VALID_SCENARIOS,
    regions: Object.entries(REGIONS).map(([key, val]) => ({ key, label: val.label, description: val.description })),
    taxon_groups: VALID_TAXONS,
    endpoints: {
      POST: {
        live: 'POST /api/detect { count?, taxon?, region?, threshold? } — Fetch live iNaturalist data and run detection.',
        scenario: 'POST /api/detect { scenario, count?, threshold? } — Use generated fallback data.',
        inject: 'POST /api/detect { sightings: [...], threshold? } — Inject custom sighting data.',
      },
      GET: 'GET /api/detect — This status endpoint.',
    },
  });
}
