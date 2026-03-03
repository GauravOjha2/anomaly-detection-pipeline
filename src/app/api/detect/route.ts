import { NextRequest, NextResponse } from 'next/server';
import { generateTelemetryBatch, Scenario } from '@/lib/mock-data';
import { runDetectionPipeline } from '@/lib/detection-engine';
import { TelemetryEvent } from '@/lib/types';

const VALID_SCENARIOS: Scenario[] = ['normal', 'emergency', 'health_anomaly', 'device_failure', 'extreme', 'mixed'];
const MAX_EVENTS = 100;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    let telemetry: TelemetryEvent[];

    if (body.telemetry && Array.isArray(body.telemetry)) {
      // Direct injection mode — accept raw telemetry array
      if (body.telemetry.length > MAX_EVENTS) {
        return NextResponse.json(
          { error: `Maximum ${MAX_EVENTS} telemetry events allowed per request. Received ${body.telemetry.length}.` },
          { status: 400 }
        );
      }
      if (body.telemetry.length < 2) {
        return NextResponse.json(
          { error: 'At least 2 telemetry events required for trajectory analysis.' },
          { status: 400 }
        );
      }
      telemetry = body.telemetry as TelemetryEvent[];
    } else {
      // Scenario mode — generate from preset scenarios
      const scenario: Scenario = body.scenario || 'mixed';
      if (!VALID_SCENARIOS.includes(scenario)) {
        return NextResponse.json(
          { error: `Invalid scenario "${scenario}". Valid: ${VALID_SCENARIOS.join(', ')}` },
          { status: 400 }
        );
      }
      const count: number = Math.min(Math.max(body.count || 30, 5), MAX_EVENTS);
      const touristCount: number = Math.min(Math.max(body.tourist_count || 5, 1), 10);
      telemetry = generateTelemetryBatch(scenario, count, touristCount);
    }

    // Extract threshold override if provided
    const threshold = typeof body.threshold === 'number'
      ? Math.min(Math.max(body.threshold, 0.1), 0.9)
      : undefined;

    // Run detection pipeline
    const result = await runDetectionPipeline(telemetry, threshold);

    return NextResponse.json({
      ...result,
      telemetry_count: telemetry.length,
      source: body.telemetry ? 'injected' : 'generated',
      scenario: body.scenario || null,
    });
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
    pipeline: 'sentinel-v2.0',
    models: ['Isolation Forest', 'Elliptic Envelope', 'One-Class SVM', 'Autoencoder'],
    ensemble_weights: { isolation_forest: 0.30, elliptic_envelope: 0.25, svm: 0.20, autoencoder: 0.25 },
    features: 20,
    limits: { max_events: MAX_EVENTS, min_events: 2 },
    scenarios: VALID_SCENARIOS,
    endpoints: {
      POST: '/api/detect — Send { scenario, count, threshold? } for generated data, or { telemetry: [...], threshold? } for direct injection.',
      GET: '/api/detect — This status endpoint.',
    },
  });
}
