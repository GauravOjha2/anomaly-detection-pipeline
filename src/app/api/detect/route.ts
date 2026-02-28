import { NextRequest, NextResponse } from 'next/server';
import { generateTelemetryBatch, Scenario } from '@/lib/mock-data';
import { runDetectionPipeline } from '@/lib/detection-engine';
import { TelemetryEvent } from '@/lib/types';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    let telemetry: TelemetryEvent[];

    if (body.telemetry && Array.isArray(body.telemetry)) {
      // Direct injection mode — accept raw telemetry array
      telemetry = body.telemetry as TelemetryEvent[];
    } else {
      // Scenario mode — generate from preset scenarios
      const scenario: Scenario = body.scenario || 'mixed';
      const count: number = body.count || 30;
      const touristCount: number = body.tourist_count || 5;
      telemetry = generateTelemetryBatch(scenario, count, touristCount);
    }

    // Run detection pipeline
    const result = await runDetectionPipeline(telemetry);

    return NextResponse.json({
      ...result,
      telemetry_sample: telemetry.slice(0, 5),
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
    features: 20,
    endpoints: {
      POST: '/api/detect — Run pipeline. Send { scenario, count } for generated data, or { telemetry: [...] } for direct injection.',
    },
  });
}
