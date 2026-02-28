import { NextRequest, NextResponse } from 'next/server';
import { generateTelemetryBatch, Scenario } from '@/lib/mock-data';
import { runDetectionPipeline } from '@/lib/detection-engine';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const scenario: Scenario = body.scenario || 'mixed';
    const count: number = body.count || 30;
    const touristCount: number = body.tourist_count || 5;

    // Stage 1: Generate telemetry
    const telemetry = generateTelemetryBatch(scenario, count, touristCount);

    // Stage 2: Run detection pipeline
    const result = await runDetectionPipeline(telemetry);

    return NextResponse.json({
      ...result,
      telemetry_sample: telemetry.slice(0, 5), // Return a sample for display
      scenario,
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
      POST: '/api/detect - Run anomaly detection pipeline',
    },
  });
}
