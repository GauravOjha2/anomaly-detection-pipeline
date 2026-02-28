// Mock Data Generator - Generates realistic tourist telemetry scenarios
// Scenarios: normal, emergency, health_anomaly, device_failure, extreme, mixed

import { TelemetryEvent } from './types';

const BEIJING_CENTER = { lat: 39.9042, lng: 116.4074 };

const TOURIST_NAMES = [
  'T-001', 'T-002', 'T-003', 'T-004', 'T-005',
  'T-006', 'T-007', 'T-008', 'T-009', 'T-010',
];

const LANDMARKS = [
  { name: 'Forbidden City', lat: 39.9163, lng: 116.3972 },
  { name: 'Temple of Heaven', lat: 39.8822, lng: 116.4066 },
  { name: 'Summer Palace', lat: 39.9998, lng: 116.2755 },
  { name: "Tiananmen Square", lat: 39.9055, lng: 116.3976 },
  { name: 'Great Wall (Badaling)', lat: 40.3597, lng: 116.0195 },
  { name: 'Olympic Park', lat: 39.9929, lng: 116.3965 },
  { name: '798 Art District', lat: 39.9841, lng: 116.4953 },
  { name: 'Wangfujing', lat: 39.9148, lng: 116.4107 },
];

function randomBetween(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

function addNoise(value: number, noise: number): number {
  return value + (Math.random() - 0.5) * 2 * noise;
}

function generateNormalTrajectory(touristId: string, count: number, startTime: Date): TelemetryEvent[] {
  const events: TelemetryEvent[] = [];
  const startLandmark = LANDMARKS[Math.floor(Math.random() * LANDMARKS.length)];
  let lat = startLandmark.lat;
  let lng = startLandmark.lng;
  let heartRate = randomBetween(65, 85);
  let battery = randomBetween(70, 100);
  let time = new Date(startTime);

  for (let i = 0; i < count; i++) {
    // Walking speed: ~3-5 km/h = ~0.00003-0.00005 degrees/second
    const stepSize = randomBetween(0.00003, 0.00008);
    const angle = Math.random() * 2 * Math.PI;
    lat += Math.cos(angle) * stepSize * randomBetween(10, 60);
    lng += Math.sin(angle) * stepSize * randomBetween(10, 60);

    heartRate = Math.max(55, Math.min(110, heartRate + randomBetween(-3, 3)));
    battery = Math.max(20, battery - randomBetween(0.01, 0.05));
    time = new Date(time.getTime() + randomBetween(15000, 90000));

    events.push({
      tourist_id: touristId,
      lat: addNoise(lat, 0.0001),
      lng: addNoise(lng, 0.0001),
      timestamp: time.toISOString(),
      heart_rate: Math.round(heartRate),
      battery_level: Math.round(battery * 10) / 10,
      network_status: 'connected',
      panic_button: false,
      accuracy: randomBetween(3, 15),
      altitude: randomBetween(40, 60),
    });
  }

  return events;
}

function generateEmergencyScenario(touristId: string, count: number, startTime: Date): TelemetryEvent[] {
  const events: TelemetryEvent[] = [];
  const startLandmark = LANDMARKS[Math.floor(Math.random() * LANDMARKS.length)];
  let lat = startLandmark.lat;
  let lng = startLandmark.lng;
  let heartRate = 80;
  let battery = randomBetween(30, 60);
  let time = new Date(startTime);

  const panicPoint = Math.floor(count * 0.6);

  for (let i = 0; i < count; i++) {
    const phase = i / count;
    
    // Escalating speed
    const speedMultiplier = phase < 0.5 ? 1 : 1 + (phase - 0.5) * 20;
    const stepSize = randomBetween(0.00005, 0.0001) * speedMultiplier;
    const angle = randomBetween(-0.3, 0.3) + (i > panicPoint ? Math.random() * Math.PI : 0);
    
    lat += Math.cos(angle) * stepSize * randomBetween(20, 60);
    lng += Math.sin(angle) * stepSize * randomBetween(20, 60);

    // Escalating heart rate
    heartRate = phase < 0.4 ? 80 + randomBetween(-5, 5) :
      Math.min(185, 80 + (phase - 0.4) * 200 + randomBetween(-5, 5));

    battery = Math.max(5, battery - randomBetween(0.05, 0.2));
    time = new Date(time.getTime() + randomBetween(5000, 30000));

    events.push({
      tourist_id: touristId,
      lat: addNoise(lat, 0.0002),
      lng: addNoise(lng, 0.0002),
      timestamp: time.toISOString(),
      heart_rate: Math.round(heartRate),
      battery_level: Math.round(battery * 10) / 10,
      network_status: phase > 0.7 ? 'degraded' : 'connected',
      panic_button: i >= panicPoint,
      accuracy: randomBetween(5, phase > 0.6 ? 50 : 15),
      altitude: randomBetween(35, 70),
    });
  }

  return events;
}

function generateHealthAnomaly(touristId: string, count: number, startTime: Date): TelemetryEvent[] {
  const events: TelemetryEvent[] = [];
  const startLandmark = LANDMARKS[Math.floor(Math.random() * LANDMARKS.length)];
  let lat = startLandmark.lat;
  let lng = startLandmark.lng;
  let heartRate = 75;
  let battery = randomBetween(50, 90);
  let time = new Date(startTime);

  for (let i = 0; i < count; i++) {
    const phase = i / count;

    // Gradually slowing movement
    const stepSize = randomBetween(0.00002, 0.00006) * (1 - phase * 0.8);
    const angle = Math.random() * 2 * Math.PI;
    lat += Math.cos(angle) * stepSize * 30;
    lng += Math.sin(angle) * stepSize * 30;

    // Heart rate: normal -> gradually high -> critical
    if (phase < 0.3) {
      heartRate = 75 + randomBetween(-5, 5);
    } else if (phase < 0.6) {
      heartRate = 75 + (phase - 0.3) * 250 + randomBetween(-5, 5);
    } else {
      heartRate = Math.min(170, 150 + randomBetween(-10, 20));
    }

    battery = Math.max(10, battery - randomBetween(0.02, 0.08));
    time = new Date(time.getTime() + randomBetween(20000, 120000));

    events.push({
      tourist_id: touristId,
      lat,
      lng,
      timestamp: time.toISOString(),
      heart_rate: Math.round(heartRate),
      battery_level: Math.round(battery * 10) / 10,
      network_status: 'connected',
      panic_button: phase > 0.85,
      accuracy: randomBetween(3, 12),
      altitude: randomBetween(40, 55),
    });
  }

  return events;
}

function generateDeviceFailure(touristId: string, count: number, startTime: Date): TelemetryEvent[] {
  const events: TelemetryEvent[] = [];
  let lat = BEIJING_CENTER.lat + randomBetween(-0.05, 0.05);
  let lng = BEIJING_CENTER.lng + randomBetween(-0.05, 0.05);
  let battery = randomBetween(40, 70);
  let time = new Date(startTime);

  for (let i = 0; i < count; i++) {
    const phase = i / count;

    const stepSize = randomBetween(0.00003, 0.00007);
    const angle = Math.random() * 2 * Math.PI;
    lat += Math.cos(angle) * stepSize * 30;
    lng += Math.sin(angle) * stepSize * 30;

    // Accelerating battery drain
    battery = Math.max(0, battery - (0.1 + phase * 2));

    time = new Date(time.getTime() + randomBetween(15000, 60000));

    const networkStatuses: string[] = phase < 0.3 ? ['connected'] :
      phase < 0.6 ? ['connected', 'degraded'] :
        ['degraded', 'disconnected'];

    events.push({
      tourist_id: touristId,
      lat: addNoise(lat, 0.0001 + phase * 0.005), // GPS accuracy degrades
      lng: addNoise(lng, 0.0001 + phase * 0.005),
      timestamp: time.toISOString(),
      heart_rate: Math.round(randomBetween(60, 90)),
      battery_level: Math.round(battery * 10) / 10,
      network_status: networkStatuses[Math.floor(Math.random() * networkStatuses.length)],
      panic_button: false,
      accuracy: randomBetween(5, 10 + phase * 100),
      altitude: randomBetween(30, 80),
    });
  }

  return events;
}

function generateExtremeScenario(touristId: string, count: number, startTime: Date): TelemetryEvent[] {
  const events: TelemetryEvent[] = [];
  let lat = BEIJING_CENTER.lat + randomBetween(-0.03, 0.03);
  let lng = BEIJING_CENTER.lng + randomBetween(-0.03, 0.03);
  let time = new Date(startTime);

  for (let i = 0; i < count; i++) {
    // Extreme, erratic movement
    const stepSize = randomBetween(0.001, 0.01); // Very large jumps
    const angle = Math.random() * 2 * Math.PI;
    lat += Math.cos(angle) * stepSize;
    lng += Math.sin(angle) * stepSize;

    time = new Date(time.getTime() + randomBetween(2000, 10000));

    events.push({
      tourist_id: touristId,
      lat,
      lng,
      timestamp: time.toISOString(),
      heart_rate: Math.round(randomBetween(30, 190)),
      battery_level: Math.round(randomBetween(0, 15) * 10) / 10,
      network_status: Math.random() > 0.5 ? 'disconnected' : 'degraded',
      panic_button: Math.random() > 0.6,
      accuracy: randomBetween(50, 200),
      altitude: randomBetween(0, 200),
    });
  }

  return events;
}

export type Scenario = 'normal' | 'emergency' | 'health_anomaly' | 'device_failure' | 'extreme' | 'mixed';

export function generateTelemetryBatch(
  scenario: Scenario = 'mixed',
  count: number = 30,
  touristCount: number = 5
): TelemetryEvent[] {
  const startTime = new Date();
  startTime.setMinutes(startTime.getMinutes() - count * 2); // Start in the past
  
  const allEvents: TelemetryEvent[] = [];

  if (scenario === 'mixed') {
    // Generate a mix of scenarios across different tourists
    const scenarios: Scenario[] = ['normal', 'normal', 'emergency', 'health_anomaly', 'device_failure', 'extreme'];
    const eventsPerTourist = Math.ceil(count / Math.min(touristCount, scenarios.length));

    for (let i = 0; i < Math.min(touristCount, scenarios.length); i++) {
      const touristId = TOURIST_NAMES[i % TOURIST_NAMES.length];
      const s = scenarios[i % scenarios.length];
      const offset = new Date(startTime.getTime() + i * 60000);

      switch (s) {
        case 'normal':
          allEvents.push(...generateNormalTrajectory(touristId, eventsPerTourist, offset));
          break;
        case 'emergency':
          allEvents.push(...generateEmergencyScenario(touristId, eventsPerTourist, offset));
          break;
        case 'health_anomaly':
          allEvents.push(...generateHealthAnomaly(touristId, eventsPerTourist, offset));
          break;
        case 'device_failure':
          allEvents.push(...generateDeviceFailure(touristId, eventsPerTourist, offset));
          break;
        case 'extreme':
          allEvents.push(...generateExtremeScenario(touristId, eventsPerTourist, offset));
          break;
      }
    }
  } else {
    const eventsPerTourist = Math.ceil(count / touristCount);
    for (let i = 0; i < touristCount; i++) {
      const touristId = TOURIST_NAMES[i % TOURIST_NAMES.length];
      const offset = new Date(startTime.getTime() + i * 60000);

      switch (scenario) {
        case 'normal':
          allEvents.push(...generateNormalTrajectory(touristId, eventsPerTourist, offset));
          break;
        case 'emergency':
          allEvents.push(...generateEmergencyScenario(touristId, eventsPerTourist, offset));
          break;
        case 'health_anomaly':
          allEvents.push(...generateHealthAnomaly(touristId, eventsPerTourist, offset));
          break;
        case 'device_failure':
          allEvents.push(...generateDeviceFailure(touristId, eventsPerTourist, offset));
          break;
        case 'extreme':
          allEvents.push(...generateExtremeScenario(touristId, eventsPerTourist, offset));
          break;
      }
    }
  }

  // Sort by timestamp
  allEvents.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

  return allEvents;
}
