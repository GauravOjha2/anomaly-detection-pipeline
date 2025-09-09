#!/usr/bin/env python3
"""
Test Anomaly Detection Pipeline with Mock Data
==============================================

This script demonstrates how to integrate mock data with your anomaly detection pipeline.
It generates various scenarios and tests the detection capabilities.

Usage:
    python test_with_mock_data.py --scenario emergency
    python test_with_mock_data.py --scenario mixed --count 500
    python test_with_mock_data.py --test-api  # Test with live API
"""

import asyncio
import httpx
import pandas as pd
import json
import argparse
from datetime import datetime, timezone
from mock_data_generator import TouristDataGenerator
import time

class MockDataTester:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.generator = TouristDataGenerator()
    
    def test_detection_offline(self, data: list, scenario_name: str):
        """Test detection using the anomaly_detector module directly"""
        print(f"\n🔬 Testing {scenario_name} scenario (Offline Mode)...")
        
        try:
            from anomaly_detector import detect_anomalies
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Detect anomalies
            anomaly_indices, scores, alerts = detect_anomalies(df, return_alerts=True)
            
            print(f"📊 Results:")
            print(f"  Total data points: {len(data)}")
            print(f"  Anomalies detected: {len(anomaly_indices)}")
            print(f"  Detection rate: {len(anomaly_indices)/len(data)*100:.1f}%")
            
            if len(anomaly_indices) > 0:
                print(f"\n🚨 Detected Anomalies:")
                for i, idx in enumerate(anomaly_indices[:10]):  # Show first 10
                    row = df.iloc[idx]
                    alert = alerts[i] if i < len(alerts) else {}
                    print(f"  {i+1}. Row {idx}: {alert.get('anomaly_type', 'UNKNOWN')} "
                          f"(confidence: {alert.get('confidence_score', 0):.2f})")
                    if row.get('panic_button'):
                        print(f"     🚨 PANIC BUTTON PRESSED!")
                    if row.get('heart_rate', 0) > 150:
                        print(f"     ❤️  High heart rate: {row['heart_rate']} BPM")
                    if row.get('battery_level', 100) < 20:
                        print(f"     🔋 Low battery: {row['battery_level']}%")
            
            return len(anomaly_indices), len(data)
            
        except Exception as e:
            print(f"❌ Error in offline testing: {e}")
            return 0, len(data)
    
    async def test_detection_api(self, data: list, scenario_name: str):
        """Test detection using the live API"""
        print(f"\n🌐 Testing {scenario_name} scenario (API Mode)...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Test single detection
                print("  📊 Testing single detection...")
                response = await client.post(f"{self.api_url}/detect", json=data[0])
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"    ✅ Single detection: {result['anomaly_count']} anomalies")
                else:
                    print(f"    ❌ Single detection failed: {response.status_code}")
                    return 0, len(data)
                
                # Test batch detection
                print("  📊 Testing batch detection...")
                response = await client.post(f"{self.api_url}/detect_batch", json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"    ✅ Batch detection: {result['anomaly_count']} anomalies")
                    
                    if result['anomaly_count'] > 0:
                        print(f"\n    🚨 Detected Alerts:")
                        for i, alert in enumerate(result['alerts'][:5]):  # Show first 5
                            print(f"      {i+1}. {alert['anomaly_type']} "
                                  f"({alert['alert_level']}) - "
                                  f"Confidence: {alert['confidence_score']:.2f}")
                    
                    return result['anomaly_count'], len(data)
                else:
                    print(f"    ❌ Batch detection failed: {response.status_code}")
                    return 0, len(data)
                    
            except httpx.ConnectError:
                print(f"    ❌ Cannot connect to API at {self.api_url}")
                print(f"    💡 Make sure the API is running: python main.py")
                return 0, len(data)
            except Exception as e:
                print(f"    ❌ API test error: {e}")
                return 0, len(data)
    
    def generate_test_scenarios(self, count: int = 200):
        """Generate various test scenarios"""
        scenarios = {}
        
        print("🎭 Generating test scenarios...")
        
        # Normal scenario
        scenarios['normal'] = self.generator.generate_normal_trajectory(duration_hours=count/30)
        print(f"  ✅ Normal: {len(scenarios['normal'])} points")
        
        # Emergency scenario
        scenarios['emergency'] = self.generator.generate_emergency_scenario(duration_hours=count/30)
        print(f"  ✅ Emergency: {len(scenarios['emergency'])} points")
        
        # Health anomaly scenario
        scenarios['health'] = self.generator.generate_health_anomaly_scenario(duration_hours=count/30)
        print(f"  ✅ Health: {len(scenarios['health'])} points")
        
        # Device failure scenario
        scenarios['device'] = self.generator.generate_device_failure_scenario(duration_hours=count/30)
        print(f"  ✅ Device: {len(scenarios['device'])} points")
        
        # Mixed scenario
        scenarios['mixed'] = self.generator.generate_mixed_scenario(count=count)
        print(f"  ✅ Mixed: {len(scenarios['mixed'])} points")
        
        return scenarios
    
    def analyze_scenario(self, data: list, scenario_name: str):
        """Analyze a scenario to understand what anomalies should be detected"""
        print(f"\n📈 Analyzing {scenario_name} scenario...")
        
        df = pd.DataFrame(data)
        
        # Count different types of anomalies
        panic_count = df['panic_button'].sum() if 'panic_button' in df.columns else 0
        high_hr_count = (df['heart_rate'] > 150).sum() if 'heart_rate' in df.columns else 0
        low_battery_count = (df['battery_level'] < 20).sum() if 'battery_level' in df.columns else 0
        poor_network_count = df['network_status'].isin(['poor', 'no_signal']).sum() if 'network_status' in df.columns else 0
        low_accuracy_count = (df['accuracy'] > 15).sum() if 'accuracy' in df.columns else 0
        
        print(f"  📊 Expected anomalies:")
        print(f"    Panic button presses: {panic_count}")
        print(f"    High heart rate (>150): {high_hr_count}")
        print(f"    Low battery (<20%): {low_battery_count}")
        print(f"    Poor network: {poor_network_count}")
        print(f"    Low accuracy (>15m): {low_accuracy_count}")
        
        return {
            'panic': panic_count,
            'high_hr': high_hr_count,
            'low_battery': low_battery_count,
            'poor_network': poor_network_count,
            'low_accuracy': low_accuracy_count
        }

async def main():
    parser = argparse.ArgumentParser(description="Test anomaly detection with mock data")
    parser.add_argument("--scenario", choices=["normal", "emergency", "health", "device", "mixed", "all"], 
                       default="all", help="Scenario to test")
    parser.add_argument("--count", type=int, default=200, help="Number of data points")
    parser.add_argument("--test-api", action="store_true", help="Test with live API instead of offline")
    parser.add_argument("--save-data", action="store_true", help="Save generated data to files")
    
    args = parser.parse_args()
    
    print("🧪 Mock Data Testing Suite")
    print("=" * 50)
    
    tester = MockDataTester()
    
    if args.scenario == "all":
        scenarios = tester.generate_test_scenarios(args.count)
    else:
        generator = TouristDataGenerator()
        if args.scenario == "normal":
            data = generator.generate_normal_trajectory(duration_hours=args.count/30)
        elif args.scenario == "emergency":
            data = generator.generate_emergency_scenario(duration_hours=args.count/30)
        elif args.scenario == "health":
            data = generator.generate_health_anomaly_scenario(duration_hours=args.count/30)
        elif args.scenario == "device":
            data = generator.generate_device_failure_scenario(duration_hours=args.count/30)
        else:  # mixed
            data = generator.generate_mixed_scenario(count=args.count)
        
        scenarios = {args.scenario: data}
    
    # Save data if requested
    if args.save_data:
        for name, data in scenarios.items():
            filename = f"test_data_{name}_{args.count}points.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Saved {name} data to {filename}")
    
    # Test each scenario
    total_anomalies = 0
    total_points = 0
    
    for scenario_name, data in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Testing Scenario: {scenario_name.upper()}")
        print(f"{'='*60}")
        
        # Analyze expected anomalies
        expected = tester.analyze_scenario(data, scenario_name)
        
        # Test detection
        if args.test_api:
            anomalies, points = await tester.test_detection_api(data, scenario_name)
        else:
            anomalies, points = tester.test_detection_offline(data, scenario_name)
        
        total_anomalies += anomalies
        total_points += points
        
        # Calculate detection accuracy
        expected_total = sum(expected.values())
        if expected_total > 0:
            accuracy = (anomalies / expected_total) * 100 if anomalies <= expected_total else 100
            print(f"🎯 Detection accuracy: {accuracy:.1f}% ({anomalies}/{expected_total})")
        else:
            print(f"🎯 No expected anomalies, detected: {anomalies}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total data points tested: {total_points}")
    print(f"Total anomalies detected: {total_anomalies}")
    print(f"Overall detection rate: {total_anomalies/total_points*100:.1f}%")
    
    if args.test_api:
        print(f"\n💡 API Integration:")
        print(f"  - Anomaly Detection API: http://localhost:8000")
        print(f"  - Alert System API: http://localhost:8001")
        print(f"  - API Documentation: http://localhost:8000/docs")
    else:
        print(f"\n💡 To test with live API:")
        print(f"  1. Start the pipeline: python main.py")
        print(f"  2. Run: python test_with_mock_data.py --test-api")

if __name__ == "__main__":
    asyncio.run(main())
