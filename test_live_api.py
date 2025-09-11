#!/usr/bin/env python3
"""
Test Live API with Mock Data
============================
Send mock data to the live API to generate anomalies
"""

import requests
import json
import time
from datetime import datetime

def test_live_api():
    """Test the live API with mock data"""
    print("🚀 Testing Live API with Mock Data")
    print("=" * 50)
    
    # Test data that should trigger anomalies
    test_scenarios = [
        {
            "name": "Normal Tourist",
            "data": {
                "tourist_id": "tourist_001",
                "lat": 40.7128,
                "lng": -74.0060,
                "timestamp": datetime.now().isoformat(),
                "speed_kmh": 25.0,
                "heart_rate": 75,
                "panic_button": False,
                "battery_level": 85
            }
        },
        {
            "name": "Emergency - Panic Button",
            "data": {
                "tourist_id": "tourist_002",
                "lat": 40.7589,
                "lng": -73.9851,
                "timestamp": datetime.now().isoformat(),
                "speed_kmh": 0.0,
                "heart_rate": 160,
                "panic_button": True,
                "battery_level": 15
            }
        },
        {
            "name": "High Speed Anomaly",
            "data": {
                "tourist_id": "tourist_003",
                "lat": 40.7505,
                "lng": -73.9934,
                "timestamp": datetime.now().isoformat(),
                "speed_kmh": 120.0,
                "heart_rate": 140,
                "panic_button": False,
                "battery_level": 45
            }
        },
        {
            "name": "Health Emergency",
            "data": {
                "tourist_id": "tourist_004",
                "lat": 40.7614,
                "lng": -73.9776,
                "timestamp": datetime.now().isoformat(),
                "speed_kmh": 5.0,
                "heart_rate": 180,
                "panic_button": False,
                "battery_level": 10
            }
        }
    ]
    
    total_anomalies = 0
    
    for scenario in test_scenarios:
        print(f"\n📤 Testing: {scenario['name']}")
        print(f"   Tourist: {scenario['data']['tourist_id']}")
        print(f"   Speed: {scenario['data']['speed_kmh']} km/h")
        print(f"   Heart Rate: {scenario['data']['heart_rate']} bpm")
        print(f"   Panic Button: {scenario['data']['panic_button']}")
        print(f"   Battery: {scenario['data']['battery_level']}%")
        
        try:
            # Send to anomaly detection API
            response = requests.post(
                "http://localhost:8000/detect",
                json=scenario['data'],
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Response: {result['status']}")
                print(f"   📊 Anomalies: {result.get('anomaly_count', 0)}")
                
                if result.get("anomaly_count", 0) > 0:
                    print("   🚨 Alerts detected!")
                    for alert in result.get("alerts", []):
                        print(f"      - {alert['anomaly_type']}: {alert['alert_level']} (Confidence: {alert['confidence_score']:.2f})")
                    total_anomalies += result.get('anomaly_count', 0)
                else:
                    print("   ℹ️  No anomalies detected")
            else:
                print(f"   ❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    print(f"\n📊 Total Anomalies Detected: {total_anomalies}")
    
    # Check alerts in the system
    print("\n🔍 Checking stored alerts...")
    try:
        response = requests.get("http://localhost:8001/alerts")
        if response.status_code == 200:
            alerts_data = response.json()
            print(f"✅ Alert System: {alerts_data.get('status', 'unknown')}")
            print(f"📋 Total Alerts in System: {len(alerts_data.get('alerts', []))}")
            
            if alerts_data.get('alerts'):
                print("\n📋 Recent Alerts:")
                for alert in alerts_data['alerts'][:3]:  # Show first 3
                    print(f"   - {alert['anomaly_type']} ({alert['alert_level']}) - {alert['tourist_id']}")
        else:
            print(f"❌ Alert System Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Alert System Check Failed: {e}")

if __name__ == "__main__":
    test_live_api()
    
    print("\n🌐 Access your system:")
    print("   • Dashboard: http://localhost:8001/")
    print("   • API Docs: http://localhost:8000/docs")
    print("   • Alerts: http://localhost:8001/alerts")
