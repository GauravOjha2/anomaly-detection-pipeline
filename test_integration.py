#!/usr/bin/env python3
"""
Test script for the integrated anomaly detection pipeline
"""

import asyncio
import httpx
import json
from datetime import datetime, timezone

# Test data
test_events = [
    {
        "tourist_id": "test_001",
        "lat": 39.9042,
        "lng": 116.4074,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heart_rate": 75,
        "battery_level": 85,
        "network_status": "good",
        "panic_button": False,
        "accuracy": 5.0
    },
    {
        "tourist_id": "test_002",
        "lat": 39.9043,
        "lng": 116.4075,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heart_rate": 160,  # High heart rate - should trigger alert
        "battery_level": 15,
        "network_status": "poor",
        "panic_button": False,
        "accuracy": 10.0
    },
    {
        "tourist_id": "test_003",
        "lat": 39.9044,
        "lng": 116.4076,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heart_rate": 80,
        "battery_level": 5,  # Low battery - should trigger alert
        "network_status": "good",
        "panic_button": True,  # Panic button - should trigger critical alert
        "accuracy": 3.0
    }
]

async def test_anomaly_detection():
    """Test the anomaly detection API"""
    print("🧪 Testing Anomaly Detection API...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test single detection
        print("  📊 Testing single detection...")
        response = await client.post(
            "http://localhost:8000/detect",
            json=test_events[0]
        )
        print(f"    Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"    Anomalies found: {data['anomaly_count']}")
        else:
            print(f"    Error: {response.text}")
        
        # Test batch detection
        print("  📊 Testing batch detection...")
        response = await client.post(
            "http://localhost:8000/detect_batch",
            json=test_events
        )
        print(f"    Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"    Anomalies found: {data['anomaly_count']}")
            for i, alert in enumerate(data.get('alerts', [])):
                print(f"      Alert {i+1}: {alert['anomaly_type']} ({alert['alert_level']})")
        else:
            print(f"    Error: {response.text}")
        
        # Test health check
        print("  🏥 Testing health check...")
        response = await client.get("http://localhost:8000/health")
        print(f"    Status: {response.status_code}")
        if response.status_code == 200:
            print(f"    Health: {response.json()}")

async def test_alert_system():
    """Test the alert system API"""
    print("\n🚨 Testing Alert System API...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test health check
        print("  🏥 Testing health check...")
        try:
            response = await client.get("http://localhost:8001/health")
            print(f"    Status: {response.status_code}")
            if response.status_code == 200:
                print(f"    Health: {response.json()}")
        except httpx.ConnectError:
            print("    ❌ Alert system not running")
            return
        
        # Test alerts endpoint
        print("  📋 Testing alerts list...")
        try:
            response = await client.get("http://localhost:8001/alerts")
            print(f"    Status: {response.status_code}")
            if response.status_code == 200:
                alerts = response.json()
                print(f"    Total alerts: {len(alerts)}")
        except httpx.ConnectError:
            print("    ❌ Alert system not running")

async def test_integration():
    """Test the full integration"""
    print("\n🔗 Testing Full Integration...")
    
    # Wait a moment for alerts to be processed
    print("  ⏳ Waiting for alerts to be processed...")
    await asyncio.sleep(2)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Check if alerts were created
            response = await client.get("http://localhost:8001/alerts")
            if response.status_code == 200:
                alerts = response.json()
                print(f"  📊 Total alerts in system: {len(alerts)}")
                
                if alerts:
                    print("  🚨 Recent alerts:")
                    for alert in alerts[:3]:  # Show first 3 alerts
                        print(f"    - {alert.get('anomaly_type')} ({alert.get('alert_level')}) - {alert.get('tourist_id')}")
                else:
                    print("  ℹ️  No alerts found (this is normal if no anomalies were detected)")
            else:
                print(f"  ❌ Failed to get alerts: {response.status_code}")
        except httpx.ConnectError:
            print("  ❌ Alert system not running")

async def main():
    """Main test function"""
    print("🚀 Integration Test Suite")
    print("=" * 50)
    
    # Test anomaly detection
    await test_anomaly_detection()
    
    # Test alert system
    await test_alert_system()
    
    # Test integration
    await test_integration()
    
    print("\n✅ Test suite completed!")
    print("\n💡 Tips:")
    print("  - Make sure both services are running (python main.py)")
    print("  - Check logs for detailed information")
    print("  - Visit http://localhost:8001 for the alert dashboard")

if __name__ == "__main__":
    asyncio.run(main())
