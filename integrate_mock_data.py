#!/usr/bin/env python3
"""
Mock Data Integration Example
============================

This script shows how to integrate mock data with your anomaly detection pipeline.
It demonstrates different ways to feed data into the system.

Usage:
    python integrate_mock_data.py
"""

import asyncio
import httpx
import json
import pandas as pd
from datetime import datetime, timezone
from mock_data_generator import TouristDataGenerator

async def send_single_events(api_url: str = "http://localhost:8000"):
    """Send individual events one by one (real-time simulation)"""
    print("🔄 Method 1: Sending individual events (Real-time simulation)")
    print("-" * 60)
    
    generator = TouristDataGenerator()
    data = generator.generate_emergency_scenario(duration_hours=0.5)  # 15 points
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for i, event in enumerate(data):
            try:
                response = await client.post(f"{api_url}/detect", json=event)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Event {i+1:2d}: {result['anomaly_count']} anomalies detected")
                    
                    if result['anomaly_count'] > 0:
                        for alert in result['alerts']:
                            print(f"  🚨 {alert['anomaly_type']} ({alert['alert_level']}) - "
                                  f"Confidence: {alert['confidence_score']:.2f}")
                else:
                    print(f"Event {i+1:2d}: Error {response.status_code}")
                
                # Simulate real-time delay
                await asyncio.sleep(0.5)
                
            except httpx.ConnectError:
                print(f"❌ Cannot connect to API at {api_url}")
                print("💡 Make sure the API is running: python main.py")
                return
            except Exception as e:
                print(f"❌ Error sending event {i+1}: {e}")

async def send_batch_events(api_url: str = "http://localhost:8000"):
    """Send multiple events in a batch"""
    print("\n📦 Method 2: Sending batch events")
    print("-" * 60)
    
    generator = TouristDataGenerator()
    data = generator.generate_mixed_scenario(count=50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(f"{api_url}/detect_batch", json=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Batch processed: {len(data)} events")
                print(f"🚨 Anomalies detected: {result['anomaly_count']}")
                
                if result['anomaly_count'] > 0:
                    print(f"\nDetected alerts:")
                    for i, alert in enumerate(result['alerts'][:10]):  # Show first 10
                        print(f"  {i+1:2d}. {alert['anomaly_type']} ({alert['alert_level']}) - "
                              f"Tourist: {alert['tourist_id']} - "
                              f"Confidence: {alert['confidence_score']:.2f}")
            else:
                print(f"❌ Batch processing failed: {response.status_code}")
                
        except httpx.ConnectError:
            print(f"❌ Cannot connect to API at {api_url}")
            print("💡 Make sure the API is running: python main.py")
        except Exception as e:
            print(f"❌ Error in batch processing: {e}")

def process_with_pandas():
    """Process data using pandas and the anomaly detector directly"""
    print("\n🐼 Method 3: Direct processing with pandas")
    print("-" * 60)
    
    try:
        from anomaly_detector import detect_anomalies
        
        generator = TouristDataGenerator()
        data = generator.generate_health_anomaly_scenario(duration_hours=1.0)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"📊 Processing {len(df)} data points...")
        
        # Detect anomalies
        anomaly_indices, scores, alerts = detect_anomalies(df, return_alerts=True)
        
        print(f"🚨 Anomalies detected: {len(anomaly_indices)}")
        
        if len(anomaly_indices) > 0:
            print(f"\nTop anomalies:")
            for i, idx in enumerate(anomaly_indices[:5]):
                row = df.iloc[idx]
                alert = alerts[i] if i < len(alerts) else {}
                print(f"  {i+1}. Row {idx}: {alert.get('anomaly_type', 'UNKNOWN')} "
                      f"(score: {scores[idx]:.3f})")
                print(f"     Tourist: {row['tourist_id']}, "
                      f"HR: {row.get('heart_rate', 'N/A')}, "
                      f"Battery: {row.get('battery_level', 'N/A')}%")
        
        return len(anomaly_indices), len(df)
        
    except ImportError as e:
        print(f"❌ Cannot import anomaly_detector: {e}")
        return 0, 0
    except Exception as e:
        print(f"❌ Error in direct processing: {e}")
        return 0, 0

def load_from_file():
    """Load and process data from a file"""
    print("\n📁 Method 4: Loading data from file")
    print("-" * 60)
    
    # Generate and save some data first
    generator = TouristDataGenerator()
    data = generator.generate_mixed_scenario(count=100)
    
    # Save to JSON
    filename = "sample_data.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved sample data to {filename}")
    
    # Load from JSON
    with open(filename, 'r') as f:
        loaded_data = json.load(f)
    
    print(f"📂 Loaded {len(loaded_data)} records from {filename}")
    
    # Process the loaded data
    try:
        from anomaly_detector import detect_anomalies
        df = pd.DataFrame(loaded_data)
        anomaly_indices, scores, alerts = detect_anomalies(df, return_alerts=True)
        
        print(f"🚨 Anomalies detected: {len(anomaly_indices)}")
        
        # Show summary by tourist
        tourist_summary = {}
        for idx in anomaly_indices:
            tourist_id = df.iloc[idx]['tourist_id']
            if tourist_id not in tourist_summary:
                tourist_summary[tourist_id] = 0
            tourist_summary[tourist_id] += 1
        
        if tourist_summary:
            print(f"\nAnomalies by tourist:")
            for tourist_id, count in tourist_summary.items():
                print(f"  {tourist_id}: {count} anomalies")
        
        return len(anomaly_indices), len(df)
        
    except Exception as e:
        print(f"❌ Error processing loaded data: {e}")
        return 0, 0

async def check_alert_system():
    """Check if alerts were processed by the alert system"""
    print("\n🚨 Method 5: Checking alert system")
    print("-" * 60)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Check alert system health
            response = await client.get("http://localhost:8001/health")
            if response.status_code == 200:
                print("✅ Alert system is running")
                
                # Get recent alerts
                response = await client.get("http://localhost:8001/alerts?limit=10")
                if response.status_code == 200:
                    alerts = response.json()
                    print(f"📋 Recent alerts in system: {len(alerts)}")
                    
                    if alerts:
                        print(f"\nRecent alerts:")
                        for i, alert in enumerate(alerts[:5]):
                            print(f"  {i+1}. {alert.get('anomaly_type', 'UNKNOWN')} "
                                  f"({alert.get('alert_level', 'UNKNOWN')}) - "
                                  f"Tourist: {alert.get('tourist_id', 'UNKNOWN')}")
                    else:
                        print("  No alerts found (this is normal if no anomalies were detected)")
                else:
                    print(f"❌ Failed to get alerts: {response.status_code}")
            else:
                print(f"❌ Alert system not responding: {response.status_code}")
                
        except httpx.ConnectError:
            print("❌ Alert system not running")
            print("💡 Make sure both services are running: python main.py")
        except Exception as e:
            print(f"❌ Error checking alert system: {e}")

async def main():
    """Main integration demonstration"""
    print("🔗 Mock Data Integration Examples")
    print("=" * 60)
    print("This script demonstrates different ways to integrate mock data")
    print("with your anomaly detection pipeline.\n")
    
    # Method 1: Real-time individual events
    await send_single_events()
    
    # Method 2: Batch processing
    await send_batch_events()
    
    # Method 3: Direct pandas processing
    anomalies_3, points_3 = process_with_pandas()
    
    # Method 4: File-based processing
    anomalies_4, points_4 = load_from_file()
    
    # Method 5: Check alert system
    await check_alert_system()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"INTEGRATION SUMMARY")
    print(f"{'='*60}")
    print(f"Direct processing: {anomalies_3} anomalies from {points_3} points")
    print(f"File processing: {anomalies_4} anomalies from {points_4} points")
    
    print(f"\n💡 Integration Tips:")
    print(f"  1. Use individual events for real-time monitoring")
    print(f"  2. Use batch processing for historical data analysis")
    print(f"  3. Use direct processing for development and testing")
    print(f"  4. Use file-based processing for large datasets")
    print(f"  5. Check alert system for notification status")
    
    print(f"\n🚀 Next Steps:")
    print(f"  - Generate more data: python mock_data_generator.py --scenario mixed --count 1000")
    print(f"  - Test detection: python test_with_mock_data.py --scenario emergency")
    print(f"  - Start full pipeline: python main.py")
    print(f"  - View dashboard: http://localhost:8001")

if __name__ == "__main__":
    asyncio.run(main())
