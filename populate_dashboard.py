#!/usr/bin/env python3
"""
Populate Dashboard with Mock Data
=================================

This script creates mock alerts directly in the database for testing the enhanced frontend.
"""

import asyncio
import httpx
import json
import random
from datetime import datetime, timezone, timedelta

async def create_mock_alerts():
    """Create mock alerts for dashboard testing"""
    
    # Sample data
    alert_types = [
        "VELOCITY_HIGH", "VELOCITY_CRITICAL", "HEART_RATE_HIGH", 
        "HEART_RATE_LOW", "BATTERY_CRITICAL", "PANIC", "PROLONGED_INACTIVITY"
    ]
    
    alert_levels = ["CRITICAL", "WARNING", "INFO"]
    tourist_ids = ["tourist_001", "tourist_002", "tourist_003", "tourist_004", "tourist_005"]
    
    # Beijing coordinates
    base_lat = 39.9042
    base_lng = 116.4074
    
    alerts = []
    
    # Create 15 mock alerts with different timestamps
    for i in range(15):
        alert_type = random.choice(alert_types)
        alert_level = random.choice(alert_levels)
        tourist_id = random.choice(tourist_ids)
        
        # Generate random coordinates around Beijing
        lat = base_lat + random.uniform(-0.01, 0.01)
        lng = base_lng + random.uniform(-0.01, 0.01)
        
        # Create timestamp from last 24 hours
        hours_ago = random.randint(0, 24)
        timestamp = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        
        alert = {
            "alert_id": f"alert_{i+1:03d}_{int(timestamp.timestamp())}",
            "tourist_id": tourist_id,
            "anomaly_type": alert_type,
            "alert_level": alert_level,
            "confidence_score": round(random.uniform(0.6, 0.95), 3),
            "location": {"lat": round(lat, 6), "lng": round(lng, 6)},
            "timestamp": timestamp.isoformat(),
            "raw_evidence": {
                "heart_rate": random.randint(60, 200),
                "battery_level": random.randint(5, 100),
                "network_status": random.choice(["good", "poor", "no_signal"]),
                "panic_button": random.choice([True, False]),
                "accuracy": round(random.uniform(3, 25), 1)
            },
            "model_version": "v2.0"
        }
        
        alerts.append(alert)
    
    return alerts

async def send_alerts_to_system(alerts):
    """Send alerts to the alert system"""
    
    alert_system_url = "http://localhost:8001"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Test health first
            health_response = await client.get(f"{alert_system_url}/health")
            print(f"✅ Alert system health: {health_response.status_code}")
            
            # Send alerts in batches
            batch_size = 5
            for i in range(0, len(alerts), batch_size):
                batch = alerts[i:i+batch_size]
                
                response = await client.post(
                    f"{alert_system_url}/process-alerts",
                    json={
                        "status": "success",
                        "anomaly_count": len(batch),
                        "alerts": batch
                    }
                )
                
                if response.status_code == 200:
                    print(f"✅ Sent batch {i//batch_size + 1}: {len(batch)} alerts")
                else:
                    print(f"❌ Failed to send batch {i//batch_size + 1}: {response.status_code}")
                    print(f"   Response: {response.text}")
                
                # Small delay between batches
                await asyncio.sleep(1)
                
        except httpx.ConnectError:
            print("❌ Cannot connect to alert system. Make sure it's running on port 8001")
            return False
        except Exception as e:
            print(f"❌ Error sending alerts: {e}")
            return False
    
    return True

async def main():
    """Main function"""
    
    print("🎨 Populating Enhanced Dashboard with Mock Data")
    print("=" * 50)
    
    # Create mock alerts
    print("📊 Creating mock alerts...")
    alerts = await create_mock_alerts()
    print(f"✅ Created {len(alerts)} mock alerts")
    
    # Show sample alert
    print(f"\n📋 Sample alert:")
    sample = alerts[0]
    print(f"   Type: {sample['anomaly_type']}")
    print(f"   Level: {sample['alert_level']}")
    print(f"   Tourist: {sample['tourist_id']}")
    print(f"   Confidence: {sample['confidence_score']}")
    print(f"   Time: {sample['timestamp']}")
    
    # Send to system
    print(f"\n📤 Sending alerts to system...")
    success = await send_alerts_to_system(alerts)
    
    if success:
        print(f"\n🎉 Dashboard populated successfully!")
        print(f"\n🌐 View the enhanced dashboard at: http://localhost:8001")
        print(f"\n🎨 You should now see:")
        print(f"  • {len(alerts)} alerts in the dashboard")
        print(f"  • Real-time statistics")
        print(f"  • Beautiful alert cards with animations")
        print(f"  • Interactive features")
        print(f"  • Responsive design")
        
        # Show statistics
        critical_count = len([a for a in alerts if a['alert_level'] == 'CRITICAL'])
        warning_count = len([a for a in alerts if a['alert_level'] == 'WARNING'])
        info_count = len([a for a in alerts if a['alert_level'] == 'INFO'])
        
        print(f"\n📊 Alert Statistics:")
        print(f"  • Total: {len(alerts)}")
        print(f"  • Critical: {critical_count}")
        print(f"  • Warning: {warning_count}")
        print(f"  • Info: {info_count}")
        
    else:
        print(f"\n❌ Failed to populate dashboard")

if __name__ == "__main__":
    asyncio.run(main())
