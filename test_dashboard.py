#!/usr/bin/env python3
"""
Test Dashboard with Sample Alerts
=================================

This script generates sample alerts to test the enhanced dashboard.
"""

import asyncio
import httpx
import json
import random
from datetime import datetime, timezone
import time

async def generate_sample_alerts():
    """Generate sample alerts for testing the dashboard"""
    
    # Sample alert data
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
    
    for i in range(10):
        alert_type = random.choice(alert_types)
        alert_level = random.choice(alert_levels)
        tourist_id = random.choice(tourist_ids)
        
        # Generate random coordinates around Beijing
        lat = base_lat + random.uniform(-0.01, 0.01)
        lng = base_lng + random.uniform(-0.01, 0.01)
        
        alert = {
            "alert_id": f"alert_{i+1:03d}_{int(time.time())}",
            "tourist_id": tourist_id,
            "anomaly_type": alert_type,
            "alert_level": alert_level,
            "confidence_score": random.uniform(0.6, 0.95),
            "location": {"lat": lat, "lng": lng},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_evidence": {
                "heart_rate": random.randint(60, 200),
                "battery_level": random.randint(5, 100),
                "network_status": random.choice(["good", "poor", "no_signal"]),
                "panic_button": random.choice([True, False]),
                "accuracy": random.uniform(3, 25)
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
            
            # Send each alert
            for i, alert in enumerate(alerts):
                response = await client.post(
                    f"{alert_system_url}/process-alerts",
                    json={
                        "status": "success",
                        "anomaly_count": 1,
                        "alerts": [alert]
                    }
                )
                
                if response.status_code == 200:
                    print(f"✅ Sent alert {i+1}: {alert['anomaly_type']} ({alert['alert_level']})")
                else:
                    print(f"❌ Failed to send alert {i+1}: {response.status_code}")
                
                # Small delay between alerts
                await asyncio.sleep(0.5)
                
        except httpx.ConnectError:
            print("❌ Cannot connect to alert system. Make sure it's running on port 8001")
            return False
        except Exception as e:
            print(f"❌ Error sending alerts: {e}")
            return False
    
    return True

async def test_dashboard():
    """Test the enhanced dashboard"""
    
    print("🧪 Testing Enhanced Dashboard")
    print("=" * 40)
    
    # Generate sample alerts
    print("📊 Generating sample alerts...")
    alerts = await generate_sample_alerts()
    print(f"✅ Generated {len(alerts)} sample alerts")
    
    # Send alerts to system
    print("\n📤 Sending alerts to alert system...")
    success = await send_alerts_to_system(alerts)
    
    if success:
        print("\n✅ Dashboard test completed successfully!")
        print("\n🌐 View the enhanced dashboard at: http://localhost:8001")
        print("\n🎨 Features to test:")
        print("  • Modern glassmorphism design")
        print("  • Smooth animations and transitions")
        print("  • Real-time status updates")
        print("  • Interactive alert cards")
        print("  • Responsive design")
        print("  • Keyboard shortcuts (Ctrl+R to refresh)")
        print("  • Dark mode support")
        print("  • Mobile-friendly layout")
    else:
        print("\n❌ Dashboard test failed. Please check the alert system.")

if __name__ == "__main__":
    asyncio.run(test_dashboard())
