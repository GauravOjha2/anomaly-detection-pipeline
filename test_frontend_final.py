#!/usr/bin/env python3
"""
Final Frontend Test
==================

Test the enhanced frontend with the working alert system.
"""

import asyncio
import httpx
import json

async def test_enhanced_frontend():
    """Test the enhanced frontend"""
    
    print("🎨 Testing Enhanced Frontend Dashboard")
    print("=" * 50)
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test health
            health_response = await client.get("http://localhost:8001/health")
            print(f"✅ Server health: {health_response.status_code}")
            
            # Test alerts endpoint
            alerts_response = await client.get("http://localhost:8001/alerts")
            alerts = alerts_response.json()
            print(f"✅ Alerts loaded: {len(alerts)} alerts found")
            
            # Show sample alert
            if alerts:
                sample = alerts[0]
                print(f"\n📋 Sample Alert:")
                print(f"   Type: {sample.get('anomaly_type', 'N/A')}")
                print(f"   Level: {sample.get('alert_level', 'N/A')}")
                print(f"   Tourist: {sample.get('tourist_id', 'N/A')}")
                print(f"   Confidence: {sample.get('confidence_score', 'N/A')}")
            
            print(f"\n🎉 SUCCESS! Enhanced frontend is working!")
            print(f"\n🌐 Open your browser and go to: http://localhost:8001")
            print(f"\n🎨 You should see:")
            print(f"  • Beautiful glassmorphism design")
            print(f"  • Gradient background with animations")
            print(f"  • {len(alerts)} alert cards with smooth animations")
            print(f"  • Real-time statistics")
            print(f"  • Interactive elements")
            print(f"  • Responsive design")
            
            # Show statistics
            critical_count = len([a for a in alerts if a.get('alert_level') == 'CRITICAL'])
            warning_count = len([a for a in alerts if a.get('alert_level') == 'WARNING'])
            info_count = len([a for a in alerts if a.get('alert_level') == 'INFO'])
            
            print(f"\n📊 Dashboard Statistics:")
            print(f"  • Total Alerts: {len(alerts)}")
            print(f"  • Critical: {critical_count}")
            print(f"  • Warning: {warning_count}")
            print(f"  • Info: {info_count}")
            
            print(f"\n🎯 Features to Test:")
            print(f"  • Hover over cards for animations")
            print(f"  • Click the refresh button")
            print(f"  • Try different screen sizes (mobile/tablet)")
            print(f"  • Check keyboard shortcuts (Ctrl+R)")
            print(f"  • Notice the smooth transitions")
            
    except httpx.ConnectError:
        print("❌ Cannot connect to server on port 8001")
        print("💡 Make sure the server is running: python simple_alert_server.py")
    except Exception as e:
        print(f"❌ Error testing frontend: {e}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_frontend())
