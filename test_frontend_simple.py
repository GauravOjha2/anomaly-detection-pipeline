#!/usr/bin/env python3
"""
Simple Frontend Test
===================

Test the enhanced frontend without database dependencies.
"""

import asyncio
import httpx
import json

async def test_frontend():
    """Test the frontend directly"""
    
    print("🧪 Testing Enhanced Frontend")
    print("=" * 40)
    
    # Test alert system health
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://localhost:8001/health")
            print(f"✅ Alert system health: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Frontend is accessible!")
                print("\n🌐 Open your browser and go to: http://localhost:8001")
                print("\n🎨 You should see:")
                print("  • Beautiful glassmorphism design")
                print("  • Gradient background")
                print("  • Animated status indicators")
                print("  • Modern card layouts")
                print("  • Responsive design")
                print("  • Interactive elements")
                
                print("\n📱 Test on different screen sizes:")
                print("  • Desktop: Full grid layout")
                print("  • Tablet: Responsive grid")
                print("  • Mobile: Single column layout")
                
                print("\n⌨️  Try keyboard shortcuts:")
                print("  • Ctrl+R: Refresh data")
                print("  • Ctrl+F: Focus search")
                print("  • Escape: Close modals")
                
                print("\n🎯 Interactive features:")
                print("  • Hover over cards for animations")
                print("  • Click refresh button")
                print("  • Check mobile responsiveness")
                print("  • Try dark mode (if supported by browser)")
                
            else:
                print(f"❌ Alert system returned status: {response.status_code}")
                
    except httpx.ConnectError:
        print("❌ Cannot connect to alert system on port 8001")
        print("💡 Make sure the system is running: python main.py")
    except Exception as e:
        print(f"❌ Error testing frontend: {e}")

if __name__ == "__main__":
    asyncio.run(test_frontend())
