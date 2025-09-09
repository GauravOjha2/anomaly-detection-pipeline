#!/usr/bin/env python3
"""
Unified Anomaly Detection Pipeline with Alert System
====================================================

This is the main entry point that runs both:
1. Anomaly Detection API (port 8000)
2. Alert System API (port 8001)

Usage:
    python main.py

Environment Variables:
    - ALERT_SYSTEM_URL: URL of alert system (default: http://localhost:8001)
    - DATABASE_URL: PostgreSQL connection string for alert system
    - ADMIN_EMAILS: Comma-separated admin emails for alerts
    - WEBHOOK_URL: Optional webhook URL for external notifications
"""

import asyncio
import uvicorn
import multiprocessing
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add alert_system to Python path
sys.path.insert(0, str(Path(__file__).parent / "alert_system"))

def run_anomaly_detection_api():
    """Run the anomaly detection API on port 8000"""
    from api import app
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )

def run_alert_system_api():
    """Run the alert system API on port 8001"""
    try:
        from alert_system.main import app
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"❌ Alert system failed to start: {e}")
        print("💡 Continuing with anomaly detection only...")
        # Keep the process alive but do nothing
        import time
        while True:
            time.sleep(1)

def main():
    """Start both services"""
    print("🚀 Starting Anomaly Detection Pipeline with Alert System")
    print("=" * 60)
    print("📊 Anomaly Detection API: http://localhost:8000")
    print("🚨 Alert System API: http://localhost:8001")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🎛️  Alert Dashboard: http://localhost:8001")
    print("=" * 60)
    
    # Check if alert system dependencies are available
    try:
        import prisma
        print("✅ Alert system dependencies found")
    except ImportError:
        print("⚠️  Alert system dependencies not found. Install with:")
        print("   cd alert_system && pip install -r requirements.txt")
        print("   npx prisma generate")
        print("   Continuing with anomaly detection only...")
    
    # Start both services in separate processes
    processes = []
    
    # Start anomaly detection API
    anomaly_process = multiprocessing.Process(target=run_anomaly_detection_api)
    anomaly_process.start()
    processes.append(anomaly_process)
    
    # Start alert system API
    alert_process = multiprocessing.Process(target=run_alert_system_api)
    alert_process.start()
    processes.append(alert_process)
    
    try:
        # Wait for all processes
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()
        print("✅ Shutdown complete")

if __name__ == "__main__":
    main()
