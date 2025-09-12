#!/usr/bin/env python3
"""
Simple Alert Server - No Database Required
==========================================

This is a simplified version of the alert system that works without database dependencies
so you can see the enhanced frontend immediately.
"""

from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import uuid

# Create FastAPI app
app = FastAPI(title="Simple Alert System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for alerts (no database needed)
alerts_storage: List[Dict[str, Any]] = [
    {
        "alert_id": "alert_001_1736681086",
        "tourist_id": "tourist_001",
        "anomaly_type": "VELOCITY_HIGH",
        "alert_level": "WARNING",
        "confidence_score": 0.85,
        "location": {"lat": 39.9042, "lng": 116.4074},
        "timestamp": "2025-01-12T10:30:00Z",
        "raw_evidence": {
            "heart_rate": 120,
            "battery_level": 45,
            "network_status": "good",
            "panic_button": False,
            "accuracy": 5.2
        },
        "model_version": "v2.0"
    },
    {
        "alert_id": "alert_002_1736681087",
        "tourist_id": "tourist_002",
        "anomaly_type": "HEART_RATE_CRITICAL",
        "alert_level": "CRITICAL",
        "confidence_score": 0.92,
        "location": {"lat": 39.9052, "lng": 116.4084},
        "timestamp": "2025-01-12T11:15:00Z",
        "raw_evidence": {
            "heart_rate": 180,
            "battery_level": 78,
            "network_status": "good",
            "panic_button": True,
            "accuracy": 4.8
        },
        "model_version": "v2.0"
    },
    {
        "alert_id": "alert_003_1736681088",
        "tourist_id": "tourist_003",
        "anomaly_type": "BATTERY_CRITICAL",
        "alert_level": "WARNING",
        "confidence_score": 0.78,
        "location": {"lat": 39.9032, "lng": 116.4064},
        "timestamp": "2025-01-12T12:00:00Z",
        "raw_evidence": {
            "heart_rate": 95,
            "battery_level": 8,
            "network_status": "poor",
            "panic_button": False,
            "accuracy": 12.3
        },
        "model_version": "v2.0"
    },
    {
        "alert_id": "alert_004_1736681089",
        "tourist_id": "tourist_004",
        "anomaly_type": "GPS_ANOMALY",
        "alert_level": "INFO",
        "confidence_score": 0.65,
        "location": {"lat": 39.9042, "lng": 116.4074},
        "timestamp": "2025-01-12T12:45:00Z",
        "raw_evidence": {
            "heart_rate": 88,
            "battery_level": 62,
            "network_status": "excellent",
            "panic_button": False,
            "accuracy": 25.0
        },
        "model_version": "v2.0"
    },
    {
        "alert_id": "alert_005_1736681090",
        "tourist_id": "tourist_005",
        "anomaly_type": "FALL_DETECTED",
        "alert_level": "CRITICAL",
        "confidence_score": 0.89,
        "location": {"lat": 39.9055, "lng": 116.4080},
        "timestamp": "2025-01-12T13:20:00Z",
        "raw_evidence": {
            "heart_rate": 165,
            "battery_level": 34,
            "network_status": "good",
            "panic_button": True,
            "accuracy": 6.7
        },
        "model_version": "v2.0"
    }
]

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "alert_system", "alert_system", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Also serve CSS and JS files directly
@app.get("/dashboard.css")
def get_dashboard_css():
    return FileResponse(os.path.join(static_dir, "dashboard.css"))

@app.get("/dashboard.js")
def get_dashboard_js():
    return FileResponse(os.path.join(static_dir, "dashboard.js"))

@app.get("/")
def root():
    """Serve the enhanced dashboard"""
    response = FileResponse(os.path.join(static_dir, "index.html"))
    # Add security headers
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

@app.get("/alerts-page")
def alerts_page():
    """Serve the alerts page"""
    response = FileResponse(os.path.join(static_dir, "alerts.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

@app.get("/analytics")
def analytics_page():
    """Serve the analytics page"""
    response = FileResponse(os.path.join(static_dir, "analytics.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

@app.get("/settings")
def settings_page():
    """Serve the settings page"""
    response = FileResponse(os.path.join(static_dir, "settings.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "time": datetime.now(timezone.utc).isoformat() + "Z",
        "alerts_count": len(alerts_storage)
    }

@app.post("/process-alerts")
async def process_alerts(request: Dict[str, Any], bg: BackgroundTasks = None):
    """Process incoming alerts and store them in memory"""
    try:
        alerts = request.get("alerts", [])
        
        # Add unique IDs and timestamps if missing
        for alert in alerts:
            if "alert_id" not in alert:
                alert["alert_id"] = str(uuid.uuid4())
            if "timestamp" not in alert:
                alert["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Store alerts in memory
        alerts_storage.extend(alerts)
        
        # Keep only last 100 alerts to prevent memory issues
        if len(alerts_storage) > 100:
            alerts_storage[:] = alerts_storage[-100:]
        
        return {
            "status": "success",
            "anomaly_count": len(alerts),
            "message": f"Processed {len(alerts)} alerts"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/alerts")
async def get_alerts(limit: int = 100, offset: int = 0):
    """Get stored alerts"""
    try:
        # Return alerts in reverse chronological order
        sorted_alerts = sorted(alerts_storage, key=lambda x: x.get("timestamp", ""), reverse=True)
        return sorted_alerts[offset:offset + limit]
    except Exception as e:
        return {"error": str(e)}

@app.get("/alerts/{alert_id}")
async def get_alert(alert_id: str):
    """Get specific alert by ID"""
    try:
        for alert in alerts_storage:
            if alert.get("alert_id") == alert_id:
                return alert
        return {"error": "Alert not found"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/add-sample-alerts")
async def add_sample_alerts():
    """Add sample alerts for testing the frontend"""
    import random
    
    sample_alerts = []
    alert_types = [
        "VELOCITY_HIGH", "VELOCITY_CRITICAL", "HEART_RATE_HIGH", 
        "HEART_RATE_LOW", "BATTERY_CRITICAL", "PANIC", "PROLONGED_INACTIVITY",
        "GPS_ANOMALY", "NETWORK_LOSS", "DEVICE_TAMPERING", "FALL_DETECTED",
        "UNUSUAL_MOVEMENT", "TEMPERATURE_EXTREME", "PRESSURE_ANOMALY"
    ]
    alert_levels = ["CRITICAL", "WARNING", "INFO"]
    tourist_ids = ["tourist_001", "tourist_002", "tourist_003", "tourist_004", "tourist_005", 
                   "tourist_006", "tourist_007", "tourist_008", "tourist_009", "tourist_010"]
    
    # Beijing coordinates
    base_lat = 39.9042
    base_lng = 116.4074
    
    # Generate 15 diverse alerts
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
                "accuracy": round(random.uniform(3, 25), 1),
                "temperature": round(random.uniform(20, 40), 1),
                "pressure": round(random.uniform(950, 1050), 1)
            },
            "model_version": "v2.0"
        }
        
        sample_alerts.append(alert)
    
    # Add to storage
    alerts_storage.extend(sample_alerts)
    
    return {
        "status": "success",
        "message": f"Added {len(sample_alerts)} sample alerts",
        "total_alerts": len(alerts_storage)
    }

@app.get("/api")
def api_info():
    """API information"""
    return {
        "message": "Simple Alert System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "process_alerts": "/process-alerts",
            "get_alerts": "/alerts",
            "get_alert": "/alerts/{alert_id}",
            "add_samples": "/add-sample-alerts",
            "docs": "/docs"
        },
        "features": [
            "Enhanced frontend dashboard",
            "Real-time alert monitoring",
            "No database required",
            "Sample data generation"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simple Alert System")
    print("=" * 40)
    print("🌐 Dashboard: http://localhost:8001")
    print("📖 API Docs: http://localhost:8001/docs")
    print("🧪 Add samples: POST http://localhost:8001/add-sample-alerts")
    print("=" * 40)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )
