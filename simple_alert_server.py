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
from datetime import datetime, timezone
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
alerts_storage: List[Dict[str, Any]] = []

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
        "HEART_RATE_LOW", "BATTERY_CRITICAL", "PANIC", "PROLONGED_INACTIVITY"
    ]
    alert_levels = ["CRITICAL", "WARNING", "INFO"]
    tourist_ids = ["tourist_001", "tourist_002", "tourist_003", "tourist_004", "tourist_005"]
    
    # Beijing coordinates
    base_lat = 39.9042
    base_lng = 116.4074
    
    for i in range(10):
        alert_type = random.choice(alert_types)
        alert_level = random.choice(alert_levels)
        tourist_id = random.choice(tourist_ids)
        
        # Generate random coordinates around Beijing
        lat = base_lat + random.uniform(-0.01, 0.01)
        lng = base_lng + random.uniform(-0.01, 0.01)
        
        alert = {
            "alert_id": f"alert_{i+1:03d}_{int(datetime.now().timestamp())}",
            "tourist_id": tourist_id,
            "anomaly_type": alert_type,
            "alert_level": alert_level,
            "confidence_score": round(random.uniform(0.6, 0.95), 3),
            "location": {"lat": round(lat, 6), "lng": round(lng, 6)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_evidence": {
                "heart_rate": random.randint(60, 200),
                "battery_level": random.randint(5, 100),
                "network_status": random.choice(["good", "poor", "no_signal"]),
                "panic_button": random.choice([True, False]),
                "accuracy": round(random.uniform(3, 25), 1)
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
