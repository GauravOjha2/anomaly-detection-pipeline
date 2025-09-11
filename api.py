# api.py
from fastapi import FastAPI, BackgroundTasks
from schemas import TelemetryEvent, DetectResponse
from anomaly_detector import detect_from_event
from fastapi import Body
import os
import json
from pathlib import Path
import httpx
from typing import Optional

app = FastAPI(title="Anomaly Detection Pipeline", version="1.0.0")

# Alert system integration
ALERT_SYSTEM_URL = os.getenv("ALERT_SYSTEM_URL", "http://localhost:8001")

async def send_to_alert_system(response: DetectResponse):
    """Send detection results to alert system for processing"""
    try:
        # Convert to dict and serialize datetime objects properly
        data = response.dict()
        
        # Convert datetime objects to ISO format strings recursively
        def serialize_datetime(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: serialize_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_datetime(item) for item in obj]
            else:
                return obj
        
        # Apply serialization to the entire data structure
        serialized_data = serialize_datetime(data)
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{ALERT_SYSTEM_URL}/process-alerts",
                json=serialized_data
            )
            print(f"✅ Successfully sent {len(data.get('alerts', []))} alerts to alert system")
    except Exception as e:
        print(f"Failed to send to alert system: {e}")

@app.post("/detect", response_model=DetectResponse)
async def detect(e: TelemetryEvent, bg: BackgroundTasks):
    """Detect anomalies and send to alert system"""
    response = detect_from_event(e.dict())
    
    # Send to alert system in background if anomalies found
    if response.anomaly_count > 0:
        bg.add_task(send_to_alert_system, response)
    
    return response

@app.post("/detect_batch")
async def detect_batch(events: list[TelemetryEvent] = Body(...), bg: BackgroundTasks = None):
    """Detect anomalies in batch and send to alert system"""
    import pandas as pd
    df = pd.DataFrame([e.dict() for e in events])
    from anomaly_detector import detect_from_event_batch
    result = detect_from_event_batch(df)
    
    # Send to alert system in background if anomalies found
    if result.get("anomaly_count", 0) > 0 and bg:
        # Convert to DetectResponse format for alert system
        from schemas import DetectResponse, Alert
        alerts = [Alert(**alert) for alert in result.get("alerts", [])]
        response = DetectResponse(
            status=result["status"],
            anomaly_count=result["anomaly_count"],
            alerts=alerts
        )
        bg.add_task(send_to_alert_system, response)
    
    return result
    
@app.get("/")
def root():
    return {
        "message": "Anomaly Detection Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "detect": "/detect",
            "detect_batch": "/detect_batch", 
            "health": "/health",
            "health_details": "/health/details",
            "docs": "/docs"
        },
        "alert_system_url": ALERT_SYSTEM_URL
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/health/details")
def health_details():
    logs_dir = Path("logs")
    log_file = logs_dir / "detections.log"
    last_anomaly_ts = None
    last_anomaly_count = 0
    if log_file.exists():
        try:
            # Read last 200 lines max to find the most recent anomaly
            with log_file.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-200:]
            for line in reversed(lines):
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict) and int(rec.get("anomaly_count", 0)) > 0:
                        last_anomaly_ts = rec.get("timestamp")
                        last_anomaly_count = int(rec.get("anomaly_count", 0))
                        break
                except Exception:
                    continue
        except Exception:
            pass

    return {
        "status": "ok",
        "demo_rules": os.getenv("DEMO_RULES") == "1",
        "demo_speed_kmh": os.getenv("DEMO_SPEED_KMH"),
        "webhook_configured": bool(os.getenv("WEBHOOK_URL")),
        "logging_enabled": log_file.exists(),
        "last_anomaly_timestamp": last_anomaly_ts,
        "last_anomaly_count": last_anomaly_count,
    }
