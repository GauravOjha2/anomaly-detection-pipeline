# api.py
from fastapi import FastAPI
from schemas import TelemetryEvent, DetectResponse
from anomaly_detector import detect_from_event
from fastapi import Body
import os
import json
from pathlib import Path

app = FastAPI()

@app.post("/detect", response_model=DetectResponse)
def detect(e: TelemetryEvent):
    return detect_from_event(e.dict())

@app.post("/detect_batch")
def detect_batch(events: list[TelemetryEvent] = Body(...)):
    import pandas as pd
    df = pd.DataFrame([e.dict() for e in events])
    from anomaly_detector import detect_from_event_batch
    return detect_from_event_batch(df)
    
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
