# schemas.py
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import uuid

class TelemetryEvent(BaseModel):
    tourist_id: str
    lat: float
    lng: float
    timestamp: datetime
    heart_rate: Optional[float] = None
    battery_level: Optional[float] = None
    network_status: Optional[str] = None
    panic_button: Optional[bool] = None
    accuracy: Optional[float] = None

class Alert(BaseModel):
    model_config = {"protected_namespaces": ()}
    alert_id: str = str(uuid.uuid4())
    tourist_id: str
    anomaly_type: str
    alert_level: str
    confidence_score: float
    location: Dict[str, float]
    timestamp: datetime
    raw_evidence: Dict
    model_version: str

class DetectResponse(BaseModel):
    status: str
    anomaly_count: int
    alerts: List[Alert]
