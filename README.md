# Enhanced Anomaly Detection System - Production Ready

## 🚀 Quick Start

Your enhanced anomaly detection system is ready for production deployment with FastAPI, IoT sensor support, and real-time alerting!

### API Server

Start the FastAPI server:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn api:app --reload
```

Server runs at `http://127.0.0.1:8000`

### API Endpoints

#### Single Event Detection
```bash
POST /detect
Content-Type: application/json

{
  "tourist_id": "u001",
  "lat": 39.9847,
  "lng": 116.3184,
  "timestamp": "2025-01-01T10:00:00Z",
  "heart_rate": 92,
  "battery_level": 55,
  "network_status": "good",
  "panic_button": false,
  "accuracy": 5.0
}
```

#### Batch Detection (Recommended)
```bash
POST /detect_batch
Content-Type: application/json

[
  {
    "tourist_id": "u001",
    "lat": 39.9847,
    "lng": 116.3184,
    "timestamp": "2025-01-01T10:00:00Z",
    "heart_rate": 92
  },
  {
    "tourist_id": "u001", 
    "lat": 39.9848,
    "lng": 116.3185,
    "timestamp": "2025-01-01T10:01:00Z",
    "heart_rate": 95
  }
]
```

#### Health Check
```bash
GET /health
GET /health/details  # Shows configuration and last anomaly info
```

### Response Format

```json
{
  "status": "success",
  "anomaly_count": 1,
  "alerts": [
    {
      "alert_id": "uuid",
      "tourist_id": "u001",
      "anomaly_type": "VELOCITY_HIGH",
      "alert_level": "WARNING",
      "confidence_score": 0.7,
      "location": {"lat": 39.9847, "lng": 116.3184},
      "timestamp": "2025-01-01T10:00:00Z",
      "raw_evidence": {...},
      "model_version": "v1.0.0"
    }
  ]
}
```

### IoT Sensor Integration

The system supports real-time IoT sensor data:

- **GPS**: `lat`, `lng`, `timestamp`, `accuracy`
- **Health**: `heart_rate` (BPM)
- **Device**: `battery_level`, `network_status`
- **Emergency**: `panic_button`

### Environment Configuration

```bash
# Demo mode (for testing)
export DEMO_RULES=1
export DEMO_SPEED_KMH=60

# Webhook notifications (optional)
export WEBHOOK_URL=https://webhook.site/your-id

# Start server
uvicorn api:app --reload
```

### Logging & Monitoring

- **Detection Logs**: `logs/detections.log` (JSON format)
- **Health Status**: `GET /health/details`
- **Webhook Alerts**: POST to configured URL on anomalies

### Agent Integration

Perfect for IoT agent integration:

```python
import requests

# Your agent receives IoT sensor data
sensor_data = {
    "tourist_id": "u001",
    "lat": 39.9847,
    "lng": 116.3184, 
    "timestamp": "2025-01-01T10:00:00Z",
    "heart_rate": 92,
    "panic_button": False
}

# Send to anomaly detection
response = requests.post(
    "http://localhost:8000/detect",
    json=sensor_data
)

result = response.json()
if result["anomaly_count"] > 0:
    # Forward alerts to authorities
    for alert in result["alerts"]:
        notify_admin(alert)
```

### Enhanced Features

- **20-Feature Model**: GPS trajectory, velocity, acceleration, movement patterns
- **Ensemble Detection**: Isolation Forest + Elliptic Envelope + One-Class SVM
- **Real-time Processing**: Vectorized operations for high performance
- **IoT Ready**: Heart rate, battery, network status, panic button support
- **Production Logging**: Structured JSON logs for analysis
- **Webhook Alerts**: Real-time notifications to admin systems

### Required Data Format

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tourist_id` | string | ✅ | Unique tourist identifier |
| `lat` | float | ✅ | GPS latitude |
| `lng` | float | ✅ | GPS longitude |
| `timestamp` | string | ✅ | ISO 8601 timestamp |
| `heart_rate` | float | ❌ | Heart rate in BPM |
| `battery_level` | float | ❌ | Battery percentage (0-100) |
| `network_status` | string | ❌ | Network connection status |
| `panic_button` | boolean | ❌ | Emergency panic button pressed |
| `accuracy` | float | ❌ | GPS accuracy in meters |

### Alert Types

- `PANIC` - Critical (panic button pressed)
- `HEART_RATE_HIGH` - Critical (heart rate > 150 BPM)
- `HEART_RATE_LOW` - Warning (heart rate < 40 BPM)
- `BATTERY_CRITICAL` - Warning (battery < 10%)
- `VELOCITY_CRITICAL` - Critical (speed > 120 km/h)
- `VELOCITY_HIGH` - Warning (speed > 60 km/h)
- `PROLONGED_INACTIVITY` - Warning (low movement)

### Files

- `api.py` - FastAPI server with endpoints
- `anomaly_detector.py` - Core detection logic
- `schemas.py` - Pydantic data models
- `test_request.py` - Example client
- `trained_tourist_safety_model.pkl` - Trained model (not in git)
- `logs/detections.log` - Detection logs (not in git)

### Deployment

1. **Development**:
   ```bash
   uvicorn api:app --reload
   ```

2. **Production**:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

3. **Docker** (optional):
   ```dockerfile
   FROM python:3.11-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

### Testing

```bash
# Test single detection
python test_request.py

# Test with demo rules
export DEMO_RULES=1
export DEMO_SPEED_KMH=0.1
uvicorn api:app --reload
python test_request.py
```

### API Documentation

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

---

**Ready for production deployment with your IoT agent!** 🎯