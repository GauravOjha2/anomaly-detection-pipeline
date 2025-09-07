"""
Enhanced Anomaly Detection System - Deployment Ready
===================================================

Clean, simple anomaly detector for real-world deployment.
Works with the new enhanced ensemble model (20 features).

Usage:
    from anomaly_detector import detect_anomalies
    
    # Load your data
    data = pd.read_csv('your_data.csv')
    
    # Detect anomalies
    anomalies, scores = detect_anomalies(data)
    
    # View results
    print(f"Found {len(anomalies)} anomalies")
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timezone
import httpx
import joblib
import os
from geopy.distance import geodesic
from schemas import TelemetryEvent, DetectResponse, Alert


def detect_anomalies(data, model_path='trained_tourist_safety_model.pkl', return_alerts: bool = False):
    """
    Detect anomalies using the enhanced ensemble model
    
    Args:
        data: pandas DataFrame with columns:
              - lat, lng (GPS coordinates)
              - timestamp (optional)
              - heart_rate (optional)
              - battery_level (optional)
              - network_status (optional)
              - panic_button (optional)
              - accuracy (optional)
        model_path: Path to trained model file
    
    Returns:
        tuple: (anomaly_indices, anomaly_scores)
               - anomaly_indices: List of row indices that are anomalies
               - anomaly_scores: Anomaly scores for all points (lower = more anomalous)
    """
    
    # Load the enhanced trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_data = joblib.load(model_path)
    print(f"✅ Loaded enhanced model (version: {model_data.get('version', '1.0')})")
    print(f"📊 Analyzing {len(data)} data points...")
    
    # Extract enhanced features (20 features)
    features = extract_enhanced_features(data)
    
    # Use the ensemble model for prediction
    if model_data.get('model_type') == 'enhanced_ensemble':
        predictions, scores = predict_with_ensemble(model_data, features)
    else:
        # Fallback for older models
        isolation_forest = model_data['isolation_forest']
        scaler = model_data['scaler']
        features_scaled = scaler.transform(features)
        predictions = isolation_forest.predict(features_scaled)
        scores = isolation_forest.decision_function(features_scaled)
    
    # Get anomaly indices (where prediction == -1)
    anomaly_indices = np.where(predictions == -1)[0]
    DEMO_RULES = os.getenv("DEMO_RULES") == "1"
    if DEMO_RULES:
        # features is already computed above
        velocities_kmh = features[:, 5] if features.shape[1] > 5 else np.zeros(len(data))
        threshold_kmh = float(os.getenv("DEMO_SPEED_KMH", "60"))
        speed_idx = np.where(velocities_kmh > threshold_kmh)[0]
        # Also treat any explicit panic as anomaly for demo purposes
        if 'panic_button' in data.columns:
            panic_series = pd.Series(data['panic_button']).fillna(False).astype(bool)
            panic_idx = np.where(panic_series.values)[0]
        else:
            panic_idx = np.array([], dtype=int)
        anomaly_indices = np.unique(np.concatenate([anomaly_indices, speed_idx, panic_idx]))
        print(
            "DEMO_RULES:", True,
            "threshold_kmh:", float(threshold_kmh),
            "max_v:", float(velocities_kmh.max()),
            "speed_count:", int(len(speed_idx)),
            "panic_count:", int(len(panic_idx)),
            "final_count:", int(len(anomaly_indices))
        )
    
    
    print(f"🚨 Found {len(anomaly_indices)} anomalies out of {len(data)} points")
    
    if not return_alerts:
        return anomaly_indices, scores

    # Minimal rule-based alerts derived from inputs and features
    alerts: list[dict] = []

    # Derive velocities for simple thresholds
    features_local = features if 'features' in locals() else extract_enhanced_features(data)
    velocities_kmh = features_local[:, 5] if features_local.shape[1] > 5 else np.zeros(len(data))
    rule_idx = np.where(velocities_kmh > 60)[0]
    anomaly_indices = np.unique(np.concatenate([anomaly_indices, rule_idx]))

    for idx in anomaly_indices:
        row = data.iloc[idx]
        # Choose one primary reason if possible
        if 'panic_button' in row and bool(row['panic_button']):
            anomaly_type = 'PANIC'
            confidence = 0.99
        elif 'heart_rate' in row and pd.notna(row['heart_rate']) and row['heart_rate'] > 150:
            anomaly_type = 'HEART_RATE_HIGH'
            confidence = 0.9
        elif 'heart_rate' in row and pd.notna(row['heart_rate']) and row['heart_rate'] < 40:
            anomaly_type = 'HEART_RATE_LOW'
            confidence = 0.8
        elif 'battery_level' in row and pd.notna(row['battery_level']) and row['battery_level'] < 10:
            anomaly_type = 'BATTERY_CRITICAL'
            confidence = 0.7
        elif velocities_kmh[idx] > 120:
            anomaly_type = 'VELOCITY_CRITICAL'
            confidence = 0.85
        elif velocities_kmh[idx] > 60:
            anomaly_type = 'VELOCITY_HIGH'
            confidence = 0.7
        else:
            anomaly_type = 'PROLONGED_INACTIVITY'
            confidence = 0.6

        alerts.append({
            'anomaly_type': anomaly_type,
            'confidence_score': float(confidence)
        })

    return anomaly_indices, scores, alerts

MODEL_VERSION = "v1.0.0"

SEVERITY_MAP = {
    "VELOCITY_HIGH": "WARNING",
    "VELOCITY_CRITICAL": "CRITICAL",
    "PROLONGED_INACTIVITY": "WARNING",
    "PANIC": "CRITICAL",
    "HEART_RATE_HIGH": "CRITICAL",
    "HEART_RATE_LOW": "WARNING",
    "BATTERY_CRITICAL": "WARNING"
}
def detect_from_event(event: dict) -> DetectResponse:
    """
    Handle multiple telemetry events (trajectory) in one request.
    """
    te = TelemetryEvent(**event)
    df = pd.DataFrame([te.dict()])
    _, _, alerts = detect_anomalies(df, return_alerts=True)

    norm_alerts = []
    for a in alerts:
        norm_alerts.append(Alert(
            tourist_id=te.tourist_id,
            anomaly_type=a["anomaly_type"],
            alert_level=SEVERITY_MAP.get(a["anomaly_type"], "INFO"),
            confidence_score=a.get("confidence_score", 0.0),
            location={"lat": te.lat, "lng": te.lng},
            timestamp=te.timestamp,
            raw_evidence=te.dict(),
            model_version=MODEL_VERSION
        ))
    response = DetectResponse(status="success", anomaly_count=len(norm_alerts), alerts=norm_alerts)
    _log_detection({
        "type": "single",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": te.dict(),
        "anomaly_count": response.anomaly_count,
        "alerts": [a.dict() for a in response.alerts],
        "iot_features": _compute_iot_features_df(pd.DataFrame([te.dict()]))
    })
    if response.anomaly_count > 0:
        _send_webhook([a.dict() for a in response.alerts])
    return response

def detect_from_event_batch(df: pd.DataFrame):
    anomaly_indices, _, alerts = detect_anomalies(df, return_alerts=True)

    results = []
    for idx, a in zip(anomaly_indices, alerts):
        row = df.iloc[int(idx)]
        results.append(Alert(
            tourist_id=str(row.get("tourist_id", "")),
            anomaly_type=a["anomaly_type"],
            alert_level=SEVERITY_MAP.get(a["anomaly_type"], "INFO"),
            confidence_score=a.get("confidence_score", 0.0),
            location={"lat": float(row.get("lat", 0.0)), "lng": float(row.get("lng", 0.0))},
            timestamp=pd.to_datetime(row.get("timestamp")),
            raw_evidence=row.to_dict(),
            model_version=MODEL_VERSION
        ).dict())

    payload = {"status": "success", "anomaly_count": len(results), "alerts": results}
    _log_detection({
        "type": "batch",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_count": int(len(df)),
        "anomaly_count": int(len(results)),
        "alerts": results,
        "iot_features": _compute_iot_features_df(df)
    })
    if len(results) > 0:
        _send_webhook(results)
    return payload

def _compute_iot_features_df(df: pd.DataFrame):
    """Compute lightweight IoT-related features for logging/analysis only.
    Does NOT change the 20-feature model input.
    Returns a list of dicts aligned to df rows.
    """
    out = []
    hr = df.get('heart_rate') if 'heart_rate' in df.columns else pd.Series([None]*len(df))
    batt = df.get('battery_level') if 'battery_level' in df.columns else pd.Series([None]*len(df))
    net = df.get('network_status') if 'network_status' in df.columns else pd.Series([None]*len(df))
    panic = df.get('panic_button') if 'panic_button' in df.columns else pd.Series([False]*len(df))
    acc = df.get('accuracy') if 'accuracy' in df.columns else pd.Series([None]*len(df))

    # Simple normalizations for logging analysis
    hr_mean = float(pd.to_numeric(hr, errors='coerce').dropna().mean()) if pd.to_numeric(hr, errors='coerce').notna().any() else 0.0
    hr_std = float(pd.to_numeric(hr, errors='coerce').dropna().std()) if pd.to_numeric(hr, errors='coerce').notna().any() else 1.0

    for i in range(len(df)):
        hr_i = pd.to_numeric(hr.iloc[i], errors='coerce') if i < len(hr) else None
        batt_i = pd.to_numeric(batt.iloc[i], errors='coerce') if i < len(batt) else None
        acc_i = pd.to_numeric(acc.iloc[i], errors='coerce') if i < len(acc) else None
        net_i = str(net.iloc[i]) if i < len(net) and net.iloc[i] is not None else None
        panic_i = bool(panic.iloc[i]) if i < len(panic) else False

        hr_z = None
        if hr_i is not None and hr_std != 0:
            hr_z = float((hr_i - hr_mean) / hr_std)

        out.append({
            "heart_rate": None if hr_i is None else float(hr_i),
            "heart_rate_z": hr_z,
            "battery_level": None if batt_i is None else float(batt_i),
            "network_status": net_i,
            "panic_button": panic_i,
            "accuracy": None if acc_i is None else float(acc_i)
        })
    return out

def _log_detection(record: dict):
    try:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        with (logs_dir / "detections.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _send_webhook(alerts: list):
    url = os.getenv("WEBHOOK_URL")
    if not url:
        return
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(url, json={"alerts": alerts})
    except Exception:
        # fail-safe: never crash detection on webhook errors
        _log_detection({"type": "webhook_error", "timestamp": datetime.now(timezone.utc).isoformat()})


from haversine import haversine

def extract_enhanced_features(data):
    """
    Extract 20 enhanced features matching the training format
    """
    n_samples = len(data)
    features = np.zeros((n_samples, 20))
    
    # Convert to numpy arrays for vectorized operations
    lats = data['lat'].values if 'lat' in data.columns else np.zeros(n_samples)
    lngs = data['lng'].values if 'lng' in data.columns else np.zeros(n_samples)
    timestamps = data['timestamp'].values if 'timestamp' in data.columns else np.arange(n_samples) * 60
    
    # Basic features
    features[:, 0] = lats
    features[:, 1] = lngs
    features[:, 2] = data.get('altitude', pd.Series(np.zeros(n_samples))).values
    
    # --- Distance features ---
    coords = list(zip(lats, lngs))
    distances = [0.0]  # first row has no previous point
    for i in range(1, len(coords)):
        distances.append(haversine(coords[i-1], coords[i]))
    distances = np.array(distances)  # in km
    
    # --- Time-based features ---
    time_diffs = data['timestamp'].diff().dt.total_seconds().fillna(0)
    velocities = np.where(time_diffs > 0, distances / time_diffs * 3600, 0)  # km/h
    
    # Save into features matrix
    features[:, 3] = distances
    features[:, 4] = time_diffs
    features[:, 5] = velocities
    
    return features


    
    # Enhanced distance calculations (vectorized)
    lat_rad = np.radians(lats)
    lng_rad = np.radians(lngs)
    
    # Vectorized haversine distance calculation
    prev_lat_rad = np.roll(lat_rad, 1)
    prev_lng_rad = np.roll(lng_rad, 1)
    
    dlat = lat_rad - prev_lat_rad
    dlng = lng_rad - prev_lng_rad
    
    # Haversine formula (fully vectorized)
    a = np.sin(dlat/2)**2 + np.cos(prev_lat_rad) * np.cos(lat_rad) * np.sin(dlng/2)**2
    distances = 6371000 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # in meters
    distances[0] = 0  # First point has no previous distance
    
    features[:, 4] = distances
    
    # Enhanced velocity calculation (vectorized)
    velocities = np.where(time_diffs > 0, distances / time_diffs * 3.6, 0)  # km/h
    features[:, 5] = np.clip(velocities, 0, 300)  # Increased max speed
    
    # Acceleration (vectorized)
    accelerations = np.diff(velocities, prepend=velocities[0])
    features[:, 6] = accelerations
    
    # Distance from trajectory start (vectorized cumsum)
    cumulative_distances = np.cumsum(distances)
    features[:, 7] = cumulative_distances
    
    # Enhanced bearing calculations (vectorized)
    y = np.sin(lng_rad - prev_lng_rad) * np.cos(lat_rad)
    x = (np.cos(prev_lat_rad) * np.sin(lat_rad) - 
         np.sin(prev_lat_rad) * np.cos(lat_rad) * np.cos(lng_rad - prev_lng_rad))
    
    bearings = np.degrees(np.arctan2(y, x))
    bearings = (bearings + 360) % 360  # Normalize to 0-360
    features[:, 8] = bearings
    
    # Bearing changes (vectorized)
    bearing_changes = np.diff(bearings, prepend=bearings[0])
    bearing_changes = np.where(bearing_changes > 180, bearing_changes - 360, bearing_changes)
    bearing_changes = np.where(bearing_changes < -180, bearing_changes + 360, bearing_changes)
    features[:, 9] = np.abs(bearing_changes)
    
    # Enhanced movement patterns using pandas rolling operations
    df_temp = pd.DataFrame({
        'velocity': velocities, 
        'distance': distances, 
        'acceleration': accelerations,
        'bearing_change': np.abs(bearing_changes)
    })
    
    # Rolling statistics (vectorized)
    features[:, 10] = df_temp['velocity'].rolling(window=10, min_periods=1).mean().values
    features[:, 11] = df_temp['velocity'].rolling(window=10, min_periods=1).std().fillna(0).values
    features[:, 12] = df_temp['distance'].rolling(window=5, min_periods=1).sum().values
    features[:, 13] = df_temp['acceleration'].rolling(window=5, min_periods=1).std().fillna(0).values
    features[:, 14] = df_temp['bearing_change'].rolling(window=3, min_periods=1).mean().values
    
    # Statistical features (vectorized)
    features[:, 15] = (velocities - np.mean(velocities)) / (np.std(velocities) + 1e-8)  # Velocity z-score
    features[:, 16] = (accelerations - np.mean(accelerations)) / (np.std(accelerations) + 1e-8)  # Acceleration z-score
    
    # Time-based contextual features (vectorized)
    try:
        datetime_series = pd.to_datetime(timestamps, unit='s')
        features[:, 17] = datetime_series.hour.values  # Hour of day
        features[:, 18] = datetime_series.dayofweek.values  # Day of week
    except:
        # Fallback for non-timestamp data
        features[:, 17] = np.random.randint(0, 24, n_samples)  # Random hour
        features[:, 18] = np.random.randint(0, 7, n_samples)   # Random day
    
    # Movement efficiency metric (vectorized)
    straight_line_distances = np.sqrt((lats - lats[0])**2 + (lngs - lngs[0])**2) * 111000
    movement_efficiency = np.where(cumulative_distances > 0, 
                                 straight_line_distances / cumulative_distances, 1)
    features[:, 19] = movement_efficiency
    
    return features

def predict_with_ensemble(model_data, features):
    """
    Predict using the enhanced ensemble model
    """
    predictions = {}
    scores = {}
    
    # Get ensemble models and scalers
    models = {
        'isolation_forest': model_data.get('isolation_forest'),
        'elliptic_envelope': model_data.get('elliptic_envelope'),
        'one_class_svm': model_data.get('one_class_svm')
    }
    
    scalers = model_data.get('scalers', {})
    weights = model_data.get('weights', {})
    
    # Get predictions from each model
    for name, model in models.items():
        if model is not None and name in scalers:
            scaler = scalers[name]
            features_scaled = scaler.transform(features)
            
            pred = model.predict(features_scaled)
            score = model.decision_function(features_scaled) if hasattr(model, 'decision_function') else pred
            
            predictions[name] = pred
            scores[name] = score
    
    # Weighted ensemble prediction (vectorized)
    ensemble_scores = np.zeros(len(features))
    ensemble_predictions = np.zeros(len(features))
    
    for name, weight in weights.items():
        if name in predictions:
            ensemble_predictions += predictions[name] * weight
            ensemble_scores += scores[name] * weight
    
    # Convert to binary predictions
    final_predictions = (ensemble_predictions < 0).astype(int)
    
    return final_predictions, ensemble_scores

def show_anomaly_results(data, anomaly_indices, anomaly_scores, top_n=10):
    """
    Display detailed results of anomaly detection
    """
    
    if len(anomaly_indices) == 0:
        print("✅ No anomalies detected")
        return
    
    print(f"\n🔍 Top {min(top_n, len(anomaly_indices))} Anomalies:")
    print("-" * 80)
    
    # Sort by anomaly score (most anomalous first)
    sorted_indices = anomaly_indices[np.argsort(anomaly_scores[anomaly_indices])]
    
    for i, idx in enumerate(sorted_indices[:top_n]):
        row = data.iloc[idx]
        print(f"\nAnomaly {i+1} (Row {idx}):")
        print(f"  Anomaly Score: {anomaly_scores[idx]:.4f}")
        
        # Show available data
        if 'heart_rate' in data.columns:
            print(f"  Heart Rate: {row['heart_rate']:.1f} BPM")
        if 'lat' in data.columns and 'lng' in data.columns:
            print(f"  Location: {row['lat']:.6f}, {row['lng']:.6f}")
        if 'timestamp' in data.columns:
            print(f"  Timestamp: {row['timestamp']}")

def save_anomaly_report(data, anomaly_indices, anomaly_scores, output_file='anomaly_report.csv'):
    """
    Save anomaly detection results to CSV file
    """
    
    # Create results dataframe
    results = data.copy()
    results['anomaly_score'] = anomaly_scores
    results['is_anomaly'] = False
    results.loc[anomaly_indices, 'is_anomaly'] = True
    
    # Save to CSV
    results.to_csv(output_file, index=False)
    print(f"📄 Anomaly report saved to: {output_file}")

# Example usage
if __name__ == "__main__":
    # Example: Create sample data and test
    print("🧪 Testing enhanced model with sample data...")
    
    # Create sample dataset
    sample_data = pd.DataFrame({
        'lat': [39.9042 + i*0.00001 for i in range(50)],
        'lng': [116.4074 + i*0.00001 for i in range(50)],
        'timestamp': [i*60 for i in range(50)],
        'heart_rate': [75, 80, 85, 500, 78, 82, 500, 76, 79, 83] + [75]*40,  # 2 anomalies
        'battery_level': [100] * 50,
        'network_status': [True] * 50,
        'panic_button': [False] * 50,
        'accuracy': [5.0] * 50
    })
    
    try:
        # Detect anomalies
        anomalies, scores = detect_anomalies(sample_data)
        
        # Show results
        show_anomaly_results(sample_data, anomalies, scores)
        
        # Save report
        save_anomaly_report(sample_data, anomalies, scores)
        
        print(f"\n✅ Enhanced deployment test completed successfully!")
        print(f"📊 Summary: {len(anomalies)} anomalies detected out of {len(sample_data)} points")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have the enhanced trained model file: trained_tourist_safety_model.pkl")
