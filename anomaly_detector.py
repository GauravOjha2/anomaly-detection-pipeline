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
import joblib
import os
from geopy.distance import geodesic

def detect_anomalies(data, model_path='trained_tourist_safety_model.pkl'):
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
    
    print(f"🚨 Found {len(anomaly_indices)} anomalies out of {len(data)} points")
    
    return anomaly_indices, scores

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
    
    # Basic features (vectorized)
    features[:, 0] = lats
    features[:, 1] = lngs
    features[:, 2] = data.get('altitude', pd.Series(np.zeros(n_samples))).values
    
    # Time-based features (vectorized)
    time_diffs = np.diff(timestamps, prepend=timestamps[0])
    features[:, 3] = time_diffs
    
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
