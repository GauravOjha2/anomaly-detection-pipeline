import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import joblib
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class EnhancedVectorizedGeolifeProcessor:
    """
    Enhanced fully vectorized processing with improved performance
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.scaler = RobustScaler()  # More robust to outliers
        
    def load_all_trajectories(self, max_users: int = 50) -> pd.DataFrame:
        """
        Load multiple Geolife trajectories using vectorized operations
        """
        all_data = []
        
        # Get all user directories
        user_dirs = sorted([d for d in os.listdir(self.data_dir) 
                          if os.path.isdir(os.path.join(self.data_dir, d))])[:max_users]
        
        for user_id in user_dirs:
            trajectory_dir = os.path.join(self.data_dir, user_id, 'Trajectory')
            
            if not os.path.exists(trajectory_dir):
                continue
                
            # Get all .plt files for this user
            plt_files = glob.glob(os.path.join(trajectory_dir, '*.plt'))
            
            for plt_file in plt_files[:5]:  # Increased to 5 trajectories per user
                try:
                    # Load single trajectory
                    df = pd.read_csv(plt_file, skiprows=6, 
                                   names=['lat', 'lng', 'zero', 'altitude', 'days', 'date', 'time'])
                    
                    if len(df) < 10:  # Skip very short trajectories
                        continue
                    
                    # Add metadata
                    df['user_id'] = user_id
                    df['trajectory_id'] = os.path.basename(plt_file)
                    
                    # Convert datetime to timestamp using vectorized operations
                    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                    df['timestamp'] = df['datetime'].astype(np.int64) // 10**9
                    
                    all_data.append(df)
                    
                except Exception as e:
                    continue
        
        if not all_data:
            raise ValueError("No valid trajectory data found")
        
        # Concatenate all data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_df)} GPS points from {len(all_data)} trajectories")
        
        return combined_df
    
    def extract_enhanced_vectorized_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Enhanced feature extraction with better anomaly detection capabilities
        """
        # Sort by user and timestamp for proper sequence
        df = df.sort_values(['user_id', 'trajectory_id', 'timestamp']).reset_index(drop=True)
        
        # Initialize feature array with more features
        n_samples = len(df)
        feature_matrix = np.zeros((n_samples, 20))  # Increased feature count
        
        # Basic features (vectorized)
        feature_matrix[:, 0] = df['lat'].values
        feature_matrix[:, 1] = df['lng'].values  
        feature_matrix[:, 2] = df['altitude'].values
        
        # Time-based features (vectorized)
        timestamps = df['timestamp'].values
        time_diffs = np.diff(timestamps, prepend=timestamps[0])
        feature_matrix[:, 3] = time_diffs
        
        # Enhanced distance calculations (vectorized using broadcasting)
        lats = df['lat'].values
        lngs = df['lng'].values
        
        # Convert to radians for distance calculation
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
        
        feature_matrix[:, 4] = distances
        
        # Enhanced velocity calculation (vectorized)
        velocities = np.where(time_diffs > 0, distances / time_diffs * 3.6, 0)  # km/h
        feature_matrix[:, 5] = np.clip(velocities, 0, 300)  # Increased max speed
        
        # Acceleration (vectorized)
        accelerations = np.diff(velocities, prepend=velocities[0])
        feature_matrix[:, 6] = accelerations
        
        # Distance from trajectory start (vectorized cumsum)
        cumulative_distances = np.cumsum(distances)
        feature_matrix[:, 7] = cumulative_distances
        
        # Enhanced bearing calculations (vectorized)
        y = np.sin(lng_rad - prev_lng_rad) * np.cos(lat_rad)
        x = (np.cos(prev_lat_rad) * np.sin(lat_rad) - 
             np.sin(prev_lat_rad) * np.cos(lat_rad) * np.cos(lng_rad - prev_lng_rad))
        
        bearings = np.degrees(np.arctan2(y, x))
        bearings = (bearings + 360) % 360  # Normalize to 0-360
        feature_matrix[:, 8] = bearings
        
        # Bearing changes (vectorized)
        bearing_changes = np.diff(bearings, prepend=bearings[0])
        bearing_changes = np.where(bearing_changes > 180, bearing_changes - 360, bearing_changes)
        bearing_changes = np.where(bearing_changes < -180, bearing_changes + 360, bearing_changes)
        feature_matrix[:, 9] = np.abs(bearing_changes)
        
        # Enhanced movement patterns using pandas rolling operations
        df_temp = pd.DataFrame({
            'velocity': velocities, 
            'distance': distances, 
            'acceleration': accelerations,
            'bearing_change': np.abs(bearing_changes)
        })
        
        # Rolling statistics (vectorized)
        feature_matrix[:, 10] = df_temp['velocity'].rolling(window=10, min_periods=1).mean().values
        feature_matrix[:, 11] = df_temp['velocity'].rolling(window=10, min_periods=1).std().fillna(0).values
        feature_matrix[:, 12] = df_temp['distance'].rolling(window=5, min_periods=1).sum().values
        feature_matrix[:, 13] = df_temp['acceleration'].rolling(window=5, min_periods=1).std().fillna(0).values
        feature_matrix[:, 14] = df_temp['bearing_change'].rolling(window=3, min_periods=1).mean().values
        
        # Statistical features (vectorized)
        feature_matrix[:, 15] = (velocities - np.mean(velocities)) / (np.std(velocities) + 1e-8)  # Velocity z-score
        feature_matrix[:, 16] = (accelerations - np.mean(accelerations)) / (np.std(accelerations) + 1e-8)  # Acceleration z-score
        
        # Time-based contextual features (vectorized)
        datetime_series = pd.to_datetime(df['timestamp'], unit='s')
        feature_matrix[:, 17] = datetime_series.dt.hour.values  # Hour of day
        feature_matrix[:, 18] = datetime_series.dt.dayofweek.values  # Day of week
        
        # Movement efficiency metric (vectorized)
        straight_line_distances = np.sqrt((lats - lats[0])**2 + (lngs - lngs[0])**2) * 111000
        movement_efficiency = np.where(cumulative_distances > 0, 
                                     straight_line_distances / cumulative_distances, 1)
        feature_matrix[:, 19] = movement_efficiency
        
        return feature_matrix
    
    def create_enhanced_anomaly_labels(self, features: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """
        Create enhanced anomaly labels using multiple statistical thresholds
        """
        # Enhanced criteria for anomaly detection (all vectorized)
        
        # Criterion 1: Extreme velocities (more realistic thresholds)
        velocity_col = 5
        velocity_high_threshold = np.percentile(features[:, velocity_col], 98)  # Top 2%
        velocity_low_threshold = np.percentile(features[:, velocity_col], 2)   # Bottom 2%
        velocity_anomalies = (features[:, velocity_col] > velocity_high_threshold) | \
                           (features[:, velocity_col] < velocity_low_threshold)
        
        # Criterion 2: Large acceleration changes (vectorized)
        accel_col = 6
        accel_threshold = np.percentile(np.abs(features[:, accel_col]), 97)
        accel_anomalies = np.abs(features[:, accel_col]) > accel_threshold
        
        # Criterion 3: Sharp direction changes (vectorized)
        bearing_col = 9
        bearing_threshold = np.percentile(features[:, bearing_col], 95)
        bearing_anomalies = features[:, bearing_col] > bearing_threshold
        
        # Criterion 4: Movement efficiency anomalies (new)
        efficiency_col = 19
        efficiency_threshold = np.percentile(features[:, efficiency_col], 5)  # Very inefficient movement
        efficiency_anomalies = features[:, efficiency_col] < efficiency_threshold
        
        # Criterion 5: Time-based anomalies (unusual times)
        hour_col = 17
        unusual_hours = ((features[:, hour_col] >= 2) & (features[:, hour_col] <= 5))  # 2-5 AM
        
        # Criterion 6: Statistical outliers using multiple features
        velocity_zscore = np.abs(features[:, 15])  # Velocity z-score
        accel_zscore = np.abs(features[:, 16])     # Acceleration z-score
        statistical_anomalies = (velocity_zscore > 3) | (accel_zscore > 3)
        
        # Combine criteria using logical OR (vectorized)
        all_anomalies = (velocity_anomalies | accel_anomalies | bearing_anomalies | 
                        efficiency_anomalies | unusual_hours | statistical_anomalies)
        
        # Limit to desired contamination rate (vectorized)
        n_anomalies = int(len(features) * contamination)
        
        if np.sum(all_anomalies) > n_anomalies:
            # Prioritize anomalies by severity score
            severity_scores = (velocity_anomalies.astype(float) * 0.3 +
                             accel_anomalies.astype(float) * 0.25 +
                             bearing_anomalies.astype(float) * 0.2 +
                             efficiency_anomalies.astype(float) * 0.15 +
                             unusual_hours.astype(float) * 0.1)
            
            # Select top anomalies by severity
            top_anomaly_indices = np.argsort(severity_scores)[-n_anomalies:]
            labels = np.ones(len(features))  # Start with all normal
            labels[top_anomaly_indices] = 0  # Mark selected as anomalies
        else:
            labels = np.where(all_anomalies, 0, 1)  # 0 = anomaly, 1 = normal
        
        return labels

class EnhancedEnsembleDetector:
    """
    Enhanced ensemble detector with multiple algorithms and hyperparameter tuning
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.weights = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit ensemble of models with hyperparameter tuning
        """
        X_normal = X[y == 1] if y is not None else X
        
        # Model 1: Isolation Forest with hyperparameter tuning
        iso_params = {
            'contamination': [0.05, 0.1, 0.15],
            'n_estimators': [100, 200, 300],
            'max_features': [0.8, 1.0]
        }
        
        iso_forest = IsolationForest(random_state=42)
        iso_grid = GridSearchCV(iso_forest, iso_params, cv=3, scoring='f1')
        
        scaler_iso = StandardScaler()
        X_scaled_iso = scaler_iso.fit_transform(X_normal)
        
        # Create dummy labels for grid search (assuming normal data)
        y_dummy = np.ones(len(X_scaled_iso))
        
        try:
            iso_grid.fit(X_scaled_iso, y_dummy)
            self.models['isolation_forest'] = iso_grid.best_estimator_
        except:
            self.models['isolation_forest'] = IsolationForest(contamination=0.1, random_state=42)
            self.models['isolation_forest'].fit(X_scaled_iso)
        
        self.scalers['isolation_forest'] = scaler_iso
        
        # Model 2: Elliptic Envelope (Robust Covariance)
        elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
        scaler_elliptic = RobustScaler()
        X_scaled_elliptic = scaler_elliptic.fit_transform(X_normal)
        elliptic.fit(X_scaled_elliptic)
        
        self.models['elliptic_envelope'] = elliptic
        self.scalers['elliptic_envelope'] = scaler_elliptic
        
        # Model 3: One-Class SVM (for smaller datasets)
        if len(X_normal) < 10000:  # Only use for smaller datasets due to computational cost
            svm = OneClassSVM(gamma='scale', nu=0.1)
            scaler_svm = StandardScaler()
            X_scaled_svm = scaler_svm.fit_transform(X_normal)
            svm.fit(X_scaled_svm)
            
            self.models['one_class_svm'] = svm
            self.scalers['one_class_svm'] = scaler_svm
        
        # Set ensemble weights based on model complexity
        if 'one_class_svm' in self.models:
            self.weights = {'isolation_forest': 0.4, 'elliptic_envelope': 0.35, 'one_class_svm': 0.25}
        else:
            self.weights = {'isolation_forest': 0.6, 'elliptic_envelope': 0.4}
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using ensemble of models
        """
        predictions = {}
        scores = {}
        
        for name, model in self.models.items():
            scaler = self.scalers[name]
            X_scaled = scaler.transform(X)
            
            pred = model.predict(X_scaled)
            score = model.decision_function(X_scaled) if hasattr(model, 'decision_function') else pred
            
            predictions[name] = pred
            scores[name] = score
        
        # Weighted ensemble prediction (vectorized)
        ensemble_scores = np.zeros(len(X))
        ensemble_predictions = np.zeros(len(X))
        
        for name, weight in self.weights.items():
            if name in predictions:
                ensemble_predictions += predictions[name] * weight
                ensemble_scores += scores[name] * weight
        
        # Convert to binary predictions
        final_predictions = (ensemble_predictions < 0).astype(int)
        
        return final_predictions, ensemble_scores

def save_enhanced_model(model, scaler, model_path='trained_tourist_safety_model.pkl'):
    """
    Save enhanced model with better structure
    """
    model_data = {
        'isolation_forest': model.models.get('isolation_forest'),
        'elliptic_envelope': model.models.get('elliptic_envelope'),
        'one_class_svm': model.models.get('one_class_svm'),
        'scaler': scaler,
        'scalers': model.scalers,
        'weights': model.weights,
        'model_type': 'enhanced_ensemble',
        'version': '2.0'
    }
    
    joblib.dump(model_data, model_path)
    print(f"✅ Enhanced model saved to {model_path}")

def train_and_evaluate_enhanced_geolife(data_dir: str):
    """
    Enhanced training pipeline with improved performance
    """
    
    print("=== Enhanced Vectorized Geolife Anomaly Detection ===\n")
    
    # Step 1: Load and process data
    processor = EnhancedVectorizedGeolifeProcessor(data_dir)
    df = processor.load_all_trajectories(max_users=30)  # Increased for better training
    
    # Step 2: Extract enhanced features
    print("Extracting enhanced features...")
    features = processor.extract_enhanced_vectorized_features(df)
    labels = processor.create_enhanced_anomaly_labels(features, contamination=0.12)  # Slightly higher contamination
    
    print(f"Enhanced feature matrix shape: {features.shape}")
    print(f"Normal samples: {np.sum(labels == 1)}")
    print(f"Anomaly samples: {np.sum(labels == 0)}")
    
    # Step 3: Enhanced train/validation/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.6, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nEnhanced data split:")
    print(f"Train: {len(X_train)} samples ({len(X_train)/len(features)*100:.1f}%)")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/len(features)*100:.1f}%)")
    print(f"Test: {len(X_test)} samples ({len(X_test)/len(features)*100:.1f}%)")
    
    # Step 4: Train enhanced ensemble detector
    print("\n--- Training Enhanced Ensemble Detector ---")
    ensemble_detector = EnhancedEnsembleDetector()
    
    # Train on normal samples only
    X_train_normal = X_train[y_train == 1]
    ensemble_detector.fit(X_train_normal, y_train[y_train == 1])
    
    # Evaluate on test set
    y_pred_ensemble, scores_ensemble = ensemble_detector.predict(X_test)
    
    print("\nEnhanced Ensemble Detector Results:")
    print(classification_report(y_test, y_pred_ensemble, target_names=['Anomaly', 'Normal']))
    
    # Calculate detailed metrics
    precision = precision_score(y_test, y_pred_ensemble, pos_label=0)
    recall = recall_score(y_test, y_pred_ensemble, pos_label=0)
    f1 = f1_score(y_test, y_pred_ensemble, pos_label=0)
    accuracy = np.mean(y_test == y_pred_ensemble)
    
    print(f"\n=== Enhanced Performance Metrics ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Step 5: Save enhanced model
    scaler = StandardScaler().fit(X_train)
    save_enhanced_model(ensemble_detector, scaler)
    
    # Step 6: Enhanced visualizations
    plt.figure(figsize=(20, 12))
    
    # Plot 1: Enhanced feature distributions
    plt.subplot(3, 4, 1)
    feature_names = ['Lat', 'Lng', 'Alt', 'TimeDiff', 'Distance', 'Velocity', 
                    'Acceleration', 'CumDist', 'Bearing', 'BearingChange', 
                    'VelMean', 'VelStd', 'DistSum', 'AccelStd', 'BearingMean',
                    'VelZScore', 'AccelZScore', 'Hour', 'DayOfWeek', 'Efficiency']
    
    # Enhanced velocity distribution comparison
    normal_velocities = features[labels == 1, 5]
    anomaly_velocities = features[labels == 0, 5]
    
    plt.hist(normal_velocities, bins=50, alpha=0.7, label='Normal', density=True, color='blue')
    plt.hist(anomaly_velocities, bins=50, alpha=0.7, label='Anomaly', density=True, color='red')
    plt.xlabel('Velocity (km/h)')
    plt.ylabel('Density')
    plt.title('Enhanced Velocity Distribution')
    plt.legend()
    plt.xlim(0, 200)
    
    # Plot 2: Model performance comparison
    plt.subplot(3, 4, 2)
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    values = [precision, recall, f1, accuracy]
    colors = ['red' if v < 0.6 else 'orange' if v < 0.7 else 'green' for v in values]
    
    bars = plt.bar(metrics, values, color=colors)
    plt.ylabel('Score')
    plt.title('Enhanced Model Performance')
    plt.ylim(0, 1)
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center')
    
    # Plot 3: Confusion matrix
    plt.subplot(3, 4, 3)
    cm = confusion_matrix(y_test, y_pred_ensemble)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Anomaly', 'Normal'], 
                yticklabels=['Anomaly', 'Normal'])
    plt.title('Enhanced Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Plot 4: Feature importance (velocity vs acceleration)
    plt.subplot(3, 4, 4)
    plt.scatter(features[labels == 1, 5], features[labels == 1, 6], 
               alpha=0.5, s=10, label='Normal', color='blue')
    plt.scatter(features[labels == 0, 5], features[labels == 0, 6], 
               alpha=0.8, s=20, label='Anomaly', color='red')
    plt.xlabel('Velocity (km/h)')
    plt.ylabel('Acceleration (km/h/s)')
    plt.title('Velocity vs Acceleration')
    plt.legend()
    plt.xlim(0, 100)
    
    # Additional plots for enhanced analysis
    plt.subplot(3, 4, 5)
    plt.hist(scores_ensemble[y_test == 1], bins=30, alpha=0.7, label='Normal', density=True)
    plt.hist(scores_ensemble[y_test == 0], bins=30, alpha=0.7, label='Anomaly', density=True)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Enhanced Anomaly Score Distribution')
    plt.legend()
    
    plt.subplot(3, 4, 6)
    hour_normal = features[labels == 1, 17]
    hour_anomaly = features[labels == 0, 17]
    plt.hist(hour_normal, bins=24, alpha=0.7, label='Normal', density=True)
    plt.hist(hour_anomaly, bins=24, alpha=0.7, label='Anomaly', density=True)
    plt.xlabel('Hour of Day')
    plt.ylabel('Density')
    plt.title('Time-based Anomaly Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'enhanced_ensemble_detector': ensemble_detector,
        'enhanced_results': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        },
        'test_data': (X_test, y_test),
        'model_info': {
            'features': feature_names,
            'model_count': len(ensemble_detector.models),
            'feature_count': features.shape[1]
        }
    }

# Usage example
if __name__ == "__main__":
    # Path to your Geolife dataset
    geolife_path = input("Enter path to Geolife Data directory (or press Enter for synthetic demo): ").strip()
    
    if not geolife_path or not os.path.exists(geolife_path):
        print("Using synthetic data for enhanced demonstration...")
        
        # Create enhanced synthetic data
        np.random.seed(42)
        n_samples = 2000  # Increased sample size
        
        # More realistic feature generation
        features = np.random.randn(n_samples, 20)
        
        # Add realistic patterns
        # Velocity feature (column 5)
        features[:, 5] = np.abs(np.random.normal(30, 15, n_samples))  # Normal speed ~30 km/h
        
        # Heart rate simulation (if we had it)
        features[:, 0] = np.random.normal(75, 10, n_samples)  # Normal heart rate
        
        # Create more sophisticated anomalies
        n_anomalies = 200
        anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
        
        # Different types of anomalies
        for i, idx in enumerate(anomaly_indices):
            if i < 50:  # High speed anomalies
                features[idx, 5] = np.random.uniform(100, 200)
            elif i < 100:  # Sudden acceleration anomalies
                features[idx, 6] = np.random.uniform(10, 30)
            elif i < 150:  # Direction change anomalies
                features[idx, 9] = np.random.uniform(90, 180)
            else:  # Statistical outliers
                features[idx] += np.random.randn(20) * 4
        
        labels = np.ones(n_samples)
        labels[anomaly_indices] = 0
        
        print("Training enhanced ensemble on synthetic data...")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.6, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Train enhanced ensemble
        ensemble_detector = EnhancedEnsembleDetector()
        X_train_normal = X_train[y_train == 1]
        ensemble_detector.fit(X_train_normal, y_train[y_train == 1])
        
        # Evaluate
        y_pred, scores = ensemble_detector.predict(X_test)
        
        print("Enhanced Synthetic Data Results:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        scaler = StandardScaler().fit(X_train)
        save_enhanced_model(ensemble_detector, scaler)
        
        print(f"\n✅ Enhanced model training completed!")
        print(f"📊 Performance Summary:")
        print(f"   - Precision: {precision_score(y_test, y_pred, pos_label=0):.3f}")
        print(f"   - Recall: {recall_score(y_test, y_pred, pos_label=0):.3f}")
        print(f"   - F1-Score: {f1_score(y_test, y_pred, pos_label=0):.3f}")
        print(f"   - Accuracy: {np.mean(y_test == y_pred):.3f}")
        
    else:
        # Run with real Geolife data
        results = train_and_evaluate_enhanced_geolife(geolife_path)
        print("\n✅ Enhanced training completed! Model ready for improved deployment.")