# Enhanced Anomaly Detection System

## 🚀 Quick Start

Your enhanced anomaly detection model is ready for deployment!

### Basic Usage

```python
import pandas as pd
from anomaly_detector import detect_anomalies, show_anomaly_results

# Load your data
data = pd.read_csv('your_data.csv')

# Detect anomalies
anomalies, scores = detect_anomalies(data)

# View results
show_anomaly_results(data, anomalies, scores)
```

### Required Data Format

Your dataset should have these columns:

- `lat`, `lng` - GPS coordinates (required)
- `timestamp` - Time data (optional, for velocity calculation)
- `heart_rate` - Heart rate in BPM (optional)
- `battery_level` - Battery percentage (optional)
- `network_status` - Network connection (optional)
- `panic_button` - Panic button pressed (optional)
- `accuracy` - GPS accuracy in meters (optional)

### Enhanced Features

The new model uses **20 enhanced features** including:
- GPS trajectory analysis
- Velocity and acceleration patterns
- Movement efficiency metrics
- Time-based contextual features
- Statistical outlier detection

### Model Performance

- **Enhanced Ensemble Model** with 3 algorithms
- **20 Features** for better detection
- **Real-time Processing** with vectorized operations
- **High Recall** for anomaly detection

### Files

- `anomaly_detector.py` - Main deployment code
- `trained_tourist_safety_model.pkl` - Enhanced trained model
- `improved_training_script (2).py` - Training code (for reference)

### Example

```python
import pandas as pd
from anomaly_detector import detect_anomalies, show_anomaly_results, save_anomaly_report

# Load your real-world data
data = pd.read_csv('tourist_data.csv')

# Detect anomalies
anomalies, scores = detect_anomalies(data)

# Show top 10 anomalies
show_anomaly_results(data, anomalies, scores, top_n=10)

# Save detailed report
save_anomaly_report(data, anomalies, scores, 'my_anomaly_report.csv')
```

That's it! Your enhanced model is ready for real-world deployment.
