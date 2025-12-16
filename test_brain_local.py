import sys
import numpy as np
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Import your modules
try:
    from ml.feature_engineering.extractors import LogFeatureExtractor, SequenceFeatureExtractor
    from ml.models.isolation_forest import IsolationForestDetector
    logger.info("✅ Successfully imported ML modules")
except ImportError as e:
    logger.error(f"❌ Import failed: {e}")
    logger.error("Make sure you are running this from the root 'ai-log-analytics' folder")
    sys.exit(1)

def run_simulation():
    print("\n--- 🧠 STARTING BRAIN SIMULATION (NO DOCKER) ---")
    
    # 1. Initialize Components
    feature_extractor = LogFeatureExtractor()
    model = IsolationForestDetector(contamination=0.1)
    
    # 2. Generate "Normal" Training Data (100 fake logs)
    print("\n[1/3] Generating training data...")
    normal_logs = []
    for i in range(100):
        normal_logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "info",
            "service": "api-gateway",
            "message": "User login successful",
            "response_time": np.random.normal(0.2, 0.05), # Fast response
            "status_code": 200
        })
    
    # 3. Train the Model
    print("[2/3] Training model...")
    features = feature_extractor.extract_batch_features(normal_logs)
    feature_names = feature_extractor.get_feature_names()
    stats = model.train(features, feature_names)
    print(f"   ✅ Model Trained! (Anomaly Rate: {stats['anomaly_rate']:.0%})")

    # 4. Test "Weird" Logs (Anomalies)
    print("\n[3/3] Testing scenarios...")
    
    scenarios = [
        {
            "name": "Normal Log",
            "log": {
                "timestamp": datetime.utcnow().isoformat(),
                "severity": "info",
                "service": "api-gateway",
                "message": "User logout",
                "response_time": 0.25,
                "status_code": 200
            }
        },
        {
            "name": "🔥 CRITICAL FAILURE (High Response Time + 500 Error)",
            "log": {
                "timestamp": datetime.utcnow().isoformat(),
                "severity": "critical",
                "service": "payment-service",
                "message": "Database connection timeout detected",
                "response_time": 5.0,  # Very slow
                "status_code": 500
            }
        }
    ]

    for scenario in scenarios:
        log = scenario['log']
        # Extract features
        log_features = feature_extractor.extract_features(log)
        
        # Predict
        result = model.predict_single(log_features)
        
        print(f"\nScenario: {scenario['name']}")
        print(f"   Message: {log['message']}")
        print(f"   Result:  {'🔴 ANOMALY' if result['is_anomaly'] else '🟢 NORMAL'}")
        print(f"   Score:   {result['anomaly_score']:.4f} (Severity: {result['severity']})")

if __name__ == "__main__":
    run_simulation()