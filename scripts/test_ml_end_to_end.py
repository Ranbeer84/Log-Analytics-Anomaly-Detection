"""
End-to-end ML pipeline test
Trains all three models and evaluates them
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))

print("\n" + "="*70)
print("PHASE 4 - END-TO-END ML PIPELINE TEST")
print("="*70 + "\n")

# Generate sample data
print("📊 Generating sample log data...")
np.random.seed(42)

n_samples = 2000
data = {
    'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(n_samples)],
    'message': [
        f"Log {i}: " + np.random.choice([
            "User login successful",
            "Database query executed", 
            "API request processed",
            "Cache hit",
            "Error: Connection timeout",
            "Error: Database unreachable",
            "Warning: High memory usage"
        ]) for i in range(n_samples)
    ],
    'level': np.random.choice(
        ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        n_samples,
        p=[0.2, 0.5, 0.2, 0.1]
    ),
    'service': np.random.choice(
        ['api-service', 'auth-service', 'db-service', 'cache-service'],
        n_samples
    ),
    'response_time': np.random.lognormal(4, 1, n_samples)
}

df = pd.DataFrame(data)

# Add synthetic anomalies (10%)
anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
df.loc[anomaly_indices, 'response_time'] *= 15
df.loc[anomaly_indices, 'level'] = 'ERROR'

# Create labels (for evaluation)
y_true = np.zeros(n_samples)
y_true[anomaly_indices] = 1

print(f"✓ Generated {n_samples} log entries ({np.sum(y_true)} anomalies)")

# Split data
from sklearn.model_selection import train_test_split
train_df, test_df, y_train, y_test = train_test_split(
    df, y_true, test_size=0.3, random_state=42, stratify=y_true
)

print(f"✓ Split: {len(train_df)} train, {len(test_df)} test\n")

# Create temp directory for models
temp_dir = tempfile.mkdtemp()
print(f"📁 Using temp directory: {temp_dir}\n")

try:
    # ==========================================
    # TEST 1: ISOLATION FOREST
    # ==========================================
    print("="*70)
    print("TEST 1: ISOLATION FOREST")
    print("="*70 + "\n")
    
    from models.isolation_forest import IsolationForestDetector
    from training.data_preprocessor import LogDataPreprocessor
    
    # Preprocess
    print("🔧 Preprocessing data...")
    preprocessor = LogDataPreprocessor(max_tfidf_features=30)
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    print(f"✓ Features: {X_train.shape[1]}")
    
    # Train
    print("🎓 Training Isolation Forest...")
    if_model = IsolationForestDetector(contamination=0.1, n_estimators=100)
    if_model.fit(X_train)
    print("✓ Training complete")
    
    # Predict
    print("🔮 Predicting anomalies...")
    if_pred = if_model.predict(X_test)
    if_scores = if_model.score_samples(X_test)
    print(f"✓ Detected {np.sum(if_pred)} anomalies ({np.mean(if_pred)*100:.1f}%)")
    
    # Evaluate
    from evaluation.metrics import AnomalyDetectionMetrics
    metrics_calc = AnomalyDetectionMetrics()
    if_metrics = metrics_calc.calculate_all_metrics(y_test, if_pred, if_scores)
    
    print(f"\n📊 Isolation Forest Results:")
    print(f"   Precision: {if_metrics['precision']:.3f}")
    print(f"   Recall:    {if_metrics['recall']:.3f}")
    print(f"   F1 Score:  {if_metrics['f1_score']:.3f}")
    print(f"   ROC AUC:   {if_metrics['roc_auc']:.3f}")
    
    # ==========================================
    # TEST 2: AUTOENCODER
    # ==========================================
    print("\n" + "="*70)
    print("TEST 2: AUTOENCODER")
    print("="*70 + "\n")
    
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from models.autoencoder import Autoencoder, AutoencoderTrainer
    
    # Prepare data for neural network
    print("🔧 Preparing data for PyTorch...")
    preprocessor_ae = LogDataPreprocessor(max_tfidf_features=30, scaler_type='minmax')
    X_train_ae = preprocessor_ae.fit_transform(train_df)
    X_test_ae = preprocessor_ae.transform(test_df)
    
    # Train/val split
    X_train_split, X_val_split = train_test_split(X_train_ae, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_split)),
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_split)),
        batch_size=32
    )
    
    # Create and train model
    print("🎓 Training Autoencoder...")
    ae_model = Autoencoder(
        input_dim=X_train_ae.shape[1],
        encoding_dims=[64, 32],
        dropout_rate=0.2
    )
    ae_trainer = AutoencoderTrainer(ae_model, learning_rate=0.001, device='cpu')
    
    history = ae_trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        early_stopping_patience=5,
        verbose=False
    )
    
    print(f"✓ Training complete ({len(history['train_losses'])} epochs)")
    print(f"  Final train loss: {history['train_losses'][-1]:.4f}")
    print(f"  Final val loss:   {history['val_losses'][-1]:.4f}")
    
    # Predict
    print("🔮 Predicting anomalies...")
    X_test_tensor = torch.FloatTensor(X_test_ae)
    ae_scores, ae_pred = ae_trainer.predict_anomalies(X_test_tensor, percentile=90)
    print(f"✓ Detected {np.sum(ae_pred)} anomalies ({np.mean(ae_pred)*100:.1f}%)")
    
    # Evaluate
    ae_metrics = metrics_calc.calculate_all_metrics(y_test, ae_pred, ae_scores)
    
    print(f"\n📊 Autoencoder Results:")
    print(f"   Precision: {ae_metrics['precision']:.3f}")
    print(f"   Recall:    {ae_metrics['recall']:.3f}")
    print(f"   F1 Score:  {ae_metrics['f1_score']:.3f}")
    print(f"   ROC AUC:   {ae_metrics['roc_auc']:.3f}")
    
    # ==========================================
    # TEST 3: BATCH INFERENCE
    # ==========================================
    print("\n" + "="*70)
    print("TEST 3: BATCH INFERENCE SYSTEM")
    print("="*70 + "\n")
    
    # Save model
    print("💾 Saving Isolation Forest model...")
    if_model.save(str(Path(temp_dir) / 'isolation_forest_model.pkl'))
    preprocessor.save(str(Path(temp_dir) / 'preprocessor.pkl'))
    print("✓ Model saved")
    
    # Load with batch detector
    from inference.batch_detector import BatchAnomalyDetector
    
    print("📦 Loading model with BatchAnomalyDetector...")
    detector = BatchAnomalyDetector(
        model_dir=temp_dir,
        model_type='isolation_forest'
    )
    print("✓ Model loaded")
    
    # Run batch detection
    print("🔮 Running batch detection...")
    scores, predictions, details = detector.detect(test_df, batch_size=100)
    
    print(f"✓ Batch detection complete")
    print(f"  Detected {len(details)} anomalies")
    print(f"  Top anomaly score: {details[0]['anomaly_score']:.4f}")
    
    # ==========================================
    # TEST 4: VISUALIZATION
    # ==========================================
    print("\n" + "="*70)
    print("TEST 4: VISUALIZATION")
    print("="*70 + "\n")
    
    from evaluation.visualization import AnomalyVisualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("📈 Creating visualizations...")
    
    # ROC Curve
    AnomalyVisualization.plot_roc_curve(y_test, if_scores, save_path=None)
    plt.close('all')
    print("✓ ROC curve created")
    
    # Confusion Matrix
    AnomalyVisualization.plot_confusion_matrix(y_test, if_pred, save_path=None)
    plt.close('all')
    print("✓ Confusion matrix created")
    
    # Score Distribution
    AnomalyVisualization.plot_score_distribution(y_test, if_scores, save_path=None)
    plt.close('all')
    print("✓ Score distribution created")
    
    # ==========================================
    # MODEL COMPARISON
    # ==========================================
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70 + "\n")
    
    from evaluation.metrics import compare_models
    
    results = {
        'Isolation Forest': (if_pred, if_scores),
        'Autoencoder': (ae_pred, ae_scores)
    }
    
    comparison = compare_models(results, y_test)
    print(comparison.to_string(index=False))
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70 + "\n")
    
    print("✅ All components tested successfully!")
    print("\nComponents verified:")
    print("  ✓ Data preprocessing & feature engineering")
    print("  ✓ Isolation Forest training & inference")
    print("  ✓ Autoencoder training & inference")
    print("  ✓ Batch detection system")
    print("  ✓ Evaluation metrics calculation")
    print("  ✓ Visualization generation")
    print("  ✓ Model comparison\n")
    
    print("🎉 Phase 4 ML Pipeline is fully operational!\n")
    
    success = True

except Exception as e:
    print(f"\n❌ ERROR: {e}\n")
    import traceback
    traceback.print_exc()
    success = False

finally:
    # Cleanup
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temp directory\n")

sys.exit(0 if success else 1)