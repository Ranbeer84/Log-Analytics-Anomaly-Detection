"""
Comprehensive test suite for Phase 4 ML components
Tests all models, training, inference, and evaluation
"""
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))

# Color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class Phase4Tester:
    """Test suite for Phase 4 components"""
    
    def __init__(self):
        self.results = {
            "models": {"passed": 0, "failed": 0, "tests": []},
            "training": {"passed": 0, "failed": 0, "tests": []},
            "inference": {"passed": 0, "failed": 0, "tests": []},
            "evaluation": {"passed": 0, "failed": 0, "tests": []}
        }
        self.temp_dir = None
    
    def print_header(self, text: str):
        """Print colored header"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
    
    def print_test(self, category: str, test_name: str, passed: bool, message: str = ""):
        """Print test result"""
        status = f"{Colors.OKGREEN}✓ PASS{Colors.ENDC}" if passed else f"{Colors.FAIL}✗ FAIL{Colors.ENDC}"
        print(f"  {status} - {test_name}")
        if message:
            print(f"    {Colors.OKCYAN}→{Colors.ENDC} {message}")
        
        self.results[category]["tests"].append({
            "name": test_name,
            "passed": passed,
            "message": message
        })
        
        if passed:
            self.results[category]["passed"] += 1
        else:
            self.results[category]["failed"] += 1
    
    def generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate sample log data for testing"""
        np.random.seed(42)
        
        data = {
            'timestamp': [
                datetime.now() - timedelta(minutes=i) 
                for i in range(n_samples)
            ],
            'message': [
                f"Log message {i}: " + np.random.choice([
                    "User login successful",
                    "Database query executed",
                    "API request received",
                    "Cache miss occurred",
                    "Error connecting to database",
                    "Timeout waiting for response",
                    "Memory usage high"
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
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
        df.loc[anomaly_indices, 'response_time'] *= 10
        df.loc[anomaly_indices, 'level'] = 'ERROR'
        
        return df
    
    def test_model_imports(self):
        """Test 1: Model imports"""
        self.print_header("TEST 1: Model Imports")
        
        try:
            from models.autoencoder import Autoencoder, AutoencoderTrainer, create_autoencoder
            self.print_test("models", "Autoencoder imports", True, "All classes imported")
        except Exception as e:
            self.print_test("models", "Autoencoder imports", False, str(e))
        
        try:
            from models.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderTrainer, create_lstm_autoencoder
            self.print_test("models", "LSTM Autoencoder imports", True, "All classes imported")
        except Exception as e:
            self.print_test("models", "LSTM Autoencoder imports", False, str(e))
        
        try:
            from models.isolation_forest import IsolationForestDetector
            self.print_test("models", "Isolation Forest imports", True, "Class imported")
        except Exception as e:
            self.print_test("models", "Isolation Forest imports", False, str(e))
    
    def test_autoencoder_model(self):
        """Test 2: Autoencoder model"""
        self.print_header("TEST 2: Autoencoder Model")
        
        try:
            from models.autoencoder import Autoencoder
            
            # Create model
            model = Autoencoder(input_dim=50, encoding_dims=[32, 16], dropout_rate=0.2)
            self.print_test("models", "Autoencoder initialization", True, 
                          f"Input: 50 → Latent: 16 → Output: 50")
            
            # Test forward pass
            x = torch.randn(10, 50)
            output = model(x)
            
            if output.shape == (10, 50):
                self.print_test("models", "Autoencoder forward pass", True, 
                              f"Output shape: {output.shape}")
            else:
                self.print_test("models", "Autoencoder forward pass", False, 
                              f"Expected (10, 50), got {output.shape}")
            
            # Test reconstruction error
            error = model.reconstruction_error(x)
            if error.shape == (10,):
                self.print_test("models", "Reconstruction error calculation", True,
                              f"Error shape: {error.shape}, Mean: {error.mean():.4f}")
            else:
                self.print_test("models", "Reconstruction error calculation", False,
                              f"Expected (10,), got {error.shape}")
            
        except Exception as e:
            self.print_test("models", "Autoencoder model", False, str(e))
    
    def test_lstm_autoencoder_model(self):
        """Test 3: LSTM Autoencoder model"""
        self.print_header("TEST 3: LSTM Autoencoder Model")
        
        try:
            from models.lstm_autoencoder import LSTMAutoencoder
            
            # Create model
            model = LSTMAutoencoder(
                input_dim=30,
                hidden_dim=64,
                num_layers=2,
                dropout=0.2
            )
            self.print_test("models", "LSTM Autoencoder initialization", True,
                          f"Input: 30, Hidden: 64, Layers: 2")
            
            # Test forward pass
            x = torch.randn(8, 10, 30)  # (batch, seq_len, features)
            output = model(x)
            
            if output.shape == (8, 10, 30):
                self.print_test("models", "LSTM forward pass", True,
                              f"Output shape: {output.shape}")
            else:
                self.print_test("models", "LSTM forward pass", False,
                              f"Expected (8, 10, 30), got {output.shape}")
            
            # Test reconstruction error
            error = model.reconstruction_error(x)
            if error.shape == (8,):
                self.print_test("models", "LSTM reconstruction error", True,
                              f"Error shape: {error.shape}")
            else:
                self.print_test("models", "LSTM reconstruction error", False,
                              f"Expected (8,), got {error.shape}")
            
        except Exception as e:
            self.print_test("models", "LSTM Autoencoder model", False, str(e))
    
    def test_data_preprocessor(self):
        """Test 4: Data preprocessor"""
        self.print_header("TEST 4: Data Preprocessor")
        
        try:
            from training.data_preprocessor import LogDataPreprocessor, create_sequences
            
            # Generate sample data
            df = self.generate_sample_data(500)
            
            # Create preprocessor
            preprocessor = LogDataPreprocessor(
                max_tfidf_features=20,
                scale_numerical=True,
                scaler_type='standard'
            )
            self.print_test("training", "Preprocessor initialization", True,
                          "Created with TF-IDF=20, StandardScaler")
            
            # Fit and transform
            X = preprocessor.fit_transform(df)
            
            if X.shape[0] == len(df):
                self.print_test("training", "Feature extraction", True,
                              f"Shape: {X.shape}, Features: {len(preprocessor.feature_names)}")
            else:
                self.print_test("training", "Feature extraction", False,
                              f"Expected {len(df)} rows, got {X.shape[0]}")
            
            # Test transform on new data
            df_new = self.generate_sample_data(100)
            X_new = preprocessor.transform(df_new)
            
            if X_new.shape[1] == X.shape[1]:
                self.print_test("training", "Transform consistency", True,
                              f"Same feature count: {X_new.shape[1]}")
            else:
                self.print_test("training", "Transform consistency", False,
                              f"Expected {X.shape[1]} features, got {X_new.shape[1]}")
            
            # Test sequence creation
            sequences = create_sequences(X, sequence_length=10, stride=1)
            expected_seqs = len(X) - 10 + 1
            if sequences.shape[0] == expected_seqs:
                self.print_test("training", "Sequence creation", True,
                              f"Created {sequences.shape[0]} sequences")
            else:
                self.print_test("training", "Sequence creation", False,
                              f"Expected {expected_seqs}, got {sequences.shape[0]}")
            
        except Exception as e:
            self.print_test("training", "Data preprocessor", False, str(e))
    
    def test_feature_transformers(self):
        """Test 5: Feature transformers"""
        self.print_header("TEST 5: Feature Transformers")
        
        try:
            from feature_engineering.transformers import (
                LogLevelEncoder,
                ServiceFrequencyEncoder,
                ResponseTimeTransformer,
                TemporalPatternExtractor,
                MessageLengthTransformer
            )
            
            df = self.generate_sample_data(200)
            
            # Test LogLevelEncoder
            encoder = LogLevelEncoder()
            level_features = encoder.fit_transform(df['level'])
            if level_features.shape == (200, 1):
                self.print_test("training", "LogLevelEncoder", True,
                              f"Encoded {len(df)} levels")
            else:
                self.print_test("training", "LogLevelEncoder", False,
                              f"Expected (200, 1), got {level_features.shape}")
            
            # Test ServiceFrequencyEncoder
            encoder = ServiceFrequencyEncoder()
            service_features = encoder.fit_transform(df['service'])
            if service_features.shape == (200, 1):
                self.print_test("training", "ServiceFrequencyEncoder", True,
                              "Encoded service frequencies")
            else:
                self.print_test("training", "ServiceFrequencyEncoder", False,
                              f"Expected (200, 1), got {service_features.shape}")
            
            # Test ResponseTimeTransformer
            transformer = ResponseTimeTransformer()
            rt_features = transformer.fit_transform(df['response_time'])
            if rt_features.shape[0] == 200 and rt_features.shape[1] > 1:
                self.print_test("training", "ResponseTimeTransformer", True,
                              f"Created {rt_features.shape[1]} features")
            else:
                self.print_test("training", "ResponseTimeTransformer", False,
                              f"Unexpected shape: {rt_features.shape}")
            
            # Test TemporalPatternExtractor
            transformer = TemporalPatternExtractor()
            temporal_features = transformer.fit_transform(df)
            if temporal_features.shape[0] == 200:
                self.print_test("training", "TemporalPatternExtractor", True,
                              f"Extracted {temporal_features.shape[1]} temporal features")
            else:
                self.print_test("training", "TemporalPatternExtractor", False,
                              f"Expected 200 rows, got {temporal_features.shape[0]}")
            
            # Test MessageLengthTransformer
            transformer = MessageLengthTransformer()
            msg_features = transformer.fit_transform(df['message'])
            if msg_features.shape == (200, 4):
                self.print_test("training", "MessageLengthTransformer", True,
                              "Created 4 message length features")
            else:
                self.print_test("training", "MessageLengthTransformer", False,
                              f"Expected (200, 4), got {msg_features.shape}")
            
        except Exception as e:
            self.print_test("training", "Feature transformers", False, str(e))
    
    def test_isolation_forest_training(self):
        """Test 6: Isolation Forest training"""
        self.print_header("TEST 6: Isolation Forest Training")
        
        try:
            from models.isolation_forest import IsolationForestDetector
            from training.data_preprocessor import LogDataPreprocessor
            
            # Generate data
            df = self.generate_sample_data(300)
            
            # Preprocess
            preprocessor = LogDataPreprocessor(max_tfidf_features=10)
            X = preprocessor.fit_transform(df)
            
            # Train model
            model = IsolationForestDetector(contamination=0.1, n_estimators=50)
            model.fit(X)
            
            self.print_test("training", "Isolation Forest training", True,
                          f"Trained on {X.shape[0]} samples with {X.shape[1]} features")
            
            # Test prediction
            predictions = model.predict(X)
            scores = model.score_samples(X)
            
            if len(predictions) == len(X) and len(scores) == len(X):
                n_anomalies = np.sum(predictions)
                self.print_test("training", "Isolation Forest inference", True,
                              f"Detected {n_anomalies} anomalies ({n_anomalies/len(X)*100:.1f}%)")
            else:
                self.print_test("training", "Isolation Forest inference", False,
                              "Prediction shape mismatch")
            
        except Exception as e:
            self.print_test("training", "Isolation Forest training", False, str(e))
    
    def test_autoencoder_training(self):
        """Test 7: Autoencoder training"""
        self.print_header("TEST 7: Autoencoder Training")
        
        try:
            from models.autoencoder import Autoencoder, AutoencoderTrainer
            from torch.utils.data import DataLoader, TensorDataset
            
            # Generate synthetic data
            X_train = np.random.randn(200, 30).astype(np.float32)
            X_val = np.random.randn(50, 30).astype(np.float32)
            
            # Create data loaders
            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_train)),
                batch_size=16,
                shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val)),
                batch_size=16
            )
            
            # Create and train model
            model = Autoencoder(input_dim=30, encoding_dims=[20, 10], dropout_rate=0.2)
            trainer = AutoencoderTrainer(model, learning_rate=0.01, device='cpu')
            
            history = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=5,
                early_stopping_patience=10,
                verbose=False
            )
            
            if len(history['train_losses']) > 0:
                final_loss = history['train_losses'][-1]
                self.print_test("training", "Autoencoder training", True,
                              f"Completed 5 epochs, final loss: {final_loss:.4f}")
            else:
                self.print_test("training", "Autoencoder training", False,
                              "No training history recorded")
            
            # Test prediction
            X_test = torch.FloatTensor(X_val)
            scores, predictions = trainer.predict_anomalies(X_test, percentile=90)
            
            if len(scores) == len(X_val):
                self.print_test("training", "Autoencoder inference", True,
                              f"Predicted {np.sum(predictions)} anomalies")
            else:
                self.print_test("training", "Autoencoder inference", False,
                              "Prediction count mismatch")
            
        except Exception as e:
            self.print_test("training", "Autoencoder training", False, str(e))
    
    def test_batch_detector(self):
        """Test 8: Batch detector"""
        self.print_header("TEST 8: Batch Detector")
        
        try:
            # First train and save a model
            from models.isolation_forest import IsolationForestDetector
            from training.data_preprocessor import LogDataPreprocessor
            from inference.batch_detector import BatchAnomalyDetector
            
            # Create temp directory
            self.temp_dir = tempfile.mkdtemp()
            
            # Generate and preprocess data
            df_train = self.generate_sample_data(300)
            preprocessor = LogDataPreprocessor(max_tfidf_features=10)
            X_train = preprocessor.fit_transform(df_train)
            
            # Train and save model
            model = IsolationForestDetector(contamination=0.1, n_estimators=50)
            model.fit(X_train)
            model.save(str(Path(self.temp_dir) / 'isolation_forest_model.pkl'))
            preprocessor.save(str(Path(self.temp_dir) / 'preprocessor.pkl'))
            
            self.print_test("inference", "Model saving", True,
                          f"Saved to {self.temp_dir}")
            
            # Load detector
            detector = BatchAnomalyDetector(
                model_dir=self.temp_dir,
                model_type='isolation_forest'
            )
            self.print_test("inference", "Batch detector loading", True,
                          "Loaded model and preprocessor")
            
            # Test detection
            df_test = self.generate_sample_data(100)
            scores, predictions, details = detector.detect(df_test)
            
            if len(scores) == 100 and len(predictions) == 100:
                n_anomalies = len(details)
                self.print_test("inference", "Batch detection", True,
                              f"Detected {n_anomalies} anomalies with details")
            else:
                self.print_test("inference", "Batch detection", False,
                              "Output length mismatch")
            
        except Exception as e:
            self.print_test("inference", "Batch detector", False, str(e))
        finally:
            # Cleanup
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
    
    def test_evaluation_metrics(self):
        """Test 9: Evaluation metrics"""
        self.print_header("TEST 9: Evaluation Metrics")
        
        try:
            from evaluation.metrics import AnomalyDetectionMetrics, evaluate_model
            
            # Generate synthetic predictions
            np.random.seed(42)
            y_true = np.random.choice([0, 1], size=200, p=[0.9, 0.1])
            y_scores = np.random.randn(200)
            y_pred = (y_scores > 0).astype(int)
            
            # Calculate metrics
            calculator = AnomalyDetectionMetrics()
            metrics = calculator.calculate_all_metrics(y_true, y_pred, y_scores)
            
            required_metrics = [
                'precision', 'recall', 'f1_score', 'accuracy',
                'roc_auc', 'pr_auc', 'confusion_matrix'
            ]
            
            has_all = all(m in metrics for m in required_metrics)
            
            if has_all:
                self.print_test("evaluation", "Metric calculation", True,
                              f"Calculated {len(metrics)} metrics")
            else:
                missing = [m for m in required_metrics if m not in metrics]
                self.print_test("evaluation", "Metric calculation", False,
                              f"Missing metrics: {missing}")
            
            # Test individual metrics
            precision = calculator.precision(y_true, y_pred)
            recall = calculator.recall(y_true, y_pred)
            f1 = calculator.f1(y_true, y_pred)
            
            if 0 <= precision <= 1 and 0 <= recall <= 1 and 0 <= f1 <= 1:
                self.print_test("evaluation", "Metric ranges", True,
                              f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
            else:
                self.print_test("evaluation", "Metric ranges", False,
                              "Metrics out of valid range")
            
            # Test detection rate at K
            detection_rate = calculator.detection_rate_at_k(y_true, y_scores, k=20)
            if 0 <= detection_rate <= 1:
                self.print_test("evaluation", "Detection rate @ K", True,
                              f"Top-20 detection rate: {detection_rate:.3f}")
            else:
                self.print_test("evaluation", "Detection rate @ K", False,
                              f"Invalid rate: {detection_rate}")
            
        except Exception as e:
            self.print_test("evaluation", "Evaluation metrics", False, str(e))
    
    def test_visualization(self):
        """Test 10: Visualization"""
        self.print_header("TEST 10: Visualization")
        
        try:
            from evaluation.visualization import AnomalyVisualization
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            # Generate synthetic data
            np.random.seed(42)
            y_true = np.random.choice([0, 1], size=200, p=[0.9, 0.1])
            y_scores = np.random.randn(200)
            y_pred = (y_scores > 0).astype(int)
            
            # Test confusion matrix plot
            try:
                AnomalyVisualization.plot_confusion_matrix(
                    y_true, y_pred,
                    title='Test Confusion Matrix',
                    save_path=None
                )
                import matplotlib.pyplot as plt
                plt.close('all')
                self.print_test("evaluation", "Confusion matrix plot", True,
                              "Plot created successfully")
            except Exception as e:
                self.print_test("evaluation", "Confusion matrix plot", False, str(e))
            
            # Test ROC curve plot
            try:
                AnomalyVisualization.plot_roc_curve(
                    y_true, y_scores,
                    title='Test ROC',
                    save_path=None
                )
                plt.close('all')
                self.print_test("evaluation", "ROC curve plot", True,
                              "Plot created successfully")
            except Exception as e:
                self.print_test("evaluation", "ROC curve plot", False, str(e))
            
            # Test PR curve plot
            try:
                AnomalyVisualization.plot_precision_recall_curve(
                    y_true, y_scores,
                    title='Test PR',
                    save_path=None
                )
                plt.close('all')
                self.print_test("evaluation", "PR curve plot", True,
                              "Plot created successfully")
            except Exception as e:
                self.print_test("evaluation", "PR curve plot", False, str(e))
            
            # Test score distribution plot
            try:
                AnomalyVisualization.plot_score_distribution(
                    y_true, y_scores,
                    title='Test Distribution',
                    save_path=None
                )
                plt.close('all')
                self.print_test("evaluation", "Score distribution plot", True,
                              "Plot created successfully")
            except Exception as e:
                self.print_test("evaluation", "Score distribution plot", False, str(e))
            
        except Exception as e:
            self.print_test("evaluation", "Visualization", False, str(e))
    
    def print_summary(self):
        """Print test summary"""
        self.print_header("TEST SUMMARY")
        
        total_passed = 0
        total_failed = 0
        
        for category, data in self.results.items():
            passed = data["passed"]
            failed = data["failed"]
            total = passed + failed
            total_passed += passed
            total_failed += failed
            
            status_color = Colors.OKGREEN if failed == 0 else Colors.WARNING
            
            print(f"{status_color}{category.upper()}:{Colors.ENDC} {passed}/{total} tests passed")
        
        print(f"\n{Colors.BOLD}OVERALL:{Colors.ENDC} {total_passed}/{total_passed + total_failed} tests passed")
        
        if total_failed == 0:
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 ALL PHASE 4 TESTS PASSED! 🎉{Colors.ENDC}\n")
            return True
        else:
            print(f"\n{Colors.FAIL}{Colors.BOLD}❌ {total_failed} TEST(S) FAILED{Colors.ENDC}\n")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print(f"\n{Colors.BOLD}{Colors.OKBLUE}")
        print("╔═══════════════════════════════════════════════════════════════════╗")
        print("║                                                                   ║")
        print("║       PHASE 4 ML COMPONENTS - COMPREHENSIVE TEST SUITE           ║")
        print("║                                                                   ║")
        print("╚═══════════════════════════════════════════════════════════════════╝")
        print(Colors.ENDC)
        
        try:
            self.test_model_imports()
            self.test_autoencoder_model()
            self.test_lstm_autoencoder_model()
            self.test_data_preprocessor()
            self.test_feature_transformers()
            self.test_isolation_forest_training()
            self.test_autoencoder_training()
            self.test_batch_detector()
            self.test_evaluation_metrics()
            self.test_visualization()
            
            return self.print_summary()
            
        except Exception as e:
            print(f"\n{Colors.FAIL}Critical error during testing: {e}{Colors.ENDC}\n")
            return False


def main():
    """Main test runner"""
    print(f"{Colors.WARNING}⚠️  This will test all Phase 4 ML components{Colors.ENDC}")
    print(f"{Colors.WARNING}   Make sure you have all dependencies installed:{Colors.ENDC}")
    print(f"   • torch")
    print(f"   • scikit-learn")
    print(f"   • pandas")
    print(f"   • matplotlib")
    print(f"   • seaborn\n")
    
    input(f"{Colors.OKCYAN}Press Enter to start tests...{Colors.ENDC}")
    
    tester = Phase4Tester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()