"""
Tests for ML models
"""
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch


@pytest.mark.asyncio
class TestCollaborativeFilteringModel:
    """Test collaborative filtering model"""
    
    @pytest.fixture
    def cf_model(self):
        """Create collaborative filtering model"""
        from app.ml.models.collaborative_filtering import CollaborativeFilteringModel
        return CollaborativeFilteringModel(
            num_users=100,
            num_items=1000,
            embedding_dim=64
        )
    
    def test_model_initialization(self, cf_model):
        """Test model initialization"""
        assert cf_model is not None
        assert hasattr(cf_model, 'user_embedding')
        assert hasattr(cf_model, 'item_embedding')
    
    def test_forward_pass(self, cf_model):
        """Test forward pass through model"""
        user_ids = torch.LongTensor([1, 2, 3])
        item_ids = torch.LongTensor([10, 20, 30])
        
        predictions = cf_model(user_ids, item_ids)
        
        assert predictions.shape == (3,)
        assert torch.all((predictions >= 0) & (predictions <= 5))
    
    def test_get_user_embedding(self, cf_model):
        """Test getting user embedding"""
        user_id = 5
        embedding = cf_model.get_user_embedding(user_id)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (64,)
    
    def test_get_item_embedding(self, cf_model):
        """Test getting item embedding"""
        item_id = 100
        embedding = cf_model.get_item_embedding(item_id)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (64,)
    
    def test_predict_batch(self, cf_model):
        """Test batch prediction"""
        user_ids = np.array([1, 2, 3, 4, 5])
        item_ids = np.array([10, 20, 30, 40, 50])
        
        predictions = cf_model.predict(user_ids, item_ids)
        
        assert len(predictions) == 5
        assert all(0 <= p <= 5 for p in predictions)
    
    def test_recommend_items_for_user(self, cf_model):
        """Test recommending items for a user"""
        user_id = 1
        candidate_items = list(range(10, 20))
        
        recommendations = cf_model.recommend(user_id, candidate_items, top_k=5)
        
        assert len(recommendations) == 5
        assert all(item_id in candidate_items for item_id, _ in recommendations)
        
        # Recommendations should be sorted by score
        scores = [score for _, score in recommendations]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
class TestDeepNeuralNetwork:
    """Test deep neural network model"""
    
    @pytest.fixture
    def dnn_model(self):
        """Create DNN model"""
        from app.ml.models.deep_nn import DeepNeuralNetwork
        return DeepNeuralNetwork(
            input_dim=128,
            hidden_dims=[256, 128, 64],
            output_dim=1,
            dropout=0.2
        )
    
    def test_model_initialization(self, dnn_model):
        """Test model initialization"""
        assert dnn_model is not None
        assert len(dnn_model.layers) > 0
    
    def test_forward_pass(self, dnn_model):
        """Test forward pass"""
        batch_size = 32
        input_features = torch.randn(batch_size, 128)
        
        output = dnn_model(input_features)
        
        assert output.shape == (batch_size, 1)
    
    def test_dropout_in_training_mode(self, dnn_model):
        """Test dropout is active in training mode"""
        dnn_model.train()
        input_features = torch.randn(10, 128)
        
        output1 = dnn_model(input_features)
        output2 = dnn_model(input_features)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output1, output2)
    
    def test_no_dropout_in_eval_mode(self, dnn_model):
        """Test dropout is inactive in eval mode"""
        dnn_model.eval()
        input_features = torch.randn(10, 128)
        
        with torch.no_grad():
            output1 = dnn_model(input_features)
            output2 = dnn_model(input_features)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2)


@pytest.mark.asyncio
class TestFeatureEngineering:
    """Test feature engineering"""
    
    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer"""
        from app.ml.features.feature_engineering import FeatureEngineer
        return FeatureEngineer()
    
    async def test_extract_user_features(self, feature_engineer, test_user, test_interactions, db_session):
        """Test extracting user features"""
        features = await feature_engineer.extract_user_features(
            user_id=test_user.id,
            db=db_session
        )
        
        assert isinstance(features, dict)
        assert "user_id" in features
        assert "interaction_count" in features
        assert "avg_rating" in features or features["interaction_count"] == 0
    
    async def test_extract_item_features(self, feature_engineer, test_items, db_session):
        """Test extracting item features"""
        item = test_items[0]
        features = await feature_engineer.extract_item_features(
            item_id=item.id,
            db=db_session
        )
        
        assert isinstance(features, dict)
        assert "item_id" in features
        assert "category" in features
        assert "price" in features
        assert "rating" in features
    
    async def test_encode_categorical_features(self, feature_engineer):
        """Test encoding categorical features"""
        category = "Electronics"
        encoded = feature_engineer.encode_category(category)
        
        assert isinstance(encoded, (int, np.ndarray))
    
    async def test_normalize_numerical_features(self, feature_engineer):
        """Test normalizing numerical features"""
        features = {"price": 100.0, "rating": 4.5}
        normalized = feature_engineer.normalize(features)
        
        assert all(0 <= v <= 1 for v in normalized.values() if isinstance(v, (int, float)))
    
    async def test_combine_user_item_features(self, feature_engineer, test_user, test_items, db_session):
        """Test combining user and item features"""
        user_features = await feature_engineer.extract_user_features(
            test_user.id, db_session
        )
        item_features = await feature_engineer.extract_item_features(
            test_items[0].id, db_session
        )
        
        combined = feature_engineer.combine_features(user_features, item_features)
        
        assert isinstance(combined, (dict, np.ndarray))


@pytest.mark.asyncio
class TestModelTraining:
    """Test model training"""
    
    @pytest.fixture
    def trainer(self):
        """Create model trainer"""
        from app.ml.training.trainer import ModelTrainer
        return ModelTrainer()
    
    def test_prepare_training_data(self, trainer, test_interactions):
        """Test preparing training data"""
        train_data, val_data = trainer.prepare_data(test_interactions)
        
        assert len(train_data) > 0
        assert len(val_data) >= 0
        assert len(train_data) + len(val_data) == len(test_interactions)
    
    def test_create_data_loaders(self, trainer):
        """Test creating data loaders"""
        X = np.random.rand(100, 10)
        y = np.random.rand(100, 1)
        
        train_loader, val_loader = trainer.create_data_loaders(
            X, y, batch_size=32, val_split=0.2
        )
        
        assert train_loader is not None
        assert val_loader is not None
    
    @patch('app.ml.training.trainer.ModelTrainer.train_epoch')
    def test_training_loop(self, mock_train_epoch, trainer):
        """Test training loop"""
        mock_train_epoch.return_value = 0.5  # Mock loss
        
        model = Mock()
        optimizer = Mock()
        train_loader = Mock()
        val_loader = Mock()
        
        history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=3
        )
        
        assert "train_loss" in history
        assert len(history["train_loss"]) == 3
    
    def test_early_stopping(self, trainer):
        """Test early stopping"""
        losses = [1.0, 0.9, 0.85, 0.86, 0.87]  # Loss stops improving
        
        should_stop = trainer.check_early_stopping(
            losses, patience=2, min_delta=0.01
        )
        
        assert should_stop


@pytest.mark.asyncio
class TestModelEvaluation:
    """Test model evaluation"""
    
    @pytest.fixture
    def evaluator(self):
        """Create model evaluator"""
        from app.ml.evaluation.evaluator import ModelEvaluator
        return ModelEvaluator()
    
    def test_calculate_rmse(self, evaluator):
        """Test RMSE calculation"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
        
        rmse = evaluator.calculate_rmse(y_true, y_pred)
        
        assert rmse > 0
        assert rmse < 1  # Should be reasonably small
    
    def test_calculate_precision_at_k(self, evaluator):
        """Test Precision@K calculation"""
        relevant_items = [1, 2, 3, 4, 5]
        recommended_items = [1, 2, 6, 7, 8]
        
        precision = evaluator.calculate_precision_at_k(
            recommended_items, relevant_items, k=5
        )
        
        assert 0 <= precision <= 1
        assert precision == 0.4  # 2 out of 5 correct
    
    def test_calculate_recall_at_k(self, evaluator):
        """Test Recall@K calculation"""
        relevant_items = [1, 2, 3, 4, 5]
        recommended_items = [1, 2, 6, 7, 8]
        
        recall = evaluator.calculate_recall_at_k(
            recommended_items, relevant_items, k=5
        )
        
        assert 0 <= recall <= 1
        assert recall == 0.4  # Found 2 out of 5 relevant
    
    def test_calculate_ndcg(self, evaluator):
        """Test NDCG calculation"""
        relevance_scores = [3, 2, 3, 0, 1]
        
        ndcg = evaluator.calculate_ndcg(relevance_scores, k=5)
        
        assert 0 <= ndcg <= 1
    
    def test_calculate_map(self, evaluator):
        """Test MAP calculation"""
        all_relevant = [[1, 2, 3], [4, 5, 6]]
        all_recommended = [[1, 7, 2], [4, 8, 5]]
        
        map_score = evaluator.calculate_map(all_recommended, all_relevant, k=3)
        
        assert 0 <= map_score <= 1
    
    def test_evaluate_model_comprehensive(self, evaluator):
        """Test comprehensive model evaluation"""
        y_true = np.random.rand(100)
        y_pred = y_true + np.random.normal(0, 0.1, 100)  # Add small noise
        
        metrics = evaluator.evaluate(y_true, y_pred)
        
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2_score" in metrics
        assert all(isinstance(v, float) for v in metrics.values())


@pytest.mark.asyncio
class TestModelPersistence:
    """Test model saving and loading"""
    
    def test_save_model(self, cf_model, tmp_path):
        """Test saving model to disk"""
        from app.ml.utils.model_utils import save_model
        
        model_path = tmp_path / "model.pth"
        save_model(cf_model, str(model_path))
        
        assert model_path.exists()
    
    def test_load_model(self, cf_model, tmp_path):
        """Test loading model from disk"""
        from app.ml.utils.model_utils import save_model, load_model
        
        model_path = tmp_path / "model.pth"
        save_model(cf_model, str(model_path))
        
        loaded_model = load_model(str(model_path))
        
        assert loaded_model is not None
        
        # Test that loaded model works
        user_ids = torch.LongTensor([1, 2, 3])
        item_ids = torch.LongTensor([10, 20, 30])
        
        original_pred = cf_model(user_ids, item_ids)
        loaded_pred = loaded_model(user_ids, item_ids)
        
        assert torch.allclose(original_pred, loaded_pred)