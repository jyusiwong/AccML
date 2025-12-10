"""
Neural Network-based Hypothesis Quality Predictor

This module implements a neural network that predicts hypothesis validation results
(F1 score, precision, recall) without running full CatBoost training.

Meta-Learning Approach:
1. Validate a subset of hypotheses with CatBoost (e.g., 1000-5000)
2. Train NN on hypothesis features -> validation results mapping
3. Use NN to predict results for remaining hypotheses (100x faster)
4. Optionally validate top predictions with full CatBoost

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HypothesisFeatureExtractor:
    """
    Extract numerical features from hypothesis dictionaries.
    Features capture the complexity and characteristics of each hypothesis.
    """
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, hypothesis: Dict) -> np.ndarray:
        """
        Extract numerical features from a hypothesis.
        
        Features:
        - Noise transformation parameters
        - Strategy complexity metrics
        - Feature usage patterns
        - Temporal/clustering settings
        
        Args:
            hypothesis: Hypothesis dictionary
            
        Returns:
            Feature vector (1D numpy array)
        """
        features = []
        
        # Noise parameters (8 features)
        noise_params = hypothesis.get('noise_params', {})
        features.extend([
            noise_params.get('scale', 0.1),
            noise_params.get('frequency', 5),
            noise_params.get('phase', 0),
            noise_params.get('amplitude', 1.0),
            noise_params.get('decay', 0.01),
            int(noise_params.get('reversible', True)),
            len(noise_params.get('exclude_features', [])),
            int(noise_params.get('temporal_aware', False))
        ])
        
        # Strategy encoding (10 features)
        strategy = hypothesis.get('strategy', '')
        strategy_types = ['temporal', 'clustering', 'feature', 'ensemble', 
                         'conservative', 'aggressive', 'balanced', 'hybrid',
                         'adaptive', 'static']
        for stype in strategy_types:
            features.append(int(stype in strategy.lower()))
        
        # Clustering parameters (5 features)
        cluster_params = hypothesis.get('cluster_params', {})
        features.extend([
            cluster_params.get('n_clusters', 5),
            cluster_params.get('min_samples', 10),
            int(cluster_params.get('use_hierarchical', False)),
            cluster_params.get('linkage_threshold', 0.5),
            int(cluster_params.get('adaptive_clusters', False))
        ])
        
        # Feature selection (7 features)
        feature_config = hypothesis.get('feature_config', {})
        features.extend([
            len(feature_config.get('selected_features', [])),
            len(feature_config.get('excluded_features', [])),
            int(feature_config.get('use_pca', False)),
            feature_config.get('pca_components', 0),
            int(feature_config.get('use_interaction', False)),
            feature_config.get('interaction_degree', 1),
            int(feature_config.get('normalize', True))
        ])
        
        # Model parameters (8 features)
        model_params = hypothesis.get('model_params', {})
        features.extend([
            model_params.get('iterations', 100),
            model_params.get('learning_rate', 0.1),
            model_params.get('depth', 6),
            model_params.get('l2_leaf_reg', 3),
            model_params.get('border_count', 128),
            model_params.get('subsample', 1.0),
            int(model_params.get('use_weights', False)),
            model_params.get('weight_scale', 1.0)
        ])
        
        # Complexity metrics (5 features)
        features.extend([
            len(str(hypothesis)),  # Hypothesis string length (complexity proxy)
            len(hypothesis.keys()),  # Number of configuration keys
            self._count_nested_dicts(hypothesis),  # Nesting depth
            int(hypothesis.get('validated', False)),
            hypothesis.get('generation', 0)
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _count_nested_dicts(self, d: Dict, depth: int = 0) -> int:
        """Count maximum nesting depth of dictionaries."""
        if not isinstance(d, dict):
            return depth
        if not d:
            return depth
        return max(self._count_nested_dicts(v, depth + 1) for v in d.values())
    
    def extract_batch(self, hypotheses: List[Dict]) -> np.ndarray:
        """
        Extract features from multiple hypotheses.
        
        Args:
            hypotheses: List of hypothesis dictionaries
            
        Returns:
            Feature matrix (num_hypotheses, num_features)
        """
        features = [self.extract_features(h) for h in tqdm(hypotheses, desc="Extracting features")]
        return np.vstack(features)


class HypothesisDataset(Dataset):
    """PyTorch Dataset for hypothesis features and validation results."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Args:
            features: Feature matrix (N, num_features)
            targets: Target matrix (N, 3) with [f1_score, precision, recall]
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class HypothesisQualityNet(nn.Module):
    """
    Neural Network for predicting hypothesis validation results.
    
    Architecture:
    - Input: Hypothesis features (43 dimensions)
    - Hidden layers with BatchNorm and Dropout
    - Output: [F1 score, Precision, Recall]
    """
    
    def __init__(self, input_dim: int = 43, hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.3):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer: 3 outputs (F1, precision, recall)
        layers.append(nn.Linear(prev_dim, 3))
        layers.append(nn.Sigmoid())  # Constrain outputs to [0, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class HypothesisPredictor:
    """
    Complete predictor system for hypothesis quality.
    Handles training, prediction, and model persistence.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.feature_extractor = HypothesisFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        
        logger.info(f"HypothesisPredictor initialized on device: {device}")
    
    def train(self, train_hypotheses: List[Dict], train_results: List[Dict],
              val_hypotheses: Optional[List[Dict]] = None,
              val_results: Optional[List[Dict]] = None,
              epochs: int = 100, batch_size: int = 32, lr: float = 0.001,
              early_stopping_patience: int = 10):
        """
        Train the neural network on validated hypotheses.
        
        Args:
            train_hypotheses: List of hypothesis dictionaries
            train_results: List of validation result dictionaries
            val_hypotheses: Optional validation set hypotheses
            val_results: Optional validation set results
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Training on {len(train_hypotheses)} hypotheses...")
        
        # Extract features
        X_train = self.feature_extractor.extract_batch(train_hypotheses)
        y_train = np.array([[r['f1_score'], r['precision'], r['recall']] 
                           for r in train_results], dtype=np.float32)
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        
        # Create datasets
        train_dataset = HypothesisDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=0)
        
        # Validation set if provided
        val_loader = None
        if val_hypotheses is not None and val_results is not None:
            X_val = self.feature_extractor.extract_batch(val_hypotheses)
            y_val = np.array([[r['f1_score'], r['precision'], r['recall']] 
                            for r in val_results], dtype=np.float32)
            X_val = self.scaler.transform(X_val)
            val_dataset = HypothesisDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                   shuffle=False, num_workers=0)
        
        # Initialize model
        input_dim = X_train.shape[1]
        self.model = HypothesisQualityNet(input_dim=input_dim).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                val_maes = []
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        mae = torch.abs(outputs - y_batch).mean()
                        
                        val_losses.append(loss.item())
                        val_maes.append(mae.item())
                
                avg_val_loss = np.mean(val_losses)
                avg_val_mae = np.mean(val_maes)
                history['val_loss'].append(avg_val_loss)
                history['val_mae'].append(avg_val_mae)
                
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - "
                              f"Train Loss: {avg_train_loss:.4f}, "
                              f"Val Loss: {avg_val_loss:.4f}, "
                              f"Val MAE: {avg_val_mae:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        logger.info("Training completed!")
        return history
    
    def predict(self, hypotheses: List[Dict], batch_size: int = 64) -> np.ndarray:
        """
        Predict validation results for hypotheses.
        
        Args:
            hypotheses: List of hypothesis dictionaries
            batch_size: Batch size for prediction
            
        Returns:
            Predictions array (N, 3) with [f1_score, precision, recall]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"Predicting for {len(hypotheses)} hypotheses...")
        
        # Extract and normalize features
        X = self.feature_extractor.extract_batch(hypotheses)
        X = self.scaler.transform(X)
        
        # Create dataset and loader
        dataset = HypothesisDataset(X, np.zeros((len(X), 3)))  # Dummy targets
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Predict
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for X_batch, _ in tqdm(loader, desc="Predicting"):
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.append(outputs.cpu().numpy())
        
        predictions = np.vstack(predictions)
        logger.info(f"Predictions complete. Shape: {predictions.shape}")
        
        return predictions
    
    def save(self, filepath: str):
        """Save model, scaler, and feature extractor."""
        save_dict = {
            'model_state': self.model.state_dict() if self.model else None,
            'scaler': self.scaler,
            'feature_extractor': self.feature_extractor,
            'device': self.device
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model, scaler, and feature extractor."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.scaler = save_dict['scaler']
        self.feature_extractor = save_dict['feature_extractor']
        
        if save_dict['model_state']:
            input_dim = len(self.scaler.mean_)
            self.model = HypothesisQualityNet(input_dim=input_dim).to(self.device)
            self.model.load_state_dict(save_dict['model_state'])
            self.model.eval()
        
        logger.info(f"Model loaded from {filepath}")


def hybrid_validation_workflow(hypotheses: List[Dict], 
                               X_train, y_train, X_test,
                               sample_size: int = 2000,
                               top_k: int = 5000,
                               nn_predictor_path: str = './data/hypothesis_predictor.pkl'):
    """
    Hybrid workflow: NN prediction + selective full validation.
    
    Steps:
    1. Validate random sample with CatBoost
    2. Train NN on sample results
    3. Predict all hypotheses with NN (fast)
    4. Full-validate top K predictions with CatBoost
    
    Args:
        hypotheses: List of all hypotheses
        X_train, y_train, X_test: Training/test data
        sample_size: Number to validate for NN training
        top_k: Number of top predictions to fully validate
        nn_predictor_path: Path to save/load NN model
        
    Returns:
        all_results: List of validation results
    """
    from gpu_validator import SequentialGPUValidator
    from pathlib import Path
    import pickle as pkl
    
    logger.info("=" * 80)
    logger.info("HYBRID VALIDATION WORKFLOW")
    logger.info("=" * 80)
    
    # Step 1: Check for pre-validated training data from notebook
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    
    nn_train_hyp_file = data_dir / 'nn_train_hypotheses.pkl'
    nn_train_res_file = data_dir / 'nn_train_results.pkl'
    
    if nn_train_hyp_file.exists() and nn_train_res_file.exists():
        logger.info(f"\n✓ Found pre-validated training data from notebook!")
        logger.info(f"  Loading from: {nn_train_hyp_file}")
        
        with open(nn_train_hyp_file, 'rb') as f:
            sample_hypotheses = pkl.load(f)
        with open(nn_train_res_file, 'rb') as f:
            sample_results = pkl.load(f)
        
        logger.info(f"✓ Loaded {len(sample_hypotheses)} pre-validated hypotheses")
        logger.info("  (Generated in notebook Cell 5b - saves ~30 minutes!)")
        
        # Use these instead of validating new sample
        logger.info(f"\nStep 1: Using pre-validated sample (skipping CatBoost validation)")
    else:
        # Fallback: Validate random sample
        logger.info(f"\nStep 1: No pre-validated data found. Validating random sample...")
        logger.info("  (Tip: Run notebook Cell 5b first to pre-generate this data)")
        
        sample_indices = np.random.choice(len(hypotheses), sample_size, replace=False)
        sample_hypotheses = [hypotheses[i] for i in sample_indices]
        
        validator = SequentialGPUValidator(X_train, y_train, X_test,
                                          iterations=5, learning_rate=1.0, depth=3)
        sample_results = validator.validate_batch(sample_hypotheses)
    
    # Step 2: Train NN
    logger.info(f"\nStep 2: Training neural network on {len(sample_results)} results...")
    predictor = HypothesisPredictor()
    
    # 80/20 train/val split
    n_train = int(0.8 * len(sample_hypotheses))
    train_hyp, val_hyp = sample_hypotheses[:n_train], sample_hypotheses[n_train:]
    train_res, val_res = sample_results[:n_train], sample_results[n_train:]
    
    predictor.train(train_hyp, train_res, val_hyp, val_res, 
                   epochs=100, batch_size=32, lr=0.001)
    predictor.save(nn_predictor_path)
    
    # Step 3: Predict all
    logger.info(f"\nStep 3: Predicting all {len(hypotheses)} hypotheses with NN...")
    predictions = predictor.predict(hypotheses, batch_size=64)
    
    # Create result dictionaries with predictions
    predicted_results = []
    for i, (hyp, pred) in enumerate(zip(hypotheses, predictions)):
        predicted_results.append({
            'hypothesis_id': i,
            'hypothesis': hyp,
            'f1_score': float(pred[0]),
            'precision': float(pred[1]),
            'recall': float(pred[2]),
            'prediction_method': 'neural_network'
        })
    
    # Step 4: Full validation of top K
    logger.info(f"\nStep 4: Full validation of top {top_k} predictions...")
    sorted_results = sorted(predicted_results, key=lambda x: x['f1_score'], reverse=True)
    top_indices = [r['hypothesis_id'] for r in sorted_results[:top_k]]
    top_hypotheses = [hypotheses[i] for i in top_indices]
    
    # Initialize validator if not already created (pre-validated path)
    if 'validator' not in locals():
        validator = SequentialGPUValidator(X_train, y_train, X_test,
                                          iterations=5, learning_rate=1.0, depth=3)
    
    top_validated = validator.validate_batch(top_hypotheses)
    
    # Update results with full validation
    for idx, result in zip(top_indices, top_validated):
        predicted_results[idx].update(result)
        predicted_results[idx]['prediction_method'] = 'catboost_validated'
    
    logger.info("\nHybrid validation complete!")
    logger.info(f"Total validated with CatBoost: {sample_size + top_k}")
    logger.info(f"NN predictions only: {len(hypotheses) - sample_size - top_k}")
    
    return predicted_results


if __name__ == "__main__":
    logger.info("Hypothesis Predictor NN module loaded successfully!")
    logger.info(f"PyTorch device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
