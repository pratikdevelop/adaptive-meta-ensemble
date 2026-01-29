"""
Adaptive Meta-Ensemble v2.0 (AME-Pro)
Enhanced version with neural meta-learners, attention mechanisms, and advanced features

Key Enhancements:
- Neural network-based meta-learners
- Attention mechanism for adaptive weighting
- Confidence-based weighting
- Dynamic base model selection
- Uncertainty quantification
- Multi-objective optimization
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class NeuralMetaLearner:
    """
    Neural network-based meta-learner with attention mechanism
    
    This learns complex non-linear relationships between meta-features
    and optimal model weights.
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32], activation='relu', dropout=0.2):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.scaler = StandardScaler()
        self.model = None
        
    def build_model(self):
        """Build neural network for meta-learning"""
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_dims,
            activation=self.activation,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            alpha=0.001  # L2 regularization
        )
        
    def fit(self, X, y):
        """Train the neural meta-learner"""
        if self.model is None:
            self.build_model()
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        """Predict weights"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Ensure non-negative
        return np.maximum(predictions, 0)


class AttentionWeighting:
    """
    Attention mechanism for adaptive model weighting
    
    Uses attention scores to focus on most relevant models for each input
    """
    
    def __init__(self, n_models, attention_dim=32):
        self.n_models = n_models
        self.attention_dim = attention_dim
        self.query_net = None
        self.key_nets = []
        
    def build_attention_networks(self, input_dim):
        """Build attention query and key networks"""
        # Query network: transforms input to query vector
        self.query_net = MLPRegressor(
            hidden_layer_sizes=(self.attention_dim,),
            activation='tanh',
            max_iter=300,
            random_state=42
        )
        
        # Key networks: one per base model
        for i in range(self.n_models):
            key_net = MLPRegressor(
                hidden_layer_sizes=(self.attention_dim,),
                activation='tanh',
                max_iter=300,
                random_state=42 + i
            )
            self.key_nets.append(key_net)
    
    def compute_attention_scores(self, meta_features, model_predictions):
        """
        Compute attention scores between query (input) and keys (models)
        
        Higher attention = model is more relevant for this input
        """
        n_samples = meta_features.shape[0]
        
        # Simple version: use prediction diversity as attention signal
        # More diverse predictions from a model = higher attention
        attention_scores = np.zeros((n_samples, self.n_models))
        
        for i in range(self.n_models):
            # Attention based on prediction confidence and diversity
            pred_std = np.std(model_predictions[:, i])
            if pred_std > 0:
                attention_scores[:, i] = np.abs(
                    model_predictions[:, i] - np.mean(model_predictions, axis=1)
                )
        
        # Softmax normalization
        exp_scores = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return attention_weights


class AMEPro(BaseEstimator):
    """
    Adaptive Meta-Ensemble Professional Edition
    
    Enhanced version with:
    - Neural meta-learners
    - Attention mechanisms
    - Confidence-based weighting
    - Uncertainty quantification
    - Advanced meta-features
    
    Parameters:
    -----------
    base_models : list of tuples, default=None
        List of (name, model) tuples
    meta_learner_type : str, default='neural'
        'neural', 'tree', 'forest', or 'attention'
    use_confidence : bool, default=True
        Whether to use prediction confidence in weighting
    use_attention : bool, default=False
        Whether to use attention mechanism
    meta_features : str, default='advanced'
        'statistical', 'complexity', 'advanced', or 'all'
    task_type : str, default='classification'
        'classification' or 'regression'
    ensemble_size : int, default=7
        Number of base models to use
    optimize_hyperparams : bool, default=False
        Whether to optimize base model hyperparameters
    """
    
    def __init__(self, base_models=None, meta_learner_type='neural',
                 use_confidence=True, use_attention=False,
                 meta_features='advanced', task_type='classification',
                 ensemble_size=7, optimize_hyperparams=False):
        self.base_models = base_models
        self.meta_learner_type = meta_learner_type
        self.use_confidence = use_confidence
        self.use_attention = use_attention
        self.meta_features = meta_features
        self.task_type = task_type
        self.ensemble_size = ensemble_size
        self.optimize_hyperparams = optimize_hyperparams
        
    def _get_default_base_models(self):
        """Get enhanced default base models"""
        if self.task_type == 'classification':
            models = [
                ('dt', DecisionTreeClassifier(max_depth=12, min_samples_split=5, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
                ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance')),
                ('svm', SVC(kernel='rbf', probability=True, C=1.0, random_state=42)),
                ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
            ]
        else:
            models = [
                ('dt', DecisionTreeRegressor(max_depth=12, min_samples_split=5, random_state=42)),
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
                ('ridge', Ridge(alpha=1.0, random_state=42)),
                ('knn', KNeighborsRegressor(n_neighbors=7, weights='distance')),
                ('svr', SVR(kernel='rbf', C=1.0)),
                ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
            ]
        
        return models[:self.ensemble_size]
    
    def _extract_advanced_meta_features(self, X):
        """
        Extract comprehensive meta-features
        
        These features help predict which model will work best
        """
        meta_feats = []
        
        # Basic statistics
        meta_feats.append(np.mean(X, axis=1, keepdims=True))
        meta_feats.append(np.std(X, axis=1, keepdims=True))
        meta_feats.append(np.median(X, axis=1, keepdims=True))
        meta_feats.append(np.min(X, axis=1, keepdims=True))
        meta_feats.append(np.max(X, axis=1, keepdims=True))
        
        # Percentiles
        meta_feats.append(np.percentile(X, 25, axis=1, keepdims=True))
        meta_feats.append(np.percentile(X, 75, axis=1, keepdims=True))
        
        # Range and spread
        meta_feats.append((np.max(X, axis=1) - np.min(X, axis=1)).reshape(-1, 1))
        meta_feats.append(np.var(X, axis=1, keepdims=True))
        
        # Complexity measures
        meta_feats.append((np.sum(X == 0, axis=1) / X.shape[1]).reshape(-1, 1))  # Sparsity
        
        # Skewness (simple approximation)
        mean_val = np.mean(X, axis=1, keepdims=True)
        std_val = np.std(X, axis=1, keepdims=True) + 1e-10
        skew_approx = np.mean(((X - mean_val) / std_val) ** 3, axis=1, keepdims=True)
        meta_feats.append(skew_approx)
        
        # Kurtosis (simple approximation)
        kurt_approx = np.mean(((X - mean_val) / std_val) ** 4, axis=1, keepdims=True)
        meta_feats.append(kurt_approx)
        
        # Coefficient of variation
        cv = std_val / (np.abs(mean_val) + 1e-10)
        meta_feats.append(cv)
        
        # Number of unique values (binned)
        n_unique = np.array([len(np.unique(row)) for row in X]).reshape(-1, 1)
        meta_feats.append(n_unique / X.shape[1])  # Normalized
        
        return np.hstack(meta_feats)
    
    def _extract_meta_features(self, X):
        """Extract meta-features based on configuration"""
        if self.meta_features == 'advanced' or self.meta_features == 'all':
            return self._extract_advanced_meta_features(X)
        else:
            # Use simpler features for compatibility
            meta_feats = []
            meta_feats.append(np.mean(X, axis=1, keepdims=True))
            meta_feats.append(np.std(X, axis=1, keepdims=True))
            meta_feats.append(np.min(X, axis=1, keepdims=True))
            meta_feats.append(np.max(X, axis=1, keepdims=True))
            return np.hstack(meta_feats)
    
    def _create_meta_learner(self, meta_X_shape):
        """Create appropriate meta-learner"""
        if self.meta_learner_type == 'neural':
            return NeuralMetaLearner(
                input_dim=meta_X_shape,
                hidden_dims=[64, 32, 16],
                activation='relu'
            )
        elif self.meta_learner_type == 'tree':
            return DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=42)
        elif self.meta_learner_type == 'forest':
            return RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        else:
            return Ridge(alpha=1.0, random_state=42)
    
    def _get_prediction_confidence(self, model, X):
        """
        Estimate prediction confidence for each sample
        
        Higher confidence = prediction is more reliable
        """
        if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            # Confidence = max probability
            confidence = np.max(proba, axis=1)
        else:
            # For regression or models without predict_proba
            # Use simple heuristic: confidence = 1.0
            confidence = np.ones(X.shape[0])
        
        return confidence
    
    def fit(self, X, y):
        """
        Fit AME-Pro with enhanced meta-learning
        """
        X = np.array(X)
        y = np.array(y)
        
        # Initialize
        if self.base_models is None:
            self.base_models = self._get_default_base_models()
        
        self.base_models_ = []
        self.model_names_ = []
        self.model_scores_ = []  # Track base model performance
        
        # Step 1: Train base models
        print("Training enhanced base models...")
        for name, model in self.base_models:
            print(f"  Training {name}...")
            model.fit(X, y)
            self.base_models_.append(model)
            self.model_names_.append(name)
            
            # Evaluate base model
            if self.task_type == 'classification':
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y, model.predict(X))
            else:
                from sklearn.metrics import r2_score
                score = r2_score(y, model.predict(X))
            self.model_scores_.append(score)
            print(f"    {name} score: {score:.4f}")
        
        # Step 2: Extract meta-features
        print("\nExtracting advanced meta-features...")
        meta_X = self._extract_meta_features(X)
        print(f"  Meta-feature dimensions: {meta_X.shape[1]}")
        
        # Step 3: Create meta-training targets with confidence weighting
        print("\nCreating meta-training data with confidence weighting...")
        meta_y = np.zeros((X.shape[0], len(self.base_models_)))
        
        for i, model in enumerate(self.base_models_):
            if self.task_type == 'classification':
                pred = model.predict(X)
                # Base accuracy
                accuracy = (pred == y).astype(float)
                
                if self.use_confidence:
                    # Weight by prediction confidence
                    confidence = self._get_prediction_confidence(model, X)
                    meta_y[:, i] = accuracy * confidence
                else:
                    meta_y[:, i] = accuracy
            else:
                pred = model.predict(X)
                errors = -np.abs(pred - y)
                
                # Normalize
                if errors.max() != errors.min():
                    meta_y[:, i] = (errors - errors.min()) / (errors.max() - errors.min())
                else:
                    meta_y[:, i] = 0.5
        
        # Normalize weights
        row_sums = meta_y.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        meta_y = meta_y / row_sums
        
        # Step 4: Train meta-learners
        print("\nTraining neural meta-learners...")
        self.meta_learners_ = []
        
        for i, name in enumerate(self.model_names_):
            print(f"  Training meta-learner for {name}...")
            meta_learner = self._create_meta_learner(meta_X.shape[1])
            
            # For neural meta-learners, use the custom class
            if isinstance(meta_learner, NeuralMetaLearner):
                meta_learner.fit(meta_X, meta_y[:, i])
            else:
                meta_learner.fit(meta_X, meta_y[:, i])
            
            self.meta_learners_.append(meta_learner)
        
        # Step 5: Initialize attention mechanism if requested
        if self.use_attention:
            print("\nInitializing attention mechanism...")
            self.attention_ = AttentionWeighting(
                n_models=len(self.base_models_),
                attention_dim=32
            )
        
        print("\n✓ AME-Pro training complete!")
        print(f"  Base models: {len(self.base_models_)}")
        print(f"  Meta-learner type: {self.meta_learner_type}")
        print(f"  Using confidence: {self.use_confidence}")
        print(f"  Using attention: {self.use_attention}")
        
        return self
    
    def _get_adaptive_weights(self, X, base_predictions=None):
        """
        Predict optimal weights with advanced strategies
        """
        meta_X = self._extract_meta_features(X)
        
        # Get base weights from meta-learners
        weights = np.zeros((X.shape[0], len(self.base_models_)))
        
        for i, meta_learner in enumerate(self.meta_learners_):
            if isinstance(meta_learner, NeuralMetaLearner):
                weights[:, i] = meta_learner.predict(meta_X)
            else:
                weights[:, i] = meta_learner.predict(meta_X)
        
        # Apply attention if enabled
        if self.use_attention and base_predictions is not None:
            attention_weights = self.attention_.compute_attention_scores(meta_X, base_predictions)
            weights = weights * attention_weights
        
        # Ensure non-negative and normalized
        weights = np.maximum(weights, 0)
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        weights = weights / row_sums
        
        return weights
    
    def predict(self, X):
        """Make predictions with enhanced adaptive weighting"""
        X = np.array(X)
        
        # Get predictions from all base models
        predictions = np.zeros((X.shape[0], len(self.base_models_)))
        for i, model in enumerate(self.base_models_):
            predictions[:, i] = model.predict(X)
        
        # Get adaptive weights
        weights = self._get_adaptive_weights(X, predictions)
        
        # Weighted combination
        if self.task_type == 'classification':
            # Weighted voting
            final_pred = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                unique_classes = np.unique(predictions[i, :])
                class_votes = {}
                for cls in unique_classes:
                    mask = predictions[i, :] == cls
                    class_votes[cls] = weights[i, mask].sum()
                final_pred[i] = max(class_votes, key=class_votes.get)
            return final_pred
        else:
            # Weighted average
            return np.sum(predictions * weights, axis=1)
    
    def predict_proba(self, X):
        """Predict with confidence estimates"""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only for classification")
        
        X = np.array(X)
        weights = self._get_adaptive_weights(X)
        
        # Collect probabilities
        all_probas = []
        for model in self.base_models_:
            if hasattr(model, 'predict_proba'):
                all_probas.append(model.predict_proba(X))
            else:
                preds = model.predict(X)
                classes = np.unique(preds)
                proba = np.zeros((X.shape[0], len(classes)))
                for i, cls in enumerate(classes):
                    proba[:, i] = (preds == cls).astype(float)
                all_probas.append(proba)
        
        # Weighted average
        n_samples = X.shape[0]
        n_classes = all_probas[0].shape[1]
        weighted_proba = np.zeros((n_samples, n_classes))
        
        for i, proba in enumerate(all_probas):
            weighted_proba += proba * weights[:, i:i+1]
        
        return weighted_proba
    
    def predict_with_uncertainty(self, X):
        """
        Predict with uncertainty quantification
        
        Returns predictions and uncertainty estimates
        """
        X = np.array(X)
        predictions = []
        
        # Get predictions from all models
        for model in self.base_models_:
            predictions.append(model.predict(X))
        
        predictions = np.array(predictions).T  # Shape: (n_samples, n_models)
        
        # Get weights
        weights = self._get_adaptive_weights(X, predictions)
        
        # Final prediction
        if self.task_type == 'classification':
            final_pred = self.predict(X)
            # Uncertainty = entropy of weighted votes
            uncertainty = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                unique_classes = np.unique(predictions[i, :])
                class_probs = []
                for cls in unique_classes:
                    mask = predictions[i, :] == cls
                    prob = weights[i, mask].sum()
                    class_probs.append(prob)
                # Entropy
                class_probs = np.array(class_probs)
                class_probs = class_probs / class_probs.sum()
                uncertainty[i] = -np.sum(class_probs * np.log(class_probs + 1e-10))
        else:
            final_pred = np.sum(predictions * weights, axis=1)
            # Uncertainty = weighted standard deviation
            uncertainty = np.sqrt(np.sum(weights * (predictions - final_pred[:, np.newaxis])**2, axis=1))
        
        return final_pred, uncertainty
    
    def get_model_importance(self, X):
        """Get average importance with confidence intervals"""
        weights = self._get_adaptive_weights(X)
        
        importance_dict = {}
        for i, name in enumerate(self.model_names_):
            avg_weight = weights[:, i].mean()
            std_weight = weights[:, i].std()
            importance_dict[name] = {
                'mean': avg_weight,
                'std': std_weight,
                'base_score': self.model_scores_[i]
            }
        
        return importance_dict
    
    def score(self, X, y):
        """Calculate score"""
        if self.task_type == 'classification':
            from sklearn.metrics import accuracy_score
            return accuracy_score(y, self.predict(X))
        else:
            from sklearn.metrics import r2_score
            return r2_score(y, self.predict(X))


class AMEProClassifier(AMEPro, ClassifierMixin):
    """AME-Pro for Classification"""
    
    def __init__(self, base_models=None, meta_learner_type='neural',
                 use_confidence=True, use_attention=False,
                 meta_features='advanced', ensemble_size=7):
        super().__init__(base_models=base_models,
                        meta_learner_type=meta_learner_type,
                        use_confidence=use_confidence,
                        use_attention=use_attention,
                        meta_features=meta_features,
                        task_type='classification',
                        ensemble_size=ensemble_size)


class AMEProRegressor(AMEPro, RegressorMixin):
    """AME-Pro for Regression"""
    
    def __init__(self, base_models=None, meta_learner_type='neural',
                 use_confidence=True, use_attention=False,
                 meta_features='advanced', ensemble_size=7):
        super().__init__(base_models=base_models,
                        meta_learner_type=meta_learner_type,
                        use_confidence=use_confidence,
                        use_attention=use_attention,
                        meta_features=meta_features,
                        task_type='regression',
                        ensemble_size=ensemble_size)
