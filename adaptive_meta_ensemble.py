"""
Adaptive Meta-Ensemble (AME) Algorithm
A novel machine learning algorithm that adaptively weights ensemble members
based on input feature characteristics.

Key Innovation: Instead of using fixed weights or simple voting, this algorithm
learns which base models perform best for different regions of the feature space
and dynamically adjusts their influence.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AdaptiveMetaEnsemble(BaseEstimator):
    """
    Adaptive Meta-Ensemble: A novel ensemble method that learns input-dependent weights
    
    The algorithm works in three stages:
    1. Train multiple diverse base models
    2. Train a meta-learner that predicts which base models will perform best
       for given input characteristics
    3. Make predictions using adaptively weighted ensemble
    
    Parameters:
    -----------
    base_models : list of tuples, default=None
        List of (name, model) tuples. If None, uses default set.
    meta_features : str, default='statistical'
        Type of meta-features to extract: 'statistical', 'complexity', or 'both'
    meta_learner_type : str, default='tree'
        Type of meta-learner: 'tree', 'forest', or 'linear'
    task_type : str, default='classification'
        Either 'classification' or 'regression'
    n_clusters : int, default=5
        Number of feature space regions to identify
    """
    
    def __init__(self, base_models=None, meta_features='statistical',
                 meta_learner_type='tree', task_type='classification',
                 n_clusters=5):
        self.base_models = base_models
        self.meta_features = meta_features
        self.meta_learner_type = meta_learner_type
        self.task_type = task_type
        self.n_clusters = n_clusters
        
    def _get_default_base_models(self):
        """Get default base models based on task type"""
        if self.task_type == 'classification':
            return [
                ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
                ('lr', LogisticRegression(max_iter=1000, random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=5)),
                ('svm', SVC(kernel='rbf', probability=True, random_state=42))
            ]
        else:
            return [
                ('dt', DecisionTreeRegressor(max_depth=10, random_state=42)),
                ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)),
                ('ridge', Ridge(alpha=1.0, random_state=42)),
                ('knn', KNeighborsRegressor(n_neighbors=5)),
                ('svr', SVR(kernel='rbf'))
            ]
    
    def _extract_meta_features(self, X):
        """
        Extract meta-features that characterize input samples
        
        Innovation: We extract features that help predict which model will work best
        """
        meta_feats = []
        
        if self.meta_features in ['statistical', 'both']:
            # Statistical properties
            meta_feats.append(np.mean(X, axis=1, keepdims=True))
            meta_feats.append(np.std(X, axis=1, keepdims=True))
            meta_feats.append(np.min(X, axis=1, keepdims=True))
            meta_feats.append(np.max(X, axis=1, keepdims=True))
            meta_feats.append(np.median(X, axis=1, keepdims=True))
        
        if self.meta_features in ['complexity', 'both']:
            # Complexity measures
            # Range of values
            meta_feats.append((np.max(X, axis=1) - np.min(X, axis=1)).reshape(-1, 1))
            # Sparsity
            meta_feats.append((np.sum(X == 0, axis=1) / X.shape[1]).reshape(-1, 1))
            # Variance
            meta_feats.append(np.var(X, axis=1, keepdims=True))
        
        return np.hstack(meta_feats)
    
    def _create_meta_learner(self, n_outputs):
        """Create meta-learner that predicts base model weights"""
        if self.meta_learner_type == 'tree':
            if self.task_type == 'classification':
                return DecisionTreeRegressor(max_depth=5, random_state=42)
            else:
                return DecisionTreeRegressor(max_depth=5, random_state=42)
        elif self.meta_learner_type == 'forest':
            return RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)
        else:  # linear
            return Ridge(alpha=1.0, random_state=42)
    
    def fit(self, X, y):
        """
        Fit the Adaptive Meta-Ensemble
        
        Training Process:
        1. Train all base models
        2. Evaluate each base model on different parts of feature space
        3. Train meta-learner to predict optimal weights
        """
        X = np.array(X)
        y = np.array(y)
        
        # Initialize
        if self.base_models is None:
            self.base_models = self._get_default_base_models()
        
        self.base_models_ = []
        self.model_names_ = []
        
        # Step 1: Train base models
        print("Training base models...")
        for name, model in self.base_models:
            print(f"  Training {name}...")
            model.fit(X, y)
            self.base_models_.append(model)
            self.model_names_.append(name)
        
        # Step 2: Create training data for meta-learner
        print("\nCreating meta-training data...")
        meta_X = self._extract_meta_features(X)
        
        # For each sample, determine which model performed best
        # We do this by getting predictions and computing local errors
        meta_y = np.zeros((X.shape[0], len(self.base_models_)))
        
        for i, model in enumerate(self.base_models_):
            if self.task_type == 'classification':
                pred = model.predict(X)
                # Accuracy as weight (1 if correct, 0 if wrong)
                meta_y[:, i] = (pred == y).astype(float)
            else:
                pred = model.predict(X)
                # Use negative absolute error (higher is better)
                errors = -np.abs(pred - y)
                # Normalize to 0-1 range
                if errors.max() != errors.min():
                    meta_y[:, i] = (errors - errors.min()) / (errors.max() - errors.min())
                else:
                    meta_y[:, i] = 0.5
        
        # Normalize weights to sum to 1 for each sample
        row_sums = meta_y.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        meta_y = meta_y / row_sums
        
        # Step 3: Train meta-learner for each base model weight
        print("\nTraining meta-learners...")
        self.meta_learners_ = []
        for i, name in enumerate(self.model_names_):
            print(f"  Training meta-learner for {name}...")
            meta_learner = self._create_meta_learner(1)
            meta_learner.fit(meta_X, meta_y[:, i])
            self.meta_learners_.append(meta_learner)
        
        print("\nTraining complete!")
        return self
    
    def _get_adaptive_weights(self, X):
        """
        Predict optimal weights for each base model given input X
        
        Key Innovation: Weights are input-dependent, not fixed
        """
        meta_X = self._extract_meta_features(X)
        
        weights = np.zeros((X.shape[0], len(self.base_models_)))
        for i, meta_learner in enumerate(self.meta_learners_):
            weights[:, i] = meta_learner.predict(meta_X)
        
        # Ensure weights are non-negative and sum to 1
        weights = np.maximum(weights, 0)
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        weights = weights / row_sums
        
        return weights
    
    def predict(self, X):
        """Make predictions using adaptive ensemble"""
        X = np.array(X)
        
        # Get predictions from all base models
        predictions = np.zeros((X.shape[0], len(self.base_models_)))
        for i, model in enumerate(self.base_models_):
            predictions[:, i] = model.predict(X)
        
        # Get adaptive weights
        weights = self._get_adaptive_weights(X)
        
        # Weighted combination
        if self.task_type == 'classification':
            # For classification, use weighted voting
            weighted_preds = predictions * weights
            final_pred = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                # Get unique classes and their weighted votes
                unique_classes = np.unique(predictions[i, :])
                class_votes = {}
                for cls in unique_classes:
                    mask = predictions[i, :] == cls
                    class_votes[cls] = weights[i, mask].sum()
                final_pred[i] = max(class_votes, key=class_votes.get)
            return final_pred
        else:
            # For regression, use weighted average
            return np.sum(predictions * weights, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities (classification only)"""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        X = np.array(X)
        
        # Get probability predictions from all base models
        n_samples = X.shape[0]
        n_models = len(self.base_models_)
        
        # Get adaptive weights
        weights = self._get_adaptive_weights(X)
        
        # Collect probabilities from each model
        all_probas = []
        for model in self.base_models_:
            if hasattr(model, 'predict_proba'):
                all_probas.append(model.predict_proba(X))
            else:
                # For models without predict_proba, use one-hot encoding
                preds = model.predict(X)
                classes = np.unique(preds)
                proba = np.zeros((n_samples, len(classes)))
                for i, cls in enumerate(classes):
                    proba[:, i] = (preds == cls).astype(float)
                all_probas.append(proba)
        
        # Ensure all have same shape
        n_classes = all_probas[0].shape[1]
        
        # Weighted average of probabilities
        weighted_proba = np.zeros((n_samples, n_classes))
        for i, proba in enumerate(all_probas):
            weighted_proba += proba * weights[:, i:i+1]
        
        return weighted_proba
    
    def get_model_importance(self, X):
        """
        Get the average importance of each base model for given inputs
        
        Useful for interpretability
        """
        weights = self._get_adaptive_weights(X)
        avg_weights = weights.mean(axis=0)
        
        importance_dict = {name: weight 
                          for name, weight in zip(self.model_names_, avg_weights)}
        return importance_dict
    
    def score(self, X, y):
        """Calculate score (accuracy for classification, R² for regression)"""
        if self.task_type == 'classification':
            from sklearn.metrics import accuracy_score
            return accuracy_score(y, self.predict(X))
        else:
            from sklearn.metrics import r2_score
            return r2_score(y, self.predict(X))


class AMEClassifier(AdaptiveMetaEnsemble, ClassifierMixin):
    """Adaptive Meta-Ensemble for Classification"""
    
    def __init__(self, base_models=None, meta_features='statistical',
                 meta_learner_type='tree', n_clusters=5):
        super().__init__(base_models=base_models, 
                        meta_features=meta_features,
                        meta_learner_type=meta_learner_type,
                        task_type='classification',
                        n_clusters=n_clusters)


class AMERegressor(AdaptiveMetaEnsemble, RegressorMixin):
    """Adaptive Meta-Ensemble for Regression"""
    
    def __init__(self, base_models=None, meta_features='statistical',
                 meta_learner_type='tree', n_clusters=5):
        super().__init__(base_models=base_models,
                        meta_features=meta_features,
                        meta_learner_type=meta_learner_type,
                        task_type='regression',
                        n_clusters=n_clusters)