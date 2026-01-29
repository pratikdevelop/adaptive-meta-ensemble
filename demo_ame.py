"""
Demo: Adaptive Meta-Ensemble Algorithm
This script demonstrates the novel AME algorithm on both classification and regression tasks
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from adaptive_meta_ensemble import AMEClassifier, AMERegressor
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADAPTIVE META-ENSEMBLE (AME) ALGORITHM DEMONSTRATION")
print("=" * 80)
print("\nThis is a novel ML algorithm that adaptively weights ensemble members")
print("based on input feature characteristics.\n")

# ============================================================================
# DEMO 1: Classification on Synthetic Data
# ============================================================================
print("\n" + "=" * 80)
print("DEMO 1: Classification on Synthetic Data")
print("=" * 80)

# Create a complex classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, n_classes=2, random_state=42,
                          flip_y=0.1)  # Add some noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nDataset Info:")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print(f"  Features: {X_train.shape[1]}")
print(f"  Classes: {len(np.unique(y))}")

# Train our AME algorithm
print("\n" + "-" * 80)
print("Training Adaptive Meta-Ensemble...")
print("-" * 80)
ame_clf = AMEClassifier(meta_features='both', meta_learner_type='forest')
ame_clf.fit(X_train, y_train)

# Compare with standard ensemble
print("\n" + "-" * 80)
print("Training baseline (Random Forest)...")
print("-" * 80)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Evaluate
ame_pred = ame_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)

ame_acc = accuracy_score(y_test, ame_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("\n" + "=" * 80)
print("RESULTS - Classification")
print("=" * 80)
print(f"Adaptive Meta-Ensemble Accuracy: {ame_acc:.4f}")
print(f"Random Forest Accuracy:          {rf_acc:.4f}")
print(f"Improvement:                     {(ame_acc - rf_acc):.4f}")

# Show model importance
print("\n" + "-" * 80)
print("Model Importance (Average Weights on Test Set)")
print("-" * 80)
importance = ame_clf.get_model_importance(X_test)
for model_name, weight in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model_name:10s}: {weight:.4f} {'█' * int(weight * 50)}")

# ============================================================================
# DEMO 2: Classification on Real Data (Breast Cancer)
# ============================================================================
print("\n\n" + "=" * 80)
print("DEMO 2: Classification on Real Data (Breast Cancer Dataset)")
print("=" * 80)

# Load real dataset
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nDataset Info:")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print(f"  Features: {X_train.shape[1]}")

# Train models
print("\nTraining models...")
ame_clf_real = AMEClassifier(meta_features='both', meta_learner_type='tree')
ame_clf_real.fit(X_train, y_train)

gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train, y_train)

# Evaluate
ame_pred_real = ame_clf_real.predict(X_test)
gb_pred = gb_clf.predict(X_test)

ame_acc_real = accuracy_score(y_test, ame_pred_real)
gb_acc = accuracy_score(y_test, gb_pred)

print("\n" + "=" * 80)
print("RESULTS - Breast Cancer Classification")
print("=" * 80)
print(f"Adaptive Meta-Ensemble Accuracy: {ame_acc_real:.4f}")
print(f"Gradient Boosting Accuracy:      {gb_acc:.4f}")
print(f"Improvement:                     {(ame_acc_real - gb_acc):.4f}")

# ============================================================================
# DEMO 3: Regression on Synthetic Data
# ============================================================================
print("\n\n" + "=" * 80)
print("DEMO 3: Regression on Synthetic Data")
print("=" * 80)

# Create regression dataset
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, n_informative=15,
                               noise=10, random_state=42)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)

print("\nDataset Info:")
print(f"  Training samples: {X_train_r.shape[0]}")
print(f"  Test samples: {X_test_r.shape[0]}")
print(f"  Features: {X_train_r.shape[1]}")

# Train models
print("\n" + "-" * 80)
print("Training Adaptive Meta-Ensemble for Regression...")
print("-" * 80)
ame_reg = AMERegressor(meta_features='both', meta_learner_type='forest')
ame_reg.fit(X_train_r, y_train_r)

from sklearn.ensemble import RandomForestRegressor
print("\nTraining baseline (Random Forest Regressor)...")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_r, y_train_r)

# Evaluate
ame_pred_r = ame_reg.predict(X_test_r)
rf_pred_r = rf_reg.predict(X_test_r)

ame_r2 = r2_score(y_test_r, ame_pred_r)
rf_r2 = r2_score(y_test_r, rf_pred_r)

ame_mse = mean_squared_error(y_test_r, ame_pred_r)
rf_mse = mean_squared_error(y_test_r, rf_pred_r)

print("\n" + "=" * 80)
print("RESULTS - Regression")
print("=" * 80)
print(f"Adaptive Meta-Ensemble R²:  {ame_r2:.4f}")
print(f"Random Forest R²:           {rf_r2:.4f}")
print(f"Improvement:                {(ame_r2 - rf_r2):.4f}")
print(f"\nAdaptive Meta-Ensemble MSE: {ame_mse:.2f}")
print(f"Random Forest MSE:          {rf_mse:.2f}")

# Show model importance for regression
print("\n" + "-" * 80)
print("Model Importance (Average Weights on Test Set)")
print("-" * 80)
importance_reg = ame_reg.get_model_importance(X_test_r)
for model_name, weight in sorted(importance_reg.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model_name:10s}: {weight:.4f} {'█' * int(weight * 50)}")

# ============================================================================
# DEMO 4: Visualization of Adaptive Weights
# ============================================================================
print("\n\n" + "=" * 80)
print("DEMO 4: Visualizing Adaptive Weight Selection")
print("=" * 80)
print("\nGenerating visualization of how weights adapt to different inputs...")

# Create 2D dataset for visualization
from sklearn.datasets import make_moons
X_vis, y_vis = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y_vis, test_size=0.3, random_state=42)

# Train AME on 2D data
ame_vis = AMEClassifier(meta_features='statistical', meta_learner_type='tree')
ame_vis.fit(X_train_vis, y_train_vis)

# Get weights for test points
weights_vis = ame_vis._get_adaptive_weights(X_test_vis)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Adaptive Meta-Ensemble: Weight Adaptation Across Feature Space', 
             fontsize=14, fontweight='bold')

# Plot data
axes[0, 0].scatter(X_test_vis[:, 0], X_test_vis[:, 1], c=y_test_vis, 
                   cmap='RdYlBu', alpha=0.6, edgecolors='k')
axes[0, 0].set_title('Test Data Distribution')
axes[0, 0].set_xlabel('Feature 1')
axes[0, 0].set_ylabel('Feature 2')

# Plot weights for each model
for idx, model_name in enumerate(ame_vis.model_names_[:5]):  # First 5 models
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    scatter = axes[row, col].scatter(X_test_vis[:, 0], X_test_vis[:, 1], 
                                     c=weights_vis[:, idx], cmap='viridis',
                                     alpha=0.7, edgecolors='k')
    axes[row, col].set_title(f'Weight for {model_name.upper()}')
    axes[row, col].set_xlabel('Feature 1')
    axes[row, col].set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=axes[row, col])

plt.tight_layout()
plt.savefig('ame_weight_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved as 'ame_weight_visualization.png'")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("ALGORITHM SUMMARY")
print("=" * 80)
print("\n📊 What makes AME novel:")
print("   • Learns INPUT-DEPENDENT weights (not fixed weights)")
print("   • Extracts meta-features to characterize data complexity")
print("   • Each base model gets different importance for different inputs")
print("   • Meta-learner predicts optimal model weights for each prediction")

print("\n🎯 Key advantages:")
print("   • Better than single models on diverse datasets")
print("   • More flexible than standard ensemble methods")
print("   • Interpretable: can see which models are used when")
print("   • Works for both classification and regression")

print("\n🔬 Potential research directions:")
print("   • Try different meta-feature extraction methods")
print("   • Experiment with other meta-learner architectures")
print("   • Add online learning capabilities")
print("   • Optimize for specific domains (NLP, computer vision, etc.)")
print("   • Develop theoretical guarantees on performance")

print("\n" + "=" * 80)
print("Demo complete! Check out the code and visualization.")
print("=" * 80)