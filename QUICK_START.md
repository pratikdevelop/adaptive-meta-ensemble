# AME Pro - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Installation

```bash
# Install dependencies
pip install numpy scikit-learn matplotlib pandas seaborn --break-system-packages

# Download AME files
# (files: ame_pro.py, adaptive_meta_ensemble.py)
```

### Basic Usage

```python
from ame_pro import AMEClassifierPro, AMERegressorPro
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Standardize (recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train AME Pro
model = AMEClassifierPro(
    meta_features='advanced',      # Best feature extraction
    meta_learner_type='neural',    # Most powerful meta-learner
    auto_select_models=True,       # Automatically pick best models
    n_models_to_keep=5,           # Use top 5 base models
    uncertainty_estimation=True,   # Enable uncertainty quantification
    verbose=1                      # Show progress
)

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Get predictions with uncertainty
pred, uncertainty = model.predict_with_uncertainty(X_test)
print(f"Average uncertainty: {uncertainty.mean():.4f}")

# See which models are most important
importance = model.get_model_importance(X_test)
for model_name, weight in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name}: {weight:.4f}")
```

## 🎛️ Configuration Options

### For Best Accuracy (slower training)
```python
model = AMEClassifierPro(
    meta_features='advanced',      # Maximum feature extraction
    meta_learner_type='neural',    # Neural meta-learner
    auto_select_models=True,
    n_models_to_keep=7,           # More models = better coverage
    use_clustering=True,          # Enable feature space clustering
    uncertainty_estimation=True,
    verbose=1
)
```

### For Speed (faster training)
```python
model = AMEClassifierPro(
    meta_features='statistical',   # Basic features only
    meta_learner_type='tree',      # Fast decision tree
    auto_select_models=True,
    n_models_to_keep=3,           # Fewer models = faster
    use_clustering=False,         # Skip clustering
    uncertainty_estimation=False, # Skip uncertainty
    verbose=0
)
```

### For Custom Base Models
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

custom_models = [
    ('rf', RandomForestClassifier(n_estimators=200)),
    ('gb', GradientBoostingClassifier(n_estimators=200)),
    ('svm', SVC(probability=True, kernel='rbf'))
]

model = AMEClassifierPro(
    base_models=custom_models,     # Use your own models
    meta_features='advanced',
    meta_learner_type='neural',
    auto_select_models=False,     # Don't filter, use all
    verbose=1
)
```

## 📊 Regression Example

```python
from ame_pro import AMERegressorPro
from sklearn.datasets import load_diabetes

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
model = AMERegressorPro(
    meta_features='advanced',
    meta_learner_type='neural',
    auto_select_models=True,
    n_models_to_keep=5,
    verbose=1
)

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
r2 = model.score(X_test, y_test)
print(f"R² Score: {r2:.4f}")

# With uncertainty
pred, uncertainty = model.predict_with_uncertainty(X_test)
print(f"Predictions with ± uncertainty:")
for i in range(min(5, len(pred))):
    print(f"  {pred[i]:.2f} ± {uncertainty[i]:.2f}")
```

## 🎯 Real-World Use Cases

### 1. Credit Risk Assessment
```python
# High-stakes financial decision
model = AMEClassifierPro(
    meta_features='advanced',
    meta_learner_type='neural',
    uncertainty_estimation=True,  # Know when model is uncertain
    verbose=1
)

model.fit(X_train, y_train)
predictions, uncertainty = model.predict_with_uncertainty(X_test)

# Flag uncertain predictions for human review
high_uncertainty = uncertainty > uncertainty.mean() + uncertainty.std()
print(f"{high_uncertainty.sum()} cases need human review")
```

### 2. Medical Diagnosis
```python
# Need explainability and reliability
model = AMEClassifierPro(
    meta_features='advanced',
    meta_learner_type='forest',  # More interpretable than neural
    uncertainty_estimation=True,
    verbose=1
)

model.fit(X_train, y_train)

# Get predictions with probability
probabilities = model.predict_proba(X_test)
predictions, uncertainty = model.predict_with_uncertainty(X_test)

# Show detailed results
for i in range(5):
    print(f"Patient {i+1}:")
    print(f"  Diagnosis: {predictions[i]}")
    print(f"  Confidence: {probabilities[i].max():.2%}")
    print(f"  Uncertainty: {uncertainty[i]:.4f}")
```

### 3. Sales Forecasting
```python
# Time-sensitive business metric
model = AMERegressorPro(
    meta_features='advanced',
    meta_learner_type='neural',
    auto_select_models=True,
    n_models_to_keep=5,
    verbose=1
)

model.fit(X_train, y_train)
forecast, uncertainty = model.predict_with_uncertainty(X_future)

# Create confidence intervals
lower_bound = forecast - 1.96 * uncertainty
upper_bound = forecast + 1.96 * uncertainty

print(f"Sales Forecast: ${forecast.sum():,.0f}")
print(f"95% CI: ${lower_bound.sum():,.0f} - ${upper_bound.sum():,.0f}")
```

## 🔍 Performance Tips

### 1. Always Standardize Features
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# AME performs much better with standardized features
```

### 2. Use Cross-Validation for Model Selection
```python
from sklearn.model_selection import cross_val_score

# Let AME auto-select best models
model = AMEClassifierPro(auto_select_models=True, verbose=0)

# Or manually evaluate
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 3. Tune Number of Models
```python
# Test different numbers of base models
for n_models in [3, 5, 7]:
    model = AMEClassifierPro(n_models_to_keep=n_models, verbose=0)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{n_models} models: {score:.4f}")
```

### 4. Balance Speed vs Accuracy
```python
import time

configs = [
    ('Fast', {'meta_learner_type': 'tree', 'n_models_to_keep': 3}),
    ('Balanced', {'meta_learner_type': 'forest', 'n_models_to_keep': 5}),
    ('Accurate', {'meta_learner_type': 'neural', 'n_models_to_keep': 7})
]

for name, config in configs:
    model = AMEClassifierPro(**config, verbose=0)
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    score = model.score(X_test, y_test)
    print(f"{name:12s}: {score:.4f} accuracy in {train_time:.2f}s")
```

## ⚠️ Common Pitfalls

### ❌ Don't: Use without scaling
```python
# BAD - features have different scales
model.fit(X_train, y_train)  # Poor performance likely
```

### ✅ Do: Scale features first
```python
# GOOD
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)
```

### ❌ Don't: Use too many models on small datasets
```python
# BAD - 100 samples, 10 base models = overfitting
model = AMEClassifierPro(n_models_to_keep=10)
model.fit(X_train_small, y_train_small)
```

### ✅ Do: Adjust to dataset size
```python
# GOOD - fewer models for small datasets
n_models = max(3, min(7, len(X_train) // 100))
model = AMEClassifierPro(n_models_to_keep=n_models)
```

### ❌ Don't: Ignore uncertainty on critical decisions
```python
# BAD - high-stakes decision without uncertainty check
prediction = model.predict(X_critical)[0]
make_critical_decision(prediction)
```

### ✅ Do: Check uncertainty for important predictions
```python
# GOOD
pred, unc = model.predict_with_uncertainty(X_critical)
if unc[0] > threshold:
    send_to_human_review(X_critical[0])
else:
    make_critical_decision(pred[0])
```

## 📈 Monitoring in Production

```python
import pandas as pd
from datetime import datetime

# Track predictions over time
predictions_log = []

def predict_with_logging(X):
    pred, unc = model.predict_with_uncertainty(X)
    
    # Log prediction details
    predictions_log.append({
        'timestamp': datetime.now(),
        'prediction': pred[0],
        'uncertainty': unc[0],
        'model_weights': model.get_model_importance(X)
    })
    
    return pred[0]

# Analyze prediction quality
df = pd.DataFrame(predictions_log)
print(f"Average uncertainty: {df['uncertainty'].mean():.4f}")
print(f"High uncertainty count: {(df['uncertainty'] > 0.5).sum()}")
```

## 🆚 When to Use vs Alternatives

### Use AME Pro When:
✅ Dataset is complex/heterogeneous  
✅ Different patterns in different data regions  
✅ Need interpretability (can see which models are used)  
✅ Want uncertainty quantification  
✅ Have 1000+ training samples  

### Use Random Forest When:
- Simple, homogeneous dataset
- Need extreme speed
- Don't care about interpretability
- Have < 500 samples

### Use Gradient Boosting When:
- Tabular data with clear patterns
- Computational resources limited
- Feature interactions are key
- Standard benchmark datasets

### Use Deep Learning When:
- Images, text, audio, video
- Have 10,000+ samples
- Can use GPU
- Transfer learning applicable

## 🎓 Learning Resources

### Understanding AME:
1. Read `AME_research_paper.md` for technical details
2. Study `demo_ame.py` for examples
3. Check `comprehensive_benchmark.py` for comparisons

### Improving Results:
1. Feature engineering is crucial - spend time on it
2. Try different meta-feature types
3. Experiment with base model selection
4. Use cross-validation for hyperparameters

### Getting Help:
- Check documentation in code comments
- Review example notebooks
- Post issues on GitHub (once released)
- Join ML communities (Reddit, Discord)

## 🔥 Advanced Features

### Custom Meta-Features
```python
def extract_custom_meta_features(X):
    """Define your own meta-features"""
    features = []
    features.append(np.median(X, axis=1, keepdims=True))
    features.append(np.percentile(X, 90, axis=1, keepdims=True))
    # Add domain-specific features
    return np.hstack(features)

# Modify AME to use custom features (requires code modification)
```

### Ensemble of Ensembles
```python
# Train multiple AME models and combine
models = []
for i in range(3):
    model = AMEClassifierPro(random_state=i, verbose=0)
    model.fit(X_train, y_train)
    models.append(model)

# Average predictions
predictions = np.array([m.predict(X_test) for m in models])
final_pred = np.round(predictions.mean(axis=0))
```

### Uncertainty-Based Active Learning
```python
# Select most uncertain samples for labeling
pred, unc = model.predict_with_uncertainty(X_unlabeled)
most_uncertain_idx = np.argsort(unc)[-10:]  # Top 10 uncertain
samples_to_label = X_unlabeled[most_uncertain_idx]

# Label these samples, retrain
# (implement labeling process)
# model.fit(X_train_expanded, y_train_expanded)
```

---

**Ready to revolutionize your ML pipeline?** Start with the basic example above and experiment! 🚀

**Questions?** Check the full documentation or open an issue on GitHub.

**Contributing?** We welcome contributions! See CONTRIBUTING.md for guidelines.
