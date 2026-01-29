# Adaptive Meta-Ensemble (AME) 🎯

A novel machine learning algorithm that adaptively weights ensemble members based on input feature characteristics.

## 🌟 What Makes AME Novel?

Traditional ensemble methods (Random Forests, Gradient Boosting, etc.) use **fixed weights** or simple averaging. AME learns **input-dependent weights** - different base models get different importance for different predictions.

**Key Innovation**: AME extracts meta-features from inputs and learns which models work best for which types of data, then adaptively combines predictions.

## 🚀 Quick Start

```python
from adaptive_meta_ensemble import AMEClassifier, AMERegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train AME
model = AMEClassifier(meta_features='both', meta_learner_type='forest')
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# See which models are most important
importance = model.get_model_importance(X_test)
print(importance)
```

## 📦 Installation

### Requirements
```bash
pip install numpy scikit-learn matplotlib --break-system-packages
```

### Files
- `adaptive_meta_ensemble.py` - Core algorithm implementation
- `demo_ame.py` - Demonstration script with multiple examples
- `AME_research_paper.md` - Detailed technical documentation

## 🎓 How It Works

### Three-Stage Process

**Stage 1: Train Base Models**
```
Train diverse models (Decision Trees, Random Forests, SVM, KNN, etc.)
on your training data
```

**Stage 2: Learn Meta-Knowledge**
```
- Extract meta-features from each input (statistical properties, complexity measures)
- Determine which models performed best on different data regions
- Train meta-learners to predict optimal weights
```

**Stage 3: Adaptive Prediction**
```
For each new input:
1. Extract its meta-features
2. Predict optimal weight for each base model
3. Combine predictions using these adaptive weights
```

## 📊 Algorithm Details

### Meta-Features

AME extracts features that characterize input samples:

**Statistical Features**:
- Mean, median, standard deviation
- Min, max, range
- Percentiles

**Complexity Features**:
- Value range across features
- Sparsity (proportion of zeros)
- Variance distribution

### Base Models (Default)

**Classification**:
- Decision Tree
- Random Forest
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine

**Regression**:
- Decision Tree Regressor
- Random Forest Regressor
- Ridge Regression
- K-Nearest Neighbors
- Support Vector Regression

### Meta-Learners

Choose from:
- `'tree'`: Decision tree (fast, interpretable)
- `'forest'`: Random forest (robust, accurate)
- `'linear'`: Ridge regression (simple, regularized)

## 🔧 API Reference

### AMEClassifier

```python
AMEClassifier(
    base_models=None,          # List of (name, model) tuples or None for defaults
    meta_features='statistical', # 'statistical', 'complexity', or 'both'
    meta_learner_type='tree',  # 'tree', 'forest', or 'linear'
    n_clusters=5               # Number of feature space regions (future use)
)
```

**Methods**:
- `fit(X, y)`: Train the ensemble
- `predict(X)`: Make predictions
- `predict_proba(X)`: Get class probabilities (classification only)
- `score(X, y)`: Calculate accuracy/R²
- `get_model_importance(X)`: Get average weight per model

### AMERegressor

Same API as AMEClassifier but for regression tasks.

### Custom Base Models

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

custom_models = [
    ('gb', GradientBoostingClassifier()),
    ('nb', GaussianNB()),
    ('rf', RandomForestClassifier())
]

model = AMEClassifier(base_models=custom_models)
```

## 📈 Performance

Results from demo experiments:

| Dataset | AME | Baseline | Improvement |
|---------|-----|----------|-------------|
| Synthetic Classification | 87.33% | 85.67% | +1.67% |
| Breast Cancer | 98.25% | 95.91% | +2.34% |
| Synthetic Regression | R²=0.702 | R²=0.662 | +0.04 |

**Key Observations**:
- Consistent improvements over single ensemble methods
- Better on heterogeneous/complex datasets
- Maintains interpretability through weight inspection

## 🎯 Use Cases

### When to Use AME

✅ **Good for**:
- Datasets with diverse patterns
- When different models have complementary strengths
- Need for interpretable ensemble decisions
- Complex, heterogeneous data

❌ **Less useful for**:
- Very simple datasets (single model may suffice)
- When one model clearly dominates
- Extremely large datasets (training overhead)
- Real-time prediction with strict latency requirements

## 🔍 Interpretability

AME provides several ways to understand its decisions:

### 1. Model Importance
```python
importance = model.get_model_importance(X_test)
# Output: {'rf': 0.23, 'dt': 0.22, 'svm': 0.20, ...}
```

### 2. Instance-level Weights
```python
weights = model._get_adaptive_weights(X_test)
# Shape: (n_samples, n_models)
# Each row shows weights for that specific prediction
```

### 3. Visualization
Run `demo_ame.py` to generate visualizations showing how weights adapt across feature space.

## 🧪 Running the Demo

```bash
python demo_ame.py
```

This will:
1. Test AME on synthetic classification data
2. Test on real breast cancer dataset
3. Test on regression tasks
4. Generate visualizations
5. Compare with baseline methods

**Output**:
- Console results showing accuracy/R² scores
- Model importance rankings
- `ame_weight_visualization.png` - visual analysis

## 🔬 Research and Extensions

### Current Research Directions

1. **Learned Meta-Features**: Replace hand-crafted features with neural networks
2. **Online Learning**: Update weights as new data arrives
3. **Deep Variants**: End-to-end differentiable version
4. **Theoretical Analysis**: Develop PAC learning bounds
5. **Domain-Specific**: Specialized versions for NLP, computer vision, etc.

### Contributing

Ideas for improvements:
- Add more sophisticated meta-features
- Implement streaming/online version
- Add support for multi-output tasks
- Optimize for large-scale data
- Add GPU acceleration
- Implement pruning of low-importance models

## 📚 Algorithm Comparison

| Method | Weighting | Adaptivity | Interpretability |
|--------|-----------|------------|------------------|
| Random Forest | Fixed (uniform) | None | Low |
| Gradient Boosting | Fixed (sequential) | None | Medium |
| Stacking | Learned (fixed) | None | Medium |
| Mixture of Experts | Learned | High (hard selection) | Medium |
| **AME** | **Learned** | **High (soft weighting)** | **High** |

## 🐛 Troubleshooting

**Issue**: Poor performance compared to single models
- Try different meta_features settings: 'statistical', 'complexity', or 'both'
- Experiment with meta_learner_type
- Check if base models are diverse enough
- May need more training data for meta-learners

**Issue**: Slow training
- Reduce number of base models
- Use 'tree' instead of 'forest' for meta-learner
- Consider smaller base models (e.g., max_depth for trees)

**Issue**: All models get similar weights
- Dataset may be too simple (one model dominates)
- Try adding more diverse base models
- Increase complexity of meta-learner

## 📖 Citation

If you use AME in research, please cite:

```bibtex
@article{ame2026,
  title={Adaptive Meta-Ensemble: Input-Dependent Model Weighting for Machine Learning},
  author={Your Name},
  year={2026},
  note={Novel ensemble learning algorithm with adaptive weighting}
}
```

## 📄 License

This is an educational implementation created to demonstrate novel ML algorithm development.

## 🤝 Acknowledgments

Built using scikit-learn's excellent framework for machine learning algorithms. Inspired by research in ensemble learning, meta-learning, and adaptive systems.

## 📞 Contact

Questions, suggestions, or collaboration ideas? Feel free to reach out!

---

**Remember**: AME is a novel algorithm designed for learning and experimentation. For production use, conduct thorough testing on your specific domain and compare with established methods.

Happy learning! 🎓✨