# Contributing to Adaptive Meta-Ensemble (AME)

Thank you for your interest in contributing to AME! 🎉

We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, code contributions, and more.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Guidelines](#coding-guidelines)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## 📜 Code of Conduct

Be respectful, inclusive, and considerate in all interactions. We want AME to be a welcoming community for everyone.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/adaptive-meta-ensemble.git
   cd adaptive-meta-ensemble
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🤝 How to Contribute

### Reporting Bugs

Found a bug? Please create an issue with:

- **Clear title** describing the problem
- **Steps to reproduce** the bug
- **Expected behavior** vs actual behavior
- **Environment details** (Python version, OS, scikit-learn version)
- **Code snippet** to reproduce (if applicable)

**Bug Report Template:**
```markdown
## Bug Description
[Clear description of the bug]

## To Reproduce
```python
# Minimal code to reproduce the issue
from ame_pro import AMEClassifierPro
# ... your code
```

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Environment
- Python version: 
- scikit-learn version:
- NumPy version:
- OS:

## Additional Context
[Any other relevant information]
```

### Suggesting Features

Have an idea? Open an issue with:

- **Clear description** of the feature
- **Use case** - why would this be useful?
- **Proposed implementation** (if you have ideas)
- **Examples** of how it would be used

### Improving Documentation

Documentation improvements are always welcome:
- Fix typos or clarify explanations
- Add examples and use cases
- Improve docstrings
- Create tutorials or guides

## 💻 Development Setup

### 1. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### 2. Install in Development Mode

```bash
pip install -e .
```

This allows you to edit the code and test changes immediately.

### 3. Verify Installation

```bash
python -c "from ame_pro import AMEClassifierPro; print('Success!')"
```

## 📝 Coding Guidelines

### Python Style

We follow **PEP 8** with these specifics:

- **Line length**: 100 characters max
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Organized (stdlib, third-party, local)
- **Naming**:
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_CASE`

### Type Hints

Use type hints for function signatures:

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """Predict class labels."""
    # implementation
```

### Docstrings

Use NumPy-style docstrings:

```python
def predict_with_uncertainty(self, X):
    """
    Predict with uncertainty quantification.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input samples to predict.
    
    Returns
    -------
    predictions : ndarray of shape (n_samples,)
        Predicted class labels or values.
    uncertainties : ndarray of shape (n_samples,)
        Uncertainty estimate for each prediction.
    
    Examples
    --------
    >>> from ame_pro import AMEClassifierPro
    >>> model = AMEClassifierPro()
    >>> model.fit(X_train, y_train)
    >>> pred, unc = model.predict_with_uncertainty(X_test)
    >>> print(f"Prediction: {pred[0]}, Uncertainty: {unc[0]:.3f}")
    """
    # implementation
```

### Code Formatting

Before committing, format your code:

```bash
# Auto-format with black
black ame_pro.py

# Check style with flake8
flake8 ame_pro.py --max-line-length=100

# Type checking with mypy
mypy ame_pro.py
```

## 🧪 Testing

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Aim for 80%+ code coverage

**Example test:**

```python
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ame_pro import AMEClassifierPro

def test_ame_classifier_basic():
    """Test basic AME classifier functionality."""
    # Create dataset
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Train model
    model = AMEClassifierPro(verbose=0)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Assertions
    assert len(predictions) == len(X_test)
    assert predictions.shape == (len(X_test),)
    
    # Check accuracy is reasonable
    accuracy = model.score(X_test, y_test)
    assert 0.5 <= accuracy <= 1.0

def test_uncertainty_estimation():
    """Test uncertainty quantification."""
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    model = AMEClassifierPro(uncertainty_estimation=True, verbose=0)
    model.fit(X_train, y_train)
    
    pred, unc = model.predict_with_uncertainty(X_test)
    
    assert len(pred) == len(X_test)
    assert len(unc) == len(X_test)
    assert np.all(unc >= 0)  # Uncertainty should be non-negative
    assert np.all(unc <= 1)  # Uncertainty should be normalized
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ame_pro --cov-report=html

# Run specific test
pytest tests/test_ame_classifier.py::test_ame_classifier_basic

# Run with verbose output
pytest -v
```

## 🔄 Pull Request Process

### 1. Before Submitting

- ✅ Code follows style guidelines
- ✅ All tests pass
- ✅ New features have tests
- ✅ Documentation is updated
- ✅ Commits have clear messages

### 2. Commit Messages

Follow conventional commits format:

```
type(scope): brief description

More detailed explanation if needed.

Closes #issue_number
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, semicolons, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(meta-learner): add LSTM-based meta-learner option"
git commit -m "fix(clustering): handle edge case with single cluster"
git commit -m "docs(readme): add installation instructions for conda"
git commit -m "test(classifier): add test for multi-class classification"
```

### 3. Submit Pull Request

1. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference any related issues
   - Screenshots (if applicable)

**PR Template:**
```markdown
## Description
[Brief description of changes]

## Motivation
[Why is this change needed?]

## Changes Made
- [List of changes]
- [...]

## Testing
- [ ] Added tests for new functionality
- [ ] All tests pass locally
- [ ] Updated documentation

## Related Issues
Closes #[issue_number]

## Screenshots (if applicable)
[Add screenshots if relevant]
```

### 4. Review Process

- Maintainer will review within 3-5 days
- Address any requested changes
- Once approved, your PR will be merged!

## 🎯 Areas We Need Help With

### High Priority
- [ ] More comprehensive test coverage
- [ ] Additional example notebooks
- [ ] Performance optimizations
- [ ] Documentation improvements
- [ ] Bug fixes

### Medium Priority
- [ ] Support for multi-output tasks
- [ ] Additional meta-feature extraction methods
- [ ] Visualization utilities
- [ ] CLI tool development

### Advanced Features
- [ ] GPU acceleration
- [ ] Distributed training support
- [ ] Integration with other frameworks (PyTorch, TensorFlow)
- [ ] Online/streaming learning

## 💬 Community

### Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/pratikdevelop/adaptive-meta-ensemble/discussions)
- **Bugs**: Create an [Issue](https://github.com/pratikdevelop/adaptive-meta-ensemble/issues)
- **Chat**: [Join our Discord](#) (if you create one)

### Staying Updated

- ⭐ Star the repository
- 👀 Watch for updates
- 🐦 Follow on Twitter (if applicable)

## 🙏 Recognition

All contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Credited in documentation
- Given attribution in academic citations (for significant contributions)

## 📚 Resources

- [scikit-learn Contribution Guide](https://scikit-learn.org/stable/developers/contributing.html)
- [NumPy Documentation Guide](https://numpy.org/doc/stable/dev/howto-docs.html)
- [PEP 8 Style Guide](https://pep8.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)

## ❓ Questions?

Don't hesitate to ask! Open an issue or discussion, and we'll be happy to help.

---

**Thank you for contributing to AME! Every contribution, no matter how small, helps make this project better.** 🚀

---

By contributing, you agree that your contributions will be licensed under the MIT License.