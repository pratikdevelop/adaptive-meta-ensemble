"""
Setup configuration for Adaptive Meta-Ensemble (AME)
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="adaptive-meta-ensemble",
    version="1.0.0",
    author="Pratik Kumar",
    author_email="pratik.raut9115@gmail.com",  # TODO: Update with your email
    description="A novel machine learning algorithm with adaptive ensemble weighting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pratikdevelop/adaptive-meta-ensemble",
    project_urls={
        "Bug Tracker": "https://github.com/pratikdevelop/adaptive-meta-ensemble/issues",
        "Documentation": "https://github.com/pratikdevelop/adaptive-meta-ensemble#readme",
        "Source Code": "https://github.com/pratikdevelop/adaptive-meta-ensemble",
    },
    py_modules=["ame_pro", "adaptive_meta_ensemble"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pandas>=1.3.0",
        ],
        "all": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pandas>=1.3.0",
            "pytest>=7.0.0",
        ],
    },
    keywords=[
        "machine-learning",
        "ensemble-learning",
        "meta-learning",
        "adaptive-algorithms",
        "scikit-learn",
        "classification",
        "regression",
        "automl",
    ],
    zip_safe=False,
)