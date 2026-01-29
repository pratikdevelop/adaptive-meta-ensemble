"""
Comprehensive Benchmarking Suite for AME-Pro
Tests against state-of-the-art methods on multiple datasets
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_classification, make_regression,
    load_breast_cancer, load_wine, load_digits,
    fetch_california_housing, load_diabetes
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
import time
import matplotlib.pyplot as plt
import seaborn as sns
from ame_pro import AMEProClassifier, AMEProRegressor
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


class BenchmarkSuite:
    """
    Comprehensive benchmarking for AME-Pro
    
    Compares against:
    - Random Forest
    - Gradient Boosting
    - AdaBoost
    - Simple Voting/Averaging
    - Stacking
    """
    
    def __init__(self):
        self.results = {
            'classification': [],
            'regression': []
        }
        
    def get_classification_datasets(self):
        """Load classification benchmark datasets"""
        datasets = []
        
        # Breast Cancer (Binary, Real)
        X, y = load_breast_cancer(return_X_y=True)
        datasets.append(('Breast Cancer', X, y, 'binary'))
        
        # Wine (Multi-class, Real)
        X, y = load_wine(return_X_y=True)
        datasets.append(('Wine', X, y, 'multiclass'))
        
        # Digits (Multi-class, Real)
        X, y = load_digits(return_X_y=True)
        datasets.append(('Digits', X, y, 'multiclass'))
        
        # Synthetic - Easy
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=5, n_classes=2, random_state=42
        )
        datasets.append(('Synthetic Easy', X, y, 'binary'))
        
        # Synthetic - Medium
        X, y = make_classification(
            n_samples=1500, n_features=30, n_informative=20,
            n_redundant=5, n_classes=3, random_state=42, flip_y=0.1
        )
        datasets.append(('Synthetic Medium', X, y, 'multiclass'))
        
        # Synthetic - Hard
        X, y = make_classification(
            n_samples=2000, n_features=40, n_informative=25,
            n_redundant=10, n_classes=4, random_state=42,
            flip_y=0.15, class_sep=0.8
        )
        datasets.append(('Synthetic Hard', X, y, 'multiclass'))
        
        return datasets
    
    def get_regression_datasets(self):
        """Load regression benchmark datasets"""
        datasets = []
        
        # California Housing (Real)
        try:
            X, y = fetch_california_housing(return_X_y=True)
            datasets.append(('California Housing', X, y))
        except:
            print("  Skipping California Housing (download failed)")
        
        # Diabetes (Real)
        X, y = load_diabetes(return_X_y=True)
        datasets.append(('Diabetes', X, y))
        
        # Synthetic - Linear
        X, y = make_regression(
            n_samples=1000, n_features=20, n_informative=15,
            noise=10, random_state=42
        )
        datasets.append(('Synthetic Linear', X, y))
        
        # Synthetic - Nonlinear
        X, y = make_regression(
            n_samples=1500, n_features=30, n_informative=20,
            noise=20, random_state=42
        )
        # Add non-linearity
        y = y + 0.1 * (X[:, 0] ** 2) + 0.05 * (X[:, 1] * X[:, 2])
        datasets.append(('Synthetic Nonlinear', X, y))
        
        # Synthetic - Complex
        X, y = make_regression(
            n_samples=2000, n_features=40, n_informative=30,
            noise=30, random_state=42
        )
        # Add interactions
        y = y + 0.2 * (X[:, 0] * X[:, 1]) + 0.1 * np.sin(X[:, 2])
        datasets.append(('Synthetic Complex', X, y))
        
        return datasets
    
    def get_baseline_classifiers(self):
        """Get baseline classification models"""
        return {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
            'Voting (Soft)': VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
                    ('lr', LogisticRegression(max_iter=1000, random_state=42))
                ],
                voting='soft'
            ),
            'Stacking': StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
                    ('lr', LogisticRegression(max_iter=1000, random_state=42))
                ],
                final_estimator=LogisticRegression(random_state=42)
            )
        }
    
    def get_baseline_regressors(self):
        """Get baseline regression models"""
        return {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'Averaging': VotingRegressor(
                estimators=[
                    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                    ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
                    ('ridge', Ridge(random_state=42))
                ]
            ),
            'Stacking': StackingRegressor(
                estimators=[
                    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                    ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
                    ('ridge', Ridge(random_state=42))
                ],
                final_estimator=Ridge(random_state=42)
            )
        }
    
    def benchmark_classification(self):
        """Run classification benchmarks"""
        print("=" * 100)
        print("CLASSIFICATION BENCHMARKS")
        print("=" * 100)
        
        datasets = self.get_classification_datasets()
        baselines = self.get_baseline_classifiers()
        
        for dataset_name, X, y, task_type in datasets:
            print(f"\n{'='*100}")
            print(f"Dataset: {dataset_name} ({task_type})")
            print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
            print(f"{'='*100}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            results_row = {
                'Dataset': dataset_name,
                'Task': task_type,
                'Samples': X.shape[0],
                'Features': X.shape[1]
            }
            
            # Test baselines
            for name, model in baselines.items():
                print(f"\n{name}:")
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred, average='weighted')
                
                results_row[f'{name}_Acc'] = acc
                results_row[f'{name}_F1'] = f1
                results_row[f'{name}_Time'] = train_time
                
                print(f"  Accuracy: {acc:.4f}")
                print(f"  F1 Score: {f1:.4f}")
                print(f"  Time: {train_time:.2f}s")
            
            # Test AME-Pro with different configurations
            print(f"\nAME-Pro (Neural):")
            start_time = time.time()
            ame_neural = AMEProClassifier(
                meta_learner_type='neural',
                use_confidence=True,
                meta_features='advanced',
                ensemble_size=7
            )
            ame_neural.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            pred = ame_neural.predict(X_test)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average='weighted')
            
            results_row['AME-Pro_Acc'] = acc
            results_row['AME-Pro_F1'] = f1
            results_row['AME-Pro_Time'] = train_time
            
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Time: {train_time:.2f}s")
            
            # Model importance
            importance = ame_neural.get_model_importance(X_test)
            print(f"\n  Model Importance:")
            for model_name, stats in sorted(importance.items(), 
                                           key=lambda x: x[1]['mean'], 
                                           reverse=True):
                print(f"    {model_name:10s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
            self.results['classification'].append(results_row)
        
        return pd.DataFrame(self.results['classification'])
    
    def benchmark_regression(self):
        """Run regression benchmarks"""
        print("\n\n" + "=" * 100)
        print("REGRESSION BENCHMARKS")
        print("=" * 100)
        
        datasets = self.get_regression_datasets()
        baselines = self.get_baseline_regressors()
        
        for dataset_name, X, y in datasets:
            print(f"\n{'='*100}")
            print(f"Dataset: {dataset_name}")
            print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
            print(f"{'='*100}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            results_row = {
                'Dataset': dataset_name,
                'Samples': X.shape[0],
                'Features': X.shape[1]
            }
            
            # Test baselines
            for name, model in baselines.items():
                print(f"\n{name}:")
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                pred = model.predict(X_test)
                r2 = r2_score(y_test, pred)
                mse = mean_squared_error(y_test, pred)
                mae = mean_absolute_error(y_test, pred)
                
                results_row[f'{name}_R2'] = r2
                results_row[f'{name}_MSE'] = mse
                results_row[f'{name}_MAE'] = mae
                results_row[f'{name}_Time'] = train_time
                
                print(f"  R²: {r2:.4f}")
                print(f"  MSE: {mse:.2f}")
                print(f"  MAE: {mae:.2f}")
                print(f"  Time: {train_time:.2f}s")
            
            # Test AME-Pro
            print(f"\nAME-Pro (Neural):")
            start_time = time.time()
            ame_neural = AMEProRegressor(
                meta_learner_type='neural',
                use_confidence=True,
                meta_features='advanced',
                ensemble_size=7
            )
            ame_neural.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            pred = ame_neural.predict(X_test)
            r2 = r2_score(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            
            results_row['AME-Pro_R2'] = r2
            results_row['AME-Pro_MSE'] = mse
            results_row['AME-Pro_MAE'] = mae
            results_row['AME-Pro_Time'] = train_time
            
            print(f"  R²: {r2:.4f}")
            print(f"  MSE: {mse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  Time: {train_time:.2f}s")
            
            # Model importance
            importance = ame_neural.get_model_importance(X_test)
            print(f"\n  Model Importance:")
            for model_name, stats in sorted(importance.items(),
                                           key=lambda x: x[1]['mean'],
                                           reverse=True):
                print(f"    {model_name:10s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
            self.results['regression'].append(results_row)
        
        return pd.DataFrame(self.results['regression'])
    
    def create_comparison_plots(self, clf_results, reg_results):
        """Create visualization comparing all methods"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('AME-Pro vs Baselines: Comprehensive Benchmark', 
                     fontsize=16, fontweight='bold')
        
        # Classification Accuracy
        ax = axes[0, 0]
        methods = ['Random Forest', 'Gradient Boosting', 'AdaBoost', 
                   'Voting (Soft)', 'Stacking', 'AME-Pro']
        acc_cols = [f'{m}_Acc' for m in methods]
        
        clf_acc_data = []
        for col in acc_cols:
            if col in clf_results.columns:
                clf_acc_data.append(clf_results[col].values)
        
        bp = ax.boxplot(clf_acc_data, labels=methods, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Classification Accuracy Comparison', fontsize=13, fontweight='bold')
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Classification F1 Score
        ax = axes[0, 1]
        f1_cols = [f'{m}_F1' for m in methods]
        
        clf_f1_data = []
        for col in f1_cols:
            if col in clf_results.columns:
                clf_f1_data.append(clf_results[col].values)
        
        bp = ax.boxplot(clf_f1_data, labels=methods, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Classification F1 Score Comparison', fontsize=13, fontweight='bold')
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Regression R²
        ax = axes[1, 0]
        reg_methods = ['Random Forest', 'Gradient Boosting', 'AdaBoost',
                      'Averaging', 'Stacking', 'AME-Pro']
        r2_cols = [f'{m}_R2' for m in reg_methods]
        
        reg_r2_data = []
        for col in r2_cols:
            if col in reg_results.columns:
                reg_r2_data.append(reg_results[col].values)
        
        bp = ax.boxplot(reg_r2_data, labels=reg_methods, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Regression R² Comparison', fontsize=13, fontweight='bold')
        ax.set_xticklabels(reg_methods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Training Time Comparison
        ax = axes[1, 1]
        time_data = []
        labels = []
        
        # Classification times
        for m in methods:
            col = f'{m}_Time'
            if col in clf_results.columns:
                time_data.append(clf_results[col].mean())
                labels.append(f'{m}\n(Clf)')
        
        # Regression times
        for m in reg_methods:
            col = f'{m}_Time'
            if col in reg_results.columns:
                time_data.append(reg_results[col].mean())
                labels.append(f'{m}\n(Reg)')
        
        colors = ['lightblue'] * len(methods) + ['lightcoral'] * len(reg_methods)
        ax.bar(range(len(time_data)), time_data, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Training Time (seconds)', fontsize=12)
        ax.set_title('Average Training Time Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('benchmark_comparison.png', dpi=150, bbox_inches='tight')
        print("\n✓ Comparison plots saved as 'benchmark_comparison.png'")


def main():
    """Run complete benchmark suite"""
    print("╔" + "═" * 98 + "╗")
    print("║" + " " * 25 + "AME-PRO COMPREHENSIVE BENCHMARK SUITE" + " " * 35 + "║")
    print("╚" + "═" * 98 + "╝")
    
    benchmark = BenchmarkSuite()
    
    # Run classification benchmarks
    clf_results = benchmark.benchmark_classification()
    
    # Run regression benchmarks
    reg_results = benchmark.benchmark_regression()
    
    # Create summary
    print("\n\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    print("\n📊 Classification Results Summary:")
    print(clf_results.to_string(index=False))
    
    print("\n\n📊 Regression Results Summary:")
    print(reg_results.to_string(index=False))
    
    # Save results
    clf_results.to_csv('classification_benchmarks.csv', index=False)
    reg_results.to_csv('regression_benchmarks.csv', index=False)
    print("\n✓ Results saved to CSV files")
    
    # Create visualizations
    benchmark.create_comparison_plots(clf_results, reg_results)
    
    # Calculate win rates
    print("\n\n" + "=" * 100)
    print("WIN RATES (AME-Pro vs Baselines)")
    print("=" * 100)
    
    clf_methods = ['Random Forest', 'Gradient Boosting', 'AdaBoost', 
                   'Voting (Soft)', 'Stacking']
    
    print("\n🏆 Classification (Accuracy):")
    for method in clf_methods:
        col = f'{method}_Acc'
        if col in clf_results.columns:
            wins = (clf_results['AME-Pro_Acc'] > clf_results[col]).sum()
            total = len(clf_results)
            win_rate = wins / total * 100
            print(f"  vs {method:20s}: {wins}/{total} wins ({win_rate:.1f}%)")
    
    reg_methods = ['Random Forest', 'Gradient Boosting', 'AdaBoost',
                  'Averaging', 'Stacking']
    
    print("\n🏆 Regression (R²):")
    for method in reg_methods:
        col = f'{method}_R2'
        if col in reg_results.columns:
            wins = (reg_results['AME-Pro_R2'] > reg_results[col]).sum()
            total = len(reg_results)
            win_rate = wins / total * 100
            print(f"  vs {method:20s}: {wins}/{total} wins ({win_rate:.1f}%)")
    
    print("\n\n" + "=" * 100)
    print("Benchmark complete! Check the output files for detailed results.")
    print("=" * 100)


if __name__ == '__main__':
    main()
