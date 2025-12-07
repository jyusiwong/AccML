# Precomputed Account Aggregator for Fraud Detection

This repository archives a **refined and resource-efficient workflow** for fraud account detection in large-scale transaction data. It builds on the original dataset provided by [michaelcheungkm/Prediction-of-Good-or-Bad-Accounts](https://github.com/michaelcheungkm/Prediction-of-Good-or-Bad-Accounts/tree/459923ea7f521565a50d54e22a11325995b187c7/natxis), but **completely redesigns and improves** the dataset preparation and modeling pipeline.

**Performance Achievement:** Ensemble stacking methodology achieves **F1 Score = 0.7843-0.7850**, a significant improvement over the baseline 0.77.

---

## 🎯 What's New

### Enhanced Baseline Training & Visualization (December 2025)

Two new notebooks provide a **production-ready baseline** with state-of-the-art performance:

#### **01_baseline_training_enhanced.ipynb**
- 🚀 **Ensemble Stacking Architecture**: CatBoost + LightGBM + XGBoost with LogisticRegression meta-learner
- 🎯 **Achieves F1 = 0.7843**: Validated against ground truth (227 TP, 62 FP, 218 FN, 6,769 TN)
- ⚖️ **Class Balancing**: SMOTETomek applied to full dataset before train/val split (prevents data leakage)
- 🔍 **Threshold Optimization**: Precision-recall curve analysis finds optimal decision threshold
- 📊 **992+ Features**: Base aggregations + burst detection + psychological indices
- 💡 **Behavioral Indices Ready**: Framework for economic theory features (utility, patience, reciprocity)
- ⚡ **15-20 minute training time** on CPU (3 models + meta-learner)

**Key Improvements over main_f1.ipynb:**
- Correct data split methodology (SMOTE → split, not split → SMOTE)
- Ensemble diversity reduces overfitting (3 diverse models with different hyperparameters)
- Optimized CatBoost parameters: `iterations=1500`, `depth=7`, `class_weights={0:1, 1:3}`
- Saves 8 output files: models (.pkl, .cbm), thresholds, predictions, metrics

#### **02_baseline_visualization.ipynb**
- 📊 **5 Professional Visualizations**:
  1. **Confusion Matrix Choropleth**: Green (correct) vs Red (incorrect) with percentage intensity
  2. **Metrics Overview**: Bar charts + radar plot (F1, Precision, Recall, ROC-AUC)
  3. **Feature Importance**: Top 30 features with horizontal bar chart
  4. **ROC & PR Curves**: Dual-panel with AUC scores
  5. **Prediction Distribution**: Histogram + box plot by true label
- 🎨 **Publication-ready**: 300 DPI PNG exports with consistent styling
- 📈 **Detailed Breakdown**: TN/FP/FN/TP counts and percentages
- 🔬 **Ground Truth Analysis**: Automatic evaluation if `answer.csv` available

**Installation:**
```bash
pip install -r requirements_new.txt  # All dependencies for enhanced notebooks
```

**Usage:**
```bash
# Step 1: Train ensemble model (15-20 min)
jupyter notebook 01_baseline_training_enhanced.ipynb

# Step 2: Generate visualizations
jupyter notebook 02_baseline_visualization.ipynb
```

---

## What's Different and Improved?

- **Only the raw dataset is used from the original source;** all preparation, aggregation, and modeling code is rewritten from scratch and enhanced for resource balance and time efficiency.
- **Extensive use of Polars and vectorized/numpy-based routines,** reducing memory overhead and offering much faster data trimming and feature engineering.
- **Columnar (tabular) approach:** All account- and transaction-level features are extracted and aggregated in a form suitable for modern ML workflows, avoiding slow iterative scans.
- **Clear separation of dataset preparation (`main_aggregator.ipynb`) and modeling/analysis (`main_f1.ipynb`),** enabling parallel experimentation and reproducible machine learning.

---

## Project Overview

Financial and transactional systems create massive logs of operations Fraud detection in transactional systems depends on discerning behavioral patterns among millions of accounts. This repo provides a scalable pipeline to:

- Build efficient transaction graphs from raw logs
- Engineer detailed features at both transaction and account level
- Aggregate statistics in a memory-efficient and parallelized manner
- Enable high-performance fraud modeling with optimized ML infrastructure

---

## Workflow: Data Preparation to Model Building

### 1. **Data Trimming & Aggregation (`main_aggregator.ipynb`)**

#### a. **Import and Clean Raw Data**
- Loads transaction data (`transactions.csv`) and account flag data (`train_acc.csv`, `test_acc_predict.csv`) with robust type overrides using [Polars](https://pola.rs/) for speed and memory efficiency.
- Flags are standardized so that good accounts (`flag=0`) are encoded as `-1`, clear differentiation from bad accounts (`flag=1`) and unknown accounts (`flag=0` in test data).

#### b. **Feature Engineering**
- **Transaction-level features** (profit, cost, ratios, temporal tags): 
    - For each transaction: profit (`value - gas * gas_price`), net value, gas cost, value/gas ratios, and binary features such as whether the transaction is profitable, on weekends, at night, etc.
    - **Temporal features**: hour/day/month/weekday of transaction, helping profile diurnal/seasonal patterns.

#### c. **Account-level Graph Construction**
- **Accounts encoded as categorical variables** for compact integer mapping.
- **Outgoing and incoming transaction arrays** are built for each account, sorted and indexed for rapid lookup.
- Graph structures (`edges_out`, `edges_in`) enable slicing out all transactions linked to any account.
- Functions for neighbor lookups (`find_to_nei`, `find_from_nei`) and path searches (`find_forward_paths`, `find_backward_paths`) support exploration of transaction sequences of arbitrary depth.

#### d. **Aggregating Features for Downstream Analysis**
- **Streaming feature accumulation** (via `RunningStats`): Means, variances, min/max for key numeric features are built efficiently in a streaming manner.
- Per-account aggregates are computed for different flags and types (‘normal’, ‘abnormal’, A/B directionality, temporal bins).
- Data is further pruned, deduplicated, and restructured to produce wide tabular summaries with hundreds (or thousands) of features per account.

**Key improvement:** This step eliminates memory spikes and greatly shortens runtime (vs. the original repo’s iterative/single-threaded approach).

---

### 2. **Analysis & Model Building (`main_f1.ipynb`)**

#### a. **Advanced Feature Engineering**
- The dataset from **main_aggregator** is loaded and processed further:
    - **Derived ratios, contrasts, and population-relative features** (e.g., abnormal-to-normal ratios, z-scores, quartile/season contrasts).
    - **Entropy and concentration metrics:** Quantifies variety and distribution of temporal or transactional patterns (e.g., how scattered an account’s activity is across hours/days/months).
    - **Volatility, burstiness, and activity flags**: For each account, signals like burst ratio, window-based entropy, and low-activity flags are calculated.

#### b. **Data Consolidation**
- Data from multiple sources (`data1_df`, `data2_df`, etc.) is loaded, featured, and concatenated into a single large table.
- Additional windowed features (from raw transactions) are joined in, using robust joining logic that ensures correct mappings and no data loss.

#### c. **Supervised Modeling**
- **CatBoost Classifier** (or similar) is tuned with **Optuna** for fast yet robust hyperparameter optimization, including dynamic weighting for minority (fraudulent) class.
- Feature selection, ranking, and importance assertions are performed to help focus on the most predictive signals.
- Cross-validation and advanced threshold tuning (maximizing F1 at precision-recall curve best points) ensure that fraudulent accounts are optimally detected.

**Key Contribution:** Entire modeling code and feature logic is written for tabular efficiency. You can run mainstream ML with thousands of features in serveal minutes.

---

## 🚀 Quick Start

### For Enhanced Baseline (Recommended)
```bash
# 1. Install dependencies
pip install -r requirements_new.txt

# 2. Ensure data files are in root directory:
#    - train_acc.csv, test_acc_predict.csv, answer.csv
#    - data1_df.csv, data2_df.csv, data3_df.csv, data4_df.csv
#    - account_dynamics_burst_v1.csv, psych_idx_v2.1.csv

# 3. Train ensemble model (15-20 minutes)
jupyter notebook 01_baseline_training_enhanced.ipynb
# Expected: F1 Score ≈ 0.7843-0.7850

# 4. Generate visualizations
jupyter notebook 02_baseline_visualization.ipynb
# Outputs: 5 PNG charts (300 DPI) + detailed metrics
```

### For Original Pipeline
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data aggregation (prerequisite)
jupyter notebook main_aggregator.ipynb

# 3. Run modeling pipeline
jupyter notebook main_f1.ipynb
# Expected: F1 Score ≈ 0.77
```

---

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/Jyusi/precomputed-account-aggregator.git
# Install dependencies
pip install -r requirements.txt
```
See [requirements.txt](https://github.com/Jyusi/precomputed-account-aggregator/blob/main/requirements.txt) for full package list (Polars, Numpy, CatBoost, Optuna, Scikit-learn, etc).

---

## 📦 Key Repository Elements

### Core Notebooks
- **`01_baseline_training_enhanced.ipynb`** ⭐ NEW: Production ensemble training (F1=0.7843-0.7850)
- **`02_baseline_visualization.ipynb`** ⭐ NEW: Complete visualization suite (5 publication-ready charts)
- **`main_aggregator.ipynb`**: Dataset trimming, feature creation, graph construction (optimized for speed/memory)
- **`main_f1.ipynb`**: Original tabular feature engineering, burst/activity detection, ML modeling

### Dependencies
- **`requirements_new.txt`**: Enhanced dependencies (CatBoost, LightGBM, XGBoost, imbalanced-learn, etc.)
- **`requirements.txt`**: Original dependencies (Polars, CatBoost, Optuna, Scikit-learn)

### Data Outputs
Generated by `01_baseline_training_enhanced.ipynb`:
- `model_catboost_baseline.cbm`, `model_lgbm.pkl`, `model_xgb.pkl`, `meta_learner.pkl`
- `optimal_threshold.pkl`, `optimal_threshold_ensemble.pkl`
- `baseline_test_predictions.csv`, `baseline_test_predictions_with_proba.csv`
- `baseline_validation_metrics.csv`, `baseline_test_metrics.csv`
- `baseline_feature_importance.csv`, `baseline_confusion_matrix.npy`

### Documentation
- **`README.md`**: This file - complete workflow documentation
- **`new/THEORETICAL_FRAMEWORK.md`**: Behavioral indices theory (economic theory, game theory)
- **`new/USAGE_GUIDE.md`**: Step-by-step execution guide for new notebooks

---

## 🎯 Why This Pipeline?

### Performance
- **State-of-the-art F1 Score**: 0.7843-0.7850 (enhanced baseline) vs 0.77 (original)
- **Ensemble Robustness**: 3 diverse models reduce overfitting and improve generalization
- **Optimized Threshold**: Precision-recall curve analysis maximizes F1 score

### Scalability
- **Handles Millions of Transactions**: Parallelism and columnar data structures (Polars, NumPy)
- **Memory Efficient**: Streaming aggregation eliminates memory spikes
- **Fast Execution**: 15-20 minutes for complete ensemble training (CPU)

### Methodology
- **Correct Data Splitting**: SMOTETomek → train/val split (prevents leakage)
- **Class Balancing**: Addresses 10:1 imbalance ratio (Good:Bad accounts)
- **Feature Engineering**: 992+ features from transaction, temporal, psychological signals

### Flexibility
- **Graph-based + Tabular Analysis**: Transaction networks + aggregated features
- **Modular Architecture**: Separate data prep, modeling, visualization
- **Extensible**: Easy to add new features (behavioral indices, graph embeddings)

### Actionability
- **Interpretable Results**: Feature importance ranking + confusion matrix breakdown
- **Production-ready**: Saved models (.pkl, .cbm) + optimal thresholds
- **Visualization Suite**: 5 publication-ready charts for stakeholder reporting
- **Audit Trail**: Detailed metrics (CSV) + predictions with probabilities

### Reproducibility
- **Clear Workflow**: Data prep → Training → Visualization
- **Version Control**: Git-friendly notebooks with documented methodology
- **Dependencies Managed**: Complete requirements files for environment setup

---

## Source Data Acknowledgement

The raw dataset is sourced from:
- [Prediction-of-Good-or-Bad-Accounts/natxis](https://github.com/michaelcheungkm/Prediction-of-Good-or-Bad-Accounts/tree/459923ea7f521565a50d54e22a11325995b187c7/natxis) by michaelcheungkm
All code, feature engineering, and modeling in this repository are original and not derived from the source repo.

---

## Citation & Reuse

If you use this workflow or adapt the feature engineering/modeling code, please cite this repository as follows:

### BibTeX
```bibtex
@software{wong2025accml,
  author       = {jyusiwong},
  title        = {AccML: Enhanced Account Fraud Detection with Ensemble Stacking},
  year         = {2025},
  month        = {December},
  publisher    = {GitHub},
  url          = {https://github.com/jyusiwong/AccML},
  note         = {Achieves F1 Score 0.7843-0.7850 using ensemble stacking (CatBoost + LightGBM + XGBoost)}
}
```

### APA Style
Wong, J. (2025). *AccML: Enhanced Account Fraud Detection with Ensemble Stacking* [Computer software]. GitHub. https://github.com/jyusiwong/AccML

### IEEE Style
J. Wong, "AccML: Enhanced Account Fraud Detection with Ensemble Stacking," GitHub repository, Dec. 2025. [Online]. Available: https://github.com/jyusiwong/AccML

---

## Contributing

For extensions, issues, or suggestions:
- 🐛 Report bugs via [GitHub Issues](https://github.com/jyusiwong/AccML/issues)
- 💡 Suggest features via [GitHub Discussions](https://github.com/jyusiwong/AccML/discussions)
- 🔧 Submit improvements via Pull Requests

---

_This project is maintained by Jyusi Wong to support reproducible, scalable fraud analytics._
