# Precomputed Account Aggregator for Fraud Detection

This repository archives a **refined and resource-efficient workflow** for fraud account detection in large-scale transaction data. It builds on the original dataset provided by [michaelcheungkm/Prediction-of-Good-or-Bad-Accounts](https://github.com/michaelcheungkm/Prediction-of-Good-or-Bad-Accounts/tree/459923ea7f521565a50d54e22a11325995b187c7/natxis), but **completely redesigns and improves** the dataset preparation and modeling pipeline.

This workflow is able to provide efficient raw data handling and decision tree machine learning, with the best f1 score of 0.77.

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

## Key Repository Elements

- **main_aggregator.ipynb:** All dataset trimming, feature creation, graph construction—optimized for speed/memory.
- **main_f1.ipynb:** New tabular feature engineering, burst/activity detection, ML modeling, feature importance ranking.
- **requirements.txt:** All dependencies for prep and modeling (Polars, CatBoost, Optuna, etc).
- **README.md:** You’re reading this!

---

## Why this pipeline?

- **Scalable**: Handles millions of transactions efficiently, using parallelism and columnar data structures.
- **Efficiency:** Preprocessing and aggregation routines are built for speed and memory balance, suitable for larger or more complex datasets.
- **Flexible**: Facilitates both graph-based and tabular analysis.
- **Rich Features**: Aggregates behavioral, temporal, and statistical signals critical for fraud detection.
- **Actionable**: Outputs are ready for machine learning but also usable for audit, analytics, and further network investigation.
- **Future-Proof:** Adaptable architecture—add new features, run different ML backends, or swap in new transaction sources easily.
- **Composability:** Clearly split data prep and modeling, supporting pipeline orchestration and reproducibility.

---

## Source Data Acknowledgement

The raw dataset is sourced from:
- [Prediction-of-Good-or-Bad-Accounts/natxis](https://github.com/michaelcheungkm/Prediction-of-Good-or-Bad-Accounts/tree/459923ea7f521565a50d54e22a11325995b187c7/natxis) by michaelcheungkm
All code, feature engineering, and modeling in this repository are original and not derived from the source repo.

---

## Citation & Reuse

If you use this workflow or adapt the feature engineering/modeling code, please cite or reference this repository. For extensions, issues or suggestions, please open an issue or PR.

---

_This project is maintained to support reproducible, scalable fraud analytics. For questions or contributions, use GitHub Issues._
