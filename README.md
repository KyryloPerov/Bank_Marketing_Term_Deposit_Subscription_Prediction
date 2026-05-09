# Bank_Marketing_Term_Deposit_Subscription_Prediction# 

> **Mid-term project** · Machine Learning course  
> Binary classification: predict whether a bank client will subscribe to a term deposit.

---

## Problem Statement

A Portuguese bank runs direct marketing campaigns (phone calls) and wants to predict **which clients are likely to subscribe to a term deposit** (`y = yes/no`).  
Knowing this in advance lets the bank prioritise calls, reduce costs, and increase conversion.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Bank Additional Full](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv) |
| File | `bank-additional-full.csv` |
| Rows (after dedup) | 41 176 |
| Features | 20 input + 1 target |
| Target imbalance | 88.7% `no` / 11.3% `yes` |

**Feature groups:**
- **Client profile** — age, job, marital status, education, default, housing loan, personal loan
- **Last contact** — contact type, month, day of week, call duration *(dropped — data leakage)*
- **Campaign history** — number of contacts, days since last contact (`pdays`), previous contacts, previous outcome
- **Macroeconomic indicators** — `euribor3m`, `emp.var.rate`, `nr.employed`, `cons.price.idx`, `cons.conf.idx`

---

## Project Structure

```
.
├── Mid_term_Project_Bank_Marketing_Dataset_СС.ipynb   # Main notebook (all analysis)
├── process_bank_marketing.py                           # Preprocessing pipeline module
├── utils.py                                            # EDA utility functions
├── bank-additional-full.csv                            # Dataset
└── README.md
```

### Module descriptions

**`process_bank_marketing.py`** — single-responsibility preprocessing helpers + two public functions:
- `engineer_features(df)` — applies all feature transformations to raw data
- `prepare_train_val(df_proc)` — stratified split + StandardScaler (fit on train only)

**`utils.py`** — reusable EDA helpers:
- `describe_columns_summary(df)` — quick overview of dtypes, unique counts, examples
- `value_counts_of_column(df, col)` — formatted top-N frequency table
- `calculate_iqr_bounds(df, col)` — IQR-based outlier statistics
- `analyze_numeric_vs_target(df, col)` — boxplot, KDE, quantile conversion rate
- `analyze_categorical_vs_target(df, col)` — countplot, normalized distribution, conversion rate bar chart

---

## Methodology

### Evaluation Metric

**Primary: ROC-AUC** — measures the model's ability to rank clients by subscription probability.  
**Secondary: F1-score** — balances Precision and Recall for the minority class.

Accuracy was explicitly rejected due to severe class imbalance (an always-`no` model would score 88.7%).

### Preprocessing Pipeline

| Step | Action | Reasoning |
|---|---|---|
| Drop `duration` | Remove column | Data leakage — value is only known after the call ends |
| Drop `day_of_week` | Remove column | Near-identical conversion rate across all weekdays |
| `age` → `age_group` | Bin into 6 buckets | U-shaped non-linear relationship: young (<20) and senior (>60) convert more |
| `pdays` + `previous` → `contacted_before` | 4-level feature | `not_contacted` / `contacted_long_ago` / `1_time` / `2+times` |
| `month` → `period` | 3-tier grouping | `high_season` (mar/sep/oct/dec >40%), `low_season` (may/jul <10%), `mid_season` |
| `education` → `education_ord` | Ordinal encoding (0–6) | Natural ordinal order exists; `unknown` mapped to median level 3 |
| `campaign` | Cap at 99th percentile | Long right tail (max=56); cap prevents distortion without losing signal |
| Nominal columns | One-hot encoding (`drop_first=True`) | Avoids dummy variable trap for Logistic Regression |
| Train/Val split | 80/20 stratified | Preserves 11.3% positive rate in both sets |
| `StandardScaler` | Fit on train only | Applied for distance/linear models; prevents data leakage from validation set |

### Key EDA Hypotheses (all confirmed)

1. `duration` is the strongest predictor but excluded (data leakage)
2. `poutcome = success` → ~6.5× higher subscription probability
3. `contact = cellular` → ~3× more effective than landline
4. Seasonal pattern: Mar/Sep/Oct/Dec conversion >40%; May/Jul <10%
5. `job = student/retired` → highest conversion rates
6. Age has a **U-shaped** relationship with subscription
7. Low `euribor3m` and `emp.var.rate` correlate with higher subscription
8. `pdays < 999` + `previous > 0` → significantly higher probability

---

## Models & Results

| Model | Params | Train AUC | Val AUC | Val F1 | Comment |
|---|---|---|---|---|---|
| Dummy (baseline) | most_frequent | 0.5000 | 0.5000 | 0.0000 | Underfit — random guess |
| Logistic Regression | C=0.1, balanced | 0.7889 | 0.7946 | 0.4466 | Good |
| kNN | k=25, distance weights | 0.9990 | 0.7323 | 0.3209 | Overfit — curse of dimensionality |
| Decision Tree | depth=8, balanced | 0.8060 | 0.7982 | 0.4700 | Good |
| XGBoost (base) | n=300, depth=4, lr=0.1 | 0.8619 | 0.8058 | 0.4667 | Slight overfit |
| LightGBM (base) | n=300, depth=4, lr=0.1 | 0.8551 | **0.8106** | **0.4757** | **Best base model** |
| LightGBM (RandomizedSearchCV) | 50 iterations, 5-fold CV | — | ~0.82 | — | Improved over base |
| LightGBM (Hyperopt TPE) | 50 evaluations, 5-fold CV | — | **~0.82+** | — | **Best overall** |

### Hyperparameter Tuning

Two methods were applied to LightGBM (best base model):

- **RandomizedSearchCV** — samples hyperparameter combinations independently from predefined distributions; 50 iterations with 5-fold stratified CV.
- **Hyperopt Bayesian Optimization (TPE)** — builds a probabilistic model of past evaluations and directs search towards promising regions; same search space and 50 evaluations.

Tuned hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `subsample`, `colsample_bytree`, `min_child_samples`, `reg_alpha`, `reg_lambda`.

Both methods improved upon the base model. Hyperopt consistently finds equal or better results than RandomizedSearchCV for the same iteration budget.

---

## Feature Importance & SHAP

**Top features by gain (LightGBM tuned):**

1. `euribor3m` — macroeconomic interest rate: low rates → clients prefer term deposits
2. `nr.employed` — employment level: proxy for economic cycle
3. `contacted_before_*` — prior campaign contact history: previously contacted clients convert more
4. `poutcome_success` — previous campaign success: strongest single categorical predictor
5. `campaign` — number of current-campaign contacts: more calls → lower conversion

**SHAP analysis confirmed:**
- `euribor3m` has a clear non-linear effect — SHAP value drops sharply above 4.0
- `poutcome_success` provides a strong positive push regardless of macroeconomic context
- The model uses economically meaningful signals, not dataset artefacts

---

## Error Analysis

**False Negatives** (missed subscribers — most costly for the bank):
- Tend to have higher `euribor3m` and more contacts in the current campaign
- The model "logically fails" — these clients genuinely look like non-subscribers by their profile

**False Positives** (wasted calls):
- Predicted probability typically in the 0.50–0.65 range — model is uncertain but crosses the threshold

**Recommendations for improvement:**
1. **Lower classification threshold** (e.g. 0.35) to increase Recall at the cost of Precision
2. **Add features** — account balance, number of bank products, CRM activity history
3. **Oversampling (SMOTE)** — as an alternative to `scale_pos_weight`
4. **Model stacking** — LightGBM + Logistic Regression ensemble
5. **Temporal features** — the dataset spans multiple years with different macro conditions

---

## How to Run

### Requirements

```
python >= 3.9
pandas
numpy
scikit-learn
lightgbm
xgboost
hyperopt
shap
matplotlib
seaborn
scipy
jupyter
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost hyperopt shap matplotlib seaborn scipy jupyter
```

### Run the notebook

```bash
git clone <repo-url>
cd <repo-folder>
# Place bank-additional-full.csv in the project root
jupyter notebook Mid_term_Project_Bank_Marketing_Dataset_СС.ipynb
```

Run all cells top-to-bottom. Sections 6 (hyperparameter tuning) take ~5–10 minutes due to 50-iteration cross-validation.

---

## Conclusions

- A LightGBM model tuned with Hyperopt achieves **Val AUC ≈ 0.82**, a **+32 p.p. improvement** over the random baseline.
- Feature engineering (`age_group`, `contacted_before`, `period`) contributed meaningfully to model quality.
- The preprocessing pipeline is fully modular and reusable via `process_bank_marketing.py`.
- SHAP analysis confirms the model's decisions are economically interpretable and trustworthy.
- The primary bottleneck is the strong class imbalance and the fact that macro-economic conditions dominate individual client characteristics.
