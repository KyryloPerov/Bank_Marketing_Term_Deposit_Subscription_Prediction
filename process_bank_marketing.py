"""
Module for preprocessing Bank Marketing Campaign data.

Structure:
  - Private helper functions (_name): each responsible for exactly one transformation.
  - Two public functions use these helpers:
      engineer_features()   — all domain-specific feature transformations
      prepare_train_val()   — stratified split + scaling
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_COL = "y"


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions — one action each
# ══════════════════════════════════════════════════════════════════════════════

def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode target column 'y': yes → 1, no → 0."""
    out = df.copy()
    out[TARGET_COL] = (out[TARGET_COL] == "yes").astype(int)
    return out


def _drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that cause data leakage or carry no predictive signal.

    - duration: known only after the call ends — direct leakage.
    - day_of_week: EDA showed near-identical conversion across all weekdays.
    """
    return df.drop(columns=["duration", "day_of_week"])


def _create_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Replace numeric 'age' with categorical 'age_group'.

    EDA revealed a U-shaped relationship: clients aged ≤20 and >60 subscribe
    more often than the middle-age bulk. Bucketing preserves this non-linear
    signal while eliminating the need to handle age outliers separately.
    """
    def _bucket(age: int) -> str:
        if age <= 20:    return "young"
        elif age <= 30:  return "20-30"
        elif age <= 40:  return "30-40"
        elif age <= 50:  return "40-50"
        elif age <= 60:  return "50-60"
        else:            return "senior"

    out = df.copy()
    out["age_group"] = out["age"].apply(_bucket)
    return out.drop(columns=["age"])


def _create_contacted_before(df: pd.DataFrame) -> pd.DataFrame:
    """Merge 'pdays' and 'previous' into a single 4-level feature.

    Levels:
      not_contacted      — previous = 0  (never reached in any prior campaign)
      contacted_long_ago — previous > 0, pdays = 999  (>999 days ago)
      1_time             — previous = 1, pdays < 999  (once, recently)
      2+times            — previous > 1, pdays < 999  (multiple times, recently)

    The raw columns are dropped afterwards; this single feature captures all
    their information in a form the model can learn from more easily.
    """
    out = df.copy()
    out["contacted_before"] = np.where(
        out["previous"] == 0, "not_contacted",
        np.where(out["pdays"] == 999, "contacted_long_ago",
        np.where(out["previous"] == 1, "1_time", "2+times"))
    )
    return out.drop(columns=["pdays", "previous"])


def _create_period(df: pd.DataFrame) -> pd.DataFrame:
    """Group months into conversion-rate tiers and drop the raw 'month' column.

    Tiers are based on empirical conversion rates from the full dataset:
      high_season  — mar(50.5%), sep(44.9%), oct(43.9%), dec(48.9%)  → >40%
      low_season   — may(6.4%), jul(9.0%)                            → <10%
      mid_season   — everything else (apr, jun, aug, nov)            → 10–21%

    Note: jan and feb have no observations in this dataset.
    Note: sep and oct are in 'high_season', NOT 'good_season' —
          both exceed 40%, the same threshold as mar and dec.
    """
    def _tier(m: str) -> str:
        if m in {"mar", "sep", "oct", "dec"}: return "high_season"
        if m in {"may", "jul"}:               return "low_season"
        return "mid_season"

    out = df.copy()
    out["period"] = out["month"].apply(_tier)
    return out.drop(columns=["month"])


def _encode_education(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ordinal encoding to 'education'.

    A natural order exists (illiterate → university degree), so label encoding
    is more appropriate than OHE. 'unknown' is mapped to the median level (3).
    """
    edu_order = {
        "illiterate": 0, "basic.4y": 1, "basic.6y": 2, "basic.9y": 3,
        "high.school": 4, "professional.course": 5, "university.degree": 6,
        "unknown": 3,
    }
    out = df.copy()
    out["education_ord"] = out["education"].map(edu_order)
    return out.drop(columns=["education"])


def _cap_campaign(df: pd.DataFrame) -> pd.DataFrame:
    """Cap 'campaign' outliers at the 99th percentile.

    The raw column has a long right tail (max = 56, 99th pct ≈ 14).
    Capping retains 99% of observations unchanged while preventing extreme
    values from distorting distance-based and linear models.
    """
    out = df.copy()
    cap = out["campaign"].quantile(0.99)
    out["campaign"] = out["campaign"].clip(upper=cap)
    return out


def _one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all remaining nominal categorical columns.

    drop_first=True removes one dummy per feature to avoid the dummy variable
    trap (perfect multicollinearity), which matters for Logistic Regression.
    """
    nominal_cols = [
        "job", "marital", "default", "housing", "loan", "contact",
        "poutcome", "age_group", "contacted_before", "period",
    ]
    return pd.get_dummies(df, columns=nominal_cols, drop_first=True)


def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple:
    """Stratified train/validation split.

    Stratification preserves the class ratio (≈11% positive) in both sets,
    which is critical given the strong class imbalance.
    """
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def _scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple:
    """Fit StandardScaler on train set and transform both sets.

    Fitting only on train data prevents data leakage: the validation set
    must be treated as if it were completely unseen during preprocessing.
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    return X_train_sc, X_val_sc, scaler


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all domain-specific feature transformations to the raw dataframe.

    Calls helper functions in sequence; each helper is responsible for
    exactly one transformation step.

    Args:
        df: Raw dataframe loaded from bank-additional-full.csv
            (duplicates already removed).

    Returns:
        Preprocessed dataframe ready for train/val split.
    """
    df = _encode_target(df)
    df = _drop_leaky_columns(df)
    df = _create_age_group(df)
    df = _create_contacted_before(df)
    df = _create_period(df)
    df = _encode_education(df)
    df = _cap_campaign(df)
    df = _one_hot_encode(df)
    return df


def prepare_train_val(
    df_proc: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Split the preprocessed dataframe and prepare feature matrices.

    Produces two versions of the features:
      - Unscaled (X_train / X_val)   : for tree-based models.
      - Scaled   (X_train_sc / X_val_sc): for distance/linear models.

    Args:
        df_proc: Output of engineer_features().
        test_size: Fraction of data for validation. Default 0.2.
        random_state: Seed for reproducibility. Default 42.

    Returns:
        Dictionary with keys:
          X_train, X_val        — unscaled feature DataFrames
          X_train_sc, X_val_sc  — scaled feature arrays
          y_train, y_val        — target Series
          feature_cols          — ordered list of feature column names
          scaler                — fitted StandardScaler instance
    """
    feature_cols = [c for c in df_proc.columns if c != TARGET_COL]
    X = df_proc[feature_cols].astype(float)
    y = df_proc[TARGET_COL]

    X_train, X_val, y_train, y_val = _split_data(X, y, test_size, random_state)
    X_train_sc, X_val_sc, scaler = _scale_features(X_train, X_val)

    return {
        "X_train":      X_train,
        "X_val":        X_val,
        "X_train_sc":   X_train_sc,
        "X_val_sc":     X_val_sc,
        "y_train":      y_train,
        "y_val":        y_val,
        "feature_cols": feature_cols,
        "scaler":       scaler,
    }
