"""
EDA utility functions for the Bank Marketing mid-term project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def value_counts_of_column(df, column, n=5):
    """Print a formatted top-N value frequency table for a given column."""
    print(f"______Підрахунок значень колонки '{column}'______\n")

    value_counts = df[column].value_counts().sort_values(ascending=False).head(n)
    value_counts_norm = (
        df[column].value_counts(normalize=True).sort_values(ascending=False).head(n) * 100
    )

    print(f"Топ {n} значень:\n")
    print(f"{'Значення':<15} | {'Кількість':>10} | {'Відсоток':>8}")
    print("-" * 42)
    for value in value_counts.index:
        print(
            f"{str(value):<15} | {str(value_counts[value]):>10} | {value_counts_norm[value]:>7.2f}%"
        )
    print()


def describe_columns_summary(df, top_n=5):
    """Print a concise summary table: dtype, unique count, missing count, and examples."""
    print(f"{'column':<20} | {'dtype':<10} | {'unique':>6} | {'missing':>7} | examples")
    print("-" * 95)

    for col in df.columns:
        dtype = df[col].dtype
        unique_vals = df[col].dropna().unique()
        n_missing = df[col].isna().sum()
        examples = ", ".join([str(v) for v in unique_vals[:top_n]])
        print(
            f"{col:<20} | {str(dtype):<10} | {len(unique_vals):>6} | {n_missing:>7} | [{examples}]"
        )

    object_columns = df.select_dtypes(include="object").columns
    numeric_columns = df.select_dtypes(include="number").columns

    print("-" * 95)
    print(f"Numeric columns: {len(numeric_columns)}     Categorical columns: {len(object_columns)}")


def calculate_iqr_bounds(df, col):
    """Return IQR-based outlier bounds and outlier count for a numeric column."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    pct_outliers = n_outliers / len(df) * 100
    return {
        "Q1": Q1, "Q3": Q3, "IQR": IQR,
        "lower": lower, "upper": upper,
        "n_outliers": n_outliers, "pct_outliers": pct_outliers,
    }


def analyze_numeric_vs_target(df, col, target="y"):
    """
    Print stats and render three plots for a numeric column vs. the binary target.

    Outputs:
      - Basic descriptive statistics.
      - IQR bounds and outlier count.
      - Pearson correlation with the target.
      - Mean / median / std broken down by target class.
      - Boxplot, KDE distribution by class, and conversion rate per quantile bin.

    Args:
        df: DataFrame containing the column and target.
        col: Name of the numeric column to analyse.
        target: Name of the binary target column. Default 'y' (matches our dataset).
    """
    sep = "-" * 42
    print(sep)
    print(f"  Аналіз: {col}")
    print(sep)

    # Basic stats
    print("\nЗагальна статистика:")
    print(df[col].describe().to_string())

    # IQR / outliers
    stats = calculate_iqr_bounds(df, col)
    print(f"\nIQR: {stats['IQR']:.2f}")
    print(f"Outlier bounds: [{stats['lower']:.2f}, {stats['upper']:.2f}]")
    print(f"Outliers: {stats['n_outliers']} ({stats['pct_outliers']:.2f}%)")

    # Correlation with target
    y_num = (df[target] == "yes").astype(int) if df[target].dtype == object else df[target]
    corr = df[col].corr(y_num)
    print(f"\nКореляція з таргетом: {corr:.3f}")

    # Stats per class
    print("\nСтатистика по класах:")
    print(df.groupby(target)[col].agg(["mean", "median", "std"]).round(3).to_string())

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(data=df, x=target, y=col, ax=axes[0])
    axes[0].set_title(f"Boxplot: {col} by {target}")

    sns.kdeplot(data=df, x=col, hue=target, fill=True, ax=axes[1])
    axes[1].set_title(f"KDE: {col} by {target}")

    df_temp = df[[col, target]].copy()
    df_temp["bin"] = pd.qcut(df_temp[col], q=5, duplicates="drop")
    bin_stats = df_temp.groupby("bin", observed=True)[target].apply(
        lambda x: (x == "yes").mean() if x.dtype == object else x.mean()
    )
    bin_stats.plot(kind="bar", ax=axes[2], color="#4C72B0", edgecolor="white")
    axes[2].set_title(f"P(y=yes) by quantile bins: {col}")
    axes[2].set_ylabel("Conversion rate")
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()


def analyze_categorical_vs_target(df, col, target="y"):
    """
    Print conversion stats and render three plots for a categorical column vs. target.

    Outputs:
      - Number of unique categories.
      - P(target=yes) per category, sorted descending.
      - Countplot, normalized distribution by target class, and conversion rate bar chart.

    Args:
        df: DataFrame containing the column and target.
        col: Name of the categorical column to analyse.
        target: Name of the binary target column. Default 'y'
    """
    sep = "-" * 42
    print(sep)
    print(f"  Аналіз: {col}")
    print(sep)
    print(f"Unique categories: {df[col].nunique()}")

    # Conversion rate per category
    conversion = (
        df.groupby(col)[target]
        .value_counts(normalize=True)
        .unstack()
    )
    conversion = (
        conversion["yes"] if "yes" in conversion.columns else conversion.iloc[:, 1]
    )
    conversion = conversion.sort_values(ascending=False)

    print(f"\nP({target}=yes) by {col}:")
    clean = (conversion * 100).round(1).astype(str) + "%"
    clean.index.name = None
    print(clean.to_string())

    vc = df[col].value_counts()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Countplot — скільки спостережень у кожній категорії
    sns.countplot(data=df, x=col, order=vc.index, ax=axes[0])
    axes[0].set_title(f"Count: {col}")
    axes[0].tick_params(axis="x", rotation=45)

    # 2. Normalized distribution by target — як розподілені класи всередині кожної категорії
    percent_df = (
        pd.crosstab(df[col], df[target], normalize="columns") * 100
    ).reset_index()
    percent_long = percent_df.melt(id_vars=col, var_name=target, value_name="percent")
    sns.barplot(data=percent_long, x=col, y="percent", hue=target,
                order=vc.index, ax=axes[1])
    axes[1].set_title(f"Normalized distribution by {target}: {col}")
    axes[1].set_ylabel("Percent")
    axes[1].tick_params(axis="x", rotation=45)

    # 3. Conversion rate — P(y=yes) для кожної категорії
    conversion.loc[vc.index].plot(kind="bar", ax=axes[2],
                                  color="#4C72B0", edgecolor="white")
    axes[2].set_title(f"P({target}=yes) by {col}")
    axes[2].set_ylabel("Conversion rate")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
