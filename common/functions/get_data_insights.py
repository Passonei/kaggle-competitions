import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np

from typing import Iterator

from common.utils import get_next_feature_from_iter


def get_quantitative_and_categorical_columns(df: pd.DataFrame, unique_threshold: int = 10):
    """
    Separate DataFrame columns into quantitative and categorical based on unique value count.
    
    Args:
        df: DataFrame to analyze
        unique_threshold: columns with more unique values than this are considered quantitative
    
    Returns:
        tuple: (quantitative_columns, categorical_columns) as lists
    """
    quantitative, categorical = [], []
    for col in df.columns:
        if len(df[col].unique()) > unique_threshold:
            quantitative.append(col)
        else:
            categorical.append(col)
    
    return quantitative, categorical


def missing_data(df: pd.DataFrame, n: int):
    """
    Docstring for missing_data
    
    :param df: df used for analysing of nulls
    :type df: pd.DataFrame
    :param n: number of null rows to display
    :type n: int
    """
    null_sum = df.isnull().sum()
    total = null_sum.sort_values(ascending=False) # Total number of missing values
    percentage = (null_sum / df.isnull().count()).sort_values(ascending=False) * 100  # % of Missing values
    unique_values = df.nunique()
    missing_data = pd.concat([total, percentage, unique_values], axis=1, 
                             keys=['Missing', '% Missing', 'Unique values'], sort=False)
    
    display(missing_data.head(n))


def guess_feature_type(df: pd.DataFrame, max_unique_count: int = 30):
    numerical_features = set()
    categorical_features = set()
    other_features = set()
    
    for col in df.columns:
        unique_count = len(df[col].unique())
        if unique_count <= max_unique_count:
            # Most likely categorical
            categorical_features.add(col)
        else:
            try:
                converted = pd.to_numeric(df[col], errors='raise')
                numerical_features.add(col)
            except Exception as e:
                print(f'Did not converted to numeric column: {col}')
                other_features.add(col)
    
    return numerical_features, categorical_features, other_features


def create_numerical_viz_iter(df: pd.DataFrame, features: Iterator, target_feature: str):
    """
    Creates 4 subplots:
      1) Histogram + KDE of feature
      2) Boxplot of feature
      3) Scatter: feature vs target
      4) Scatter: feature vs log1p(target)
    :param df: DataFrame
    :param features: Iterator of features
    :param target_feature: target feature
    """
    feature = get_next_feature_from_iter(features)
    
    if feature == target_feature:
        print("Feature is the same as target feature")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4, ax5, _ = axes.ravel()

    fig.suptitle(f'Feature diagnostics: {feature}', fontsize=30)

    # Hist + KDE
    sns.histplot(data=df, x=feature, kde=True, ax=ax1)
    ax1.set_title("Histogram + KDE")
    ax1.set_xlabel(feature)
    ax1.set_ylabel("Count")

    # Boxplot
    sns.boxplot(data=df, x=feature, ax=ax2)
    ax2.set_title("Boxplot")
    ax2.set_xlabel(feature)
    ax2.set_ylabel("Value")

    # Scatter: feature vs target
    sns.scatterplot(data=df, x=feature, y=target_feature, ax=ax3)
    ax3.set_title(f"Scatter: {feature} vs {target_feature}")
    ax3.set_xlabel(feature)
    ax3.set_ylabel(target_feature)

    tmp = df[[feature, target_feature]].copy()
    tmp = tmp.dropna()
    log_feature = f"log1p_{feature}"
    log_target_feature = f"log1p_{target_feature}"

    tmp[log_feature] = np.log1p(tmp[feature])
    tmp[log_target_feature] = np.log1p(tmp[target_feature])

    # Scatter feature vs log1p(target)
    sns.scatterplot(data=tmp, x=feature, y=log_target_feature, ax=ax4)
    ax4.set_title(f"Scatter: {feature} vs {log_target_feature}")
    ax4.set_xlabel(feature)
    ax4.set_ylabel(log_target_feature)
    
    # Scatter log1p(feature) vs log1p(target)
    sns.scatterplot(data=tmp, x=log_feature, y=log_target_feature, ax=ax5)
    ax5.set_title(f"Scatter: {log_feature} vs {log_target_feature}")
    ax5.set_xlabel(log_feature)
    ax5.set_ylabel(log_target_feature)


    plt.tight_layout()
    plt.show()
    
    return feature

    