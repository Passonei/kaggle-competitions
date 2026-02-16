from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def compare_hists(frame, cols):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(cols),
        figsize=(25, 4),
        sharey=False
    )

    for ax, col in zip(axes, cols):
        sns.boxplot(
            data=frame,
            x="loan_paid_back",
            y=col,
            hue="loan_paid_back",
            palette=["m", "g"],
            legend=False,
            showfliers=False,
            ax=ax
        )
    ax.set_title(col)

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.show()


def compare_counts(frame, cols):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(cols)//2,
        figsize=(25, 6),
        sharey=False
    )

    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        sns.countplot(
            data=frame,
            x=col,
            hue="loan_paid_back",
            ax=ax
        )
        ax.set_title(col)

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.show()


def missing_data(df: pd.DataFrame, n: int):
    """
    Displays DataFrame with info about missing data per feature

    :param df: df used for analysing of nulls
    :type df: pd.DataFrame
    :param n: number of null rows to display
    :type n: int
    """
    null_sum = df.isnull().sum()
    # Total number of missing values
    total = null_sum.sort_values(ascending=False)
    percentage = (null_sum / df.isnull().count()
                  ).sort_values(ascending=False) * 100  # % of Missing values
    unique_values = df.nunique()
    missing_data = pd.concat([total, percentage, unique_values], axis=1,
                             keys=['Missing', '% Missing', 'Unique values'], sort=False)

    return missing_data.head(n)


def score_roc(X, y, pipeline, plot=True):
    y_proba = pipeline.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    auc = roc_auc_score(y, y_proba)

    if plot:
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="random")
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    else:
        print("AUC-ROC: ", round(auc, 4))


class Log1pImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="median"):
        self.strategy = strategy
        self.imputer_ = SimpleImputer(strategy=self.strategy)

    def fit(self, X: pd.DataFrame, y=None):
        self.imputer_.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        X_out = self.imputer_.transform(X)

        if np.nanmin(X_out) < -1:
            raise ValueError("Log1pImuter values must be >= -1")
        return np.log1p(X_out)


class RatioInteractionImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="median"):
        self.strategy = strategy
        self.imputer_ = SimpleImputer(strategy=self.strategy)

    def fit(self, X: pd.DataFrame, y=None):
        self.imputer_.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        X_out = self.imputer_.transform(X)
        X_out["rate_per_score"] = X_out["intrest_rate"] / X_out["credit_score"]
        X_out["inter_cs_ir"] = X_out["intrest_rate"] * X_out["credit_score"]

        return X_out
