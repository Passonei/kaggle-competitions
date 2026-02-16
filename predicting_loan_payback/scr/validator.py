import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, auc, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.base import clone

from scr.utils import score_roc
from typing import Any


class Validator:
    def __init__(self, preprocessing_pipeline: Pipeline, model: Any, X_train: pd.DataFrame, y_train: pd.Series):
        self.preprocessing = preprocessing_pipeline
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.full_pipeline = self._fit_model(X_train, y_train)
        self.numeric_cols = X_train.select_dtypes(include=np.number).columns
        self.numeric_cols = X_train.select_dtypes(include=np.number).columns

    def stability_test(self, model, feature_frame: pd.DataFrame, target: pd.Series, folds: int) -> None:
        preprocessed_feature_frame = self.preprocessing.fit_transform(
            feature_frame.copy())
        cross_vals = cross_val_score(
            model, preprocessed_feature_frame, target, cv=folds, scoring="roc_auc")
        print("Cross validation: \n", pd.Series(
            cross_vals).describe().round(4))

    def stability_bootstrap_test(self, model, feature_frame: pd.DataFrame, target: pd.Series) -> None:
        results = []
        for _ in range(500):
            mask = np.random.choice(
                a=[False, True], size=feature_frame.shape[0], p=[0.95, 0.05])
            sampled_target = target[mask]
            sampled_feature_frame = feature_frame[mask]
            sampled_proba = self.full_pipeline.predict(sampled_feature_frame)
            results.append(roc_auc_score(sampled_target, sampled_proba))

        print("Bootstrap: \n", pd.Series(results).describe().round(4))

    def robustness_test(self, feature_frame: pd.DataFrame, target: pd.Series, noise: str, delta: int = 1) -> None:
        noisy_feature_frame = feature_frame.copy()

        if noise == "gaussian":
            noisy_feature_frame = self._add_gaussian_noise(noisy_feature_frame)
        elif noise == "quantization":
            noisy_feature_frame = self._add_quantization_noise(
                noisy_feature_frame, delta)
        else:
            raise ValueError(
                "not implemented, available: gaussian/quantization")

        score_roc(noisy_feature_frame, target, self.full_pipeline)

    def sanity_test(self, feature_frame: pd.DataFrame, target: pd.Series) -> None:
        invalid_target = np.random.permutation(target)
        score_roc(feature_frame.copy(), invalid_target, self.full_pipeline)

    def distribution_test(self, train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> None:
        train_frame["y"] = 0
        test_frame["y"] = 1

        dist_frame = pd.concat([train_frame, test_frame], axis=0)

        X_train_dist, X_val_dist, y_train_dist, y_val_dist = train_test_split(
            dist_frame.drop("y", axis=1), dist_frame["y"], test_size=0.2, random_state=42)

        pipeline = self._fit_model(X_train_dist, y_train_dist)
        score_roc(X_val_dist, y_val_dist, pipeline)

    def semantic_test(self, target_true: pd.Series, target_pred: pd.Series, type: str) -> None:
        target_frame = pd.DataFrame({"true": target_true, "pred": target_pred})
        target_frame["decile"] = pd.qcut(
            target_frame["pred"], 10, labels=False)

        if type == "error":
            self.error_analysis(target_frame)
        elif type == "calibration":
            self.calibration_analysis(target_frame)
        elif type == "score_distributions":
            self.score_distribution_analysis(target_frame)
        else:
            raise ValueError(
                "not implemented, available: error/calibration/score_distributions")

    @staticmethod
    def score_distribution_analysis(target_frame: pd.DataFrame) -> None:
        sns.histplot(
            data=target_frame,
            x="pred",
            hue='true',
            bins=50,
            stat='density',
            element='step',
            fill=True,
            alpha=0.3,
        )

    @staticmethod
    def calibration_analysis(target_frame: pd.DataFrame) -> None:
        prob_true, prob_pred = calibration_curve(
            target_frame["true"], target_frame["pred"], n_bins=10
        )

        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.show()

        brier = brier_score_loss(target_frame["true"], target_frame["pred"])
        print("Brier score:", round(brier, 2))

    @staticmethod
    def error_analysis(target_frame: pd.DataFrame, fpr_partial: float = 0.05) -> None:
        decile_stats = (
            target_frame
            .groupby("decile")
            .agg(
                positive_rate=("true", "mean"),
                count=("true", "size"),
                avg_score=("pred", "mean")
            )
            .sort_index(ascending=False)
        )
        print("Base rate: ", target_frame["true"].mean().round(3))
        print(decile_stats.round(3), "\n")

        fpr, tpr, _ = roc_curve(target_frame["true"], target_frame["pred"])
        mask = fpr <= fpr_partial
        partial_auc = auc(fpr[mask], tpr[mask]) / 0.05

        print(f"Partial AUC (FPR <= {round(fpr_partial*100)}%):", partial_auc)

    def _fit_model(self, X, y):
        return Pipeline(steps=[
            ("preprocess", clone(self.preprocessing)),
            ("model", clone(self.model))
        ]).fit(X, y)

    def _add_gaussian_noise(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        noise_strength = 0.2

        stds = feature_frame[self.numeric_cols].std()

        noise = np.random.normal(
            loc=0,
            scale=stds.values * noise_strength,
            size=feature_frame[self.numeric_cols].shape
        )

        feature_frame[self.numeric_cols] += noise
        feature_frame[self.numeric_cols] = np.where(
            feature_frame[self.numeric_cols] < 0, 0, feature_frame[self.numeric_cols])
        return feature_frame

    def _add_quantization_noise(self, feature_frame: pd.DataFrame, delta: int) -> pd.DataFrame:
        feature_frame[self.numeric_cols] = (
            np.round(feature_frame[self.numeric_cols] / delta) * delta
        )
        return feature_frame
