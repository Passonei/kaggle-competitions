"""
Housing Price Prediction Pipeline Transformers.

Custom sklearn-compatible transformers for the House Prices Kaggle competition.
Each transformer follows the BaseEstimator/TransformerMixin pattern for use in sklearn Pipelines.
"""

import numpy as np
import pandas as pd

from typing import Sequence, List, Dict, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, PolynomialFeatures


class Log1pFeatureImputer(BaseEstimator, TransformerMixin):
    """
    Apply log1p transformation to specified numeric features.
    
    Log1p (log(1 + x)) is useful for right-skewed features like LotArea, GrLivArea.
    It helps normalize distributions and reduce the impact of outliers.
    
    Args:
        features: List of column names to transform
        drop_original: If True, remove original columns after transformation
        clip_negative: If True, clip negative values to 0 before transformation
        prefix: Prefix for new column names (default: "log1p_")
    
    Example:
        >>> imputer = Log1pFeatureImputer(["LotArea", "GrLivArea"])
        >>> X_transformed = imputer.fit_transform(df)
    """
    def __init__(self, features: List[str], drop_original: bool = True, clip_negative: bool = True, prefix: str = "log1p_"):
        self.features = features
        self.drop_original = drop_original
        self.clip_negative = clip_negative
        self.prefix = prefix

    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame):
        X_out = X.copy()

        created = []
        for feature in self.features:
            if feature not in X_out.columns:
                continue
        
            vals = pd.to_numeric(X_out[feature], errors="coerce")
            
            if self.clip_negative:
                vals = vals.clip(lower=0)

            new_col = f"{self.prefix}{feature}"
            X_out[new_col] = np.log1p(vals)
            created.append(feature)
        
        if self.drop_original:
            X_out = X_out.drop(columns=created, errors="ignore")
        
        return X_out
    

class LotFrontageNeighborhoodImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing LotFrontage values using neighborhood median.
    
    Strategy: Missing LotFrontage values are filled with the median LotFrontage
    of the same Neighborhood. If neighborhood median is also NaN, uses global median.
    Also creates a binary indicator 'LotFrontage_is_missing' for the model.
    """
    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.global_median_ = X['LotFrontage'].median()
        self.nb_median_ = X.groupby('Neighborhood')['LotFrontage'].median()
        
        return self
    
    def transform(self, X: pd.DataFrame):
        X_out = X.copy()

        X_out['LotFrontage_is_missing'] = X_out['LotFrontage'].isna().astype(int)
        X_out['LotFrontage'] = X_out['LotFrontage'].fillna(X_out['Neighborhood'].map(self.nb_median_))
        X_out['LotFrontage'] = X_out['LotFrontage'].fillna(self.global_median_)
        
        return X_out
    
class MeaningfullNAImputer(BaseEstimator, TransformerMixin):
    """
    Fill NaN values with "NA" for features where missing means absence of feature.
    
    Many housing features (Alley, Pool, Garage, etc.) have NaN values that actually
    mean "not applicable" or "none" - e.g., no alley access, no pool.
    This transformer fills those with "NA" string for proper categorical encoding.
    
    Typical features: Alley, BsmtQual, BsmtCond, FireplaceQu, GarageType, PoolQC, Fence, MiscFeature
    """
    def __init__(self, features: Sequence[str]):
        self.features = features
        self.null_value = "NA"

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame):
        X_out = X.copy()

        for feature in self.features:
            if feature in X_out.columns:
                X_out[feature] = X_out[feature].fillna(self.null_value)
        
        return X_out
    

class BooleanFeaturesImputer(BaseEstimator, TransformerMixin):
    """
    Create boolean indicator features from existing columns.
    
    Creates the following binary features:
        - HasRemod: True if house was remodeled (YearRemodAdd > YearBuilt)
        - HasFireplace: True if Fireplaces > 0
        - HasGarage: True if GarageType != "NA"
        - HasPool: True if PoolQC != "NA"
        - HasFence: True if Fence != "NA"
        - HasMiscFeature: True if MiscFeature != "NA"
        - IsNormalSaleCondition: True if SaleCondition == "Normal"
        - HasBasement: True if BsmtQual != "NA"
        - Has2ndFloor: True if 2ndFlrSF > 0
    
    Note: Requires MeaningfullNAImputer to run first to fill NA values.
    """
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        X_out['HasRemod'] = X_out.apply((lambda x: x['YearRemodAdd'] > x['YearBuilt']), axis=1)
        X_out['HasFireplace'] = X_out['Fireplaces'].ne(0)
        X_out['HasGarage'] = X_out['GarageType'].ne("NA")
        X_out['HasPool'] = X_out['PoolQC'].ne("NA")
        X_out['HasFence'] = X_out['Fence'].ne("NA")
        X_out['HasMiscFeature'] = X_out['MiscFeature'].ne("NA")
        X_out['IsNormalSaleCondition'] = X_out['SaleCondition'].eq("Normal")
        X_out['HasBasement'] = X_out['BsmtQual'].ne("NA")
        X_out['Has2ndFloor'] = X_out['2ndFlrSF'].ne(0)

        return X_out


class SFImputer(BaseEstimator, TransformerMixin):
    """
    Create aggregate square footage features.
    
    Creates:
        - FloorTotalSF: Sum of 1stFlrSF + 2ndFlrSF (above ground living area)
        - TotalSF: FloorTotalSF + TotalBsmtSF (total finished square footage)
    
    These are strong predictors as larger homes generally command higher prices.
    """
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        first = X_out['1stFlrSF'].fillna(0)
        second = X_out['2ndFlrSF'].fillna(0)
        bsmt = X_out['TotalBsmtSF'].fillna(0)

        X_out['FloorTotalSF'] = first + second
        X_out['TotalSF'] = X_out['FloorTotalSF'] + bsmt

        return X_out


class GarageFeaturesImputer(BaseEstimator, TransformerMixin):
    """
    Handle garage-related features and create derived metrics.
    
    Handles:
        - GarageYrBlt: Fill missing with YearBuilt (assume garage built with house)
        - GarageAreaPerCar: Calculated ratio (0 if no garage)
        - GarageArea/GarageCars: Fill remaining NaN with 0
    """
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()
        X_out['GarageYrBlt'] = X_out['GarageYrBlt'].fillna(X_out['YearBuilt'])

        # Calculate area per car, handling division by zero (no garage = 0 cars)
        ratio = X_out['GarageArea'] / X_out['GarageCars'].replace({0: np.nan})
        X_out['GarageAreaPerCar'] = ratio.fillna(0)
        X_out['GarageArea'] = X_out['GarageArea'].fillna(0.0)
        X_out['GarageCars'] = X_out['GarageCars'].fillna(0.0)

        return X_out


class BsmtBathImputer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()
        X_out["BsmtFullBath"] = X_out["BsmtFullBath"].fillna(0.0)
        X_out["BsmtHalfBath"] = X_out["BsmtHalfBath"].fillna(0.0)
        X_out['TotalBsmtBath'] = X_out['BsmtFullBath'] + X_out['BsmtHalfBath']

        return X_out
    

class MasVnrAreaImputer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()

        if ("MasVnrArea" in X_out.columns) and ("MasVnrType" in X_out.columns):
            mask = X_out["MasVnrType"] == "NA"
            X_out.loc[mask, "MasVnrArea"] = X_out.loc[mask, "MasVnrArea"].fillna(0)
        
        # Fill remaining NaN
        if ("MasVnrArea" in X_out.columns):
            X_out["MasVnrArea"] = X_out["MasVnrArea"].fillna(0)

        return X_out


class HousingOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categories: dict, drop_base_features: bool = True, fill_value: str = "NA"):
        self.categories = categories
        self.drop_base_features = drop_base_features
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HousingOrdinalEncoder expects a pandas DataFrame.")

        self.features_ = list(self.categories.keys())
        self.ordinal_cols_ = [f"{c}_ord" for c in self.features_]

        X_enc = X[self.features_].copy()

        # Fill missing with NA so it's treated as a real level when you included "NA" in categories
        X_enc = X_enc.fillna(self.fill_value)

        self.ordinal_encoder_ = OrdinalEncoder(
            categories=[self.categories[c] for c in self.features_],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        self.ordinal_encoder_.fit(X_enc)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HousingOrdinalEncoder expects a pandas DataFrame.")

        X_out = X.copy()
        X_enc = X_out[self.features_].copy().fillna(self.fill_value)

        encoded = self.ordinal_encoder_.transform(X_enc)
        X_out[self.ordinal_cols_] = encoded

        if self.drop_base_features:
            X_out = X_out.drop(columns=self.features_, errors="ignore")

        return X_out


class HousingNominalOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        features: list[str],
        drop_base_features: bool = True,
        fill_value: str = "NA",
        prefix_sep: str = "__",
        sparse_output: bool = False,
    ):
        self.features = features
        self.drop_base_features = drop_base_features
        self.fill_value = fill_value
        self.prefix_sep = prefix_sep
        self.sparse_output = sparse_output

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HousingNominalOneHotEncoder expects a pandas DataFrame.")

        self.features_ = [c for c in self.features if c in X.columns]

        X_enc = X[self.features_].copy().fillna(self.fill_value)

        self.onehot_ = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=self.sparse_output
        )
        self.onehot_.fit(X_enc)

        self.onehot_feature_names_ = self.onehot_.get_feature_names_out(self.features_)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HousingNominalOneHotEncoder expects a pandas DataFrame.")

        X_out = X.copy()
        X_enc = X_out[self.features_].copy().fillna(self.fill_value)

        encoded = self.onehot_.transform(X_enc)

        # Convert to dense for DataFrame columns (even if sparse_output=True, we can densify here)
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()

        encoded_df = pd.DataFrame(
            encoded,
            columns=[name.replace("_", self.prefix_sep, 1) for name in self.onehot_feature_names_],
            index=X_out.index
        )

        X_out = pd.concat([X_out, encoded_df], axis=1)

        if self.drop_base_features:
            X_out = X_out.drop(columns=self.features_, errors="ignore")

        return X_out
    

class PolyFeaturesImputer(BaseEstimator, TransformerMixin):
    def __init__(self, features: list, degree: int):
        self.features = features
        self.degree = degree

    def fit(self, X: pd.DataFrame, y=None):
        X_poly = X[self.features].copy()
        
        self.poly_features = PolynomialFeatures(degree=self.degree)
        self.poly_features.fit(X_poly)
        self.new_cols = self.poly_features.get_feature_names_out(self.features)

        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_poly = X_out[self.features].copy().fillna(0.0)
        X_out[self.new_cols] = self.poly_features.transform(X_poly)
        
        return X_out


class RobustScalerDF(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.scaler = RobustScaler(**kwargs)

    def fit(self, X, y=None):
        self.cols_ = X.columns
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        arr = self.scaler.transform(X)
        return pd.DataFrame(arr, columns=self.cols_, index=X.index)

