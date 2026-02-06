# House Prices - Advanced Regression Techniques

Kaggle competition solution for predicting residential home sale prices in Ames, Iowa.

**Competition Link:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Overview

This project implements a complete machine learning pipeline to predict house sale prices using 79 explanatory variables describing almost every aspect of residential homes in Ames, Iowa.

## Project Structure

```
house-prices/
├── README.md                 # This file
├── analysis.ipynb           # Exploratory Data Analysis notebook
├── model_creation.ipynb     # Model training and evaluation notebook
├── housing_pipeline.py      # Custom sklearn transformers
├── data/
│   ├── train.csv           # Training data (1460 houses)
│   ├── test.csv            # Test data for submission
│   ├── final.csv           # Cleaned training data after EDA
│   └── data_description.txt # Feature descriptions
└── submissions/
    ├── first_attempt.csv   # Ridge regression submission
    └── second_attempt.csv  # Stacked ensemble submission
```

## Approach

### 1. Exploratory Data Analysis (`analysis.ipynb`)

- **Target Analysis:** SalePrice is right-skewed; log transformation applied
- **Feature Categorization:** 
  - 34 numerical features (including datetime features)
  - 15 ordinal categorical features (quality ratings)
  - 31 nominal categorical features
- **Outlier Detection:** Removed ~20 extreme outliers based on visual inspection
- **Missing Value Analysis:** Identified meaningful NaN values (e.g., "No Pool" = PoolQC NaN)

### 2. Feature Engineering (`housing_pipeline.py`)

Custom sklearn transformers for:

| Transformer | Purpose |
|------------|---------|
| `Log1pFeatureImputer` | Log transform skewed features (LotArea, GrLivArea) |
| `LotFrontageNeighborhoodImputer` | Impute by neighborhood median |
| `MeaningfullNAImputer` | Fill "NA" for absence features (no pool, no garage) |
| `BooleanFeaturesImputer` | Create binary indicators (HasPool, HasGarage, etc.) |
| `SFImputer` | Aggregate square footage (TotalSF = FloorSF + BsmtSF) |
| `GarageFeaturesImputer` | Handle garage features + create GarageAreaPerCar |
| `BsmtBathImputer` | Create TotalBsmtBath feature |
| `HousingOrdinalEncoder` | Encode quality features (Po/Fa/TA/Gd/Ex → 0-5) |
| `HousingNominalOneHotEncoder` | One-hot encode nominal categories |
| `PolyFeaturesImputer` | Add polynomial interactions |


### 3. Final Solution

**Stacked Ensemble:**
- Base models: Ridge + GradientBoostingRegressor
- Meta-learner: Ridge regression
- Final RMSE: ~0.076 on training set

## Key Findings

1. **Top Predictors (by correlation with SalePrice):**
   - OverallQual (0.79)
   - GrLivArea (0.71)
   - GarageCars (0.64)
   - TotalBsmtSF (0.61)
   - FullBath (0.56)

2. **Important Insights:**
   - Quality features (OverallQual, ExterQual, KitchenQual) are strongest predictors
   - Location matters: Neighborhood has significant impact
   - Log transformation of target helps with right-skewed distribution
   - Polynomial features of quality/year/size improve performance

## Usage

```python
from housing_pipeline import (
    Log1pFeatureImputer,
    LotFrontageNeighborhoodImputer,
    MeaningfullNAImputer,
    BooleanFeaturesImputer,
    SFImputer,
    GarageFeaturesImputer,
    BsmtBathImputer,
    MasVnrAreaImputer,
    HousingOrdinalEncoder,
    HousingNominalOneHotEncoder,
    PolyFeaturesImputer
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# Build pipeline
pipeline = Pipeline([
    ("log_transform", Log1pFeatureImputer(["LotArea", "GrLivArea"])),
    ("lot_frontage", LotFrontageNeighborhoodImputer()),
    ("na_imputer", MeaningfullNAImputer(features)),
    ("boolean_features", BooleanFeaturesImputer()),
    ("sf_features", SFImputer()),
    ("garage_features", GarageFeaturesImputer()),
    ("ordinal_encoder", HousingOrdinalEncoder(categories)),
    ("onehot_encoder", HousingNominalOneHotEncoder(nominal_features)),
    ("model", Ridge(alpha=10.0))
])

# Train and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## Future Improvements

- [ ] Deep error analysis on high-residual predictions
- [ ] Advanced feature engineering (interaction terms, domain-specific features)
- [ ] Add XGBoost/LightGBM to ensemble
- [ ] Feature selection using SHAP values

## Results

| Submission | Model | Kaggle Score |
|------------|-------|--------------|
| first_attempt.csv | Ridge | 0.12198 |
| second_attempt.csv | Stacked (Ridge + GBR) | 0.11814 |

## Dependencies

See `requirements.txt` in the root directory.
