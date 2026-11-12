# House Prices: Advanced Regression Techniques — DAM2026 exam project

**Competition:** [House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Task
Predict the sale price of houses in Ames, Iowa from 79 features describing each house
(size, quality, location, year built, ...). Train: 1 460 houses, test: 1 459 houses.

**Target:** `SalePrice` (US dollars). Models are trained on `log(1 + SalePrice)`.

**Metric:** RMSE between the logarithms of the predicted and actual prices. Errors are measured
relative to the price, so a $20 000 error on a cheap house counts more than on an expensive one.

## Team
| member | role |
|---|---|
| Majba ull Alam Akash | preprocessing, feature engineering, models, tuning of Lasso/SVR/KernelRidge, stacking blend (`House_Price_2.ipynb`), Kaggle submission |
| Kofi Opoku Ayim | data description, EDA, missing values, outliers, baseline (`notebooks/01_eda_baseline.ipynb`); README |
| Omisore Omolola Bimpe | leakage-free validation, effect of the engineered features, model comparison (`notebooks/03_validation_features.ipynb`) |

## Results (5-fold CV, RMSE on log price)
| model | CV RMSE |
|---|---|
| constant (mean log price) | 0.3988 |
| rule: median by Neighborhood | 0.2690 |
| rule: median by OverallQual | 0.2305 |
| Kernel ridge (RBF) | 0.1527 |
| Hist gradient boosting | 0.1202 |
| LightGBM | 0.1135 |
| Gradient boosting | 0.1126 |
| ElasticNet | 0.1121 |
| Lasso | 0.1120 |
| XGBoost | 0.1120 |
| SVR | 0.1108 |
| **Stacking blend** (Lasso 0.19, SVR 0.39, GBR 0.14, XGBoost 0.29) | **0.1063** |

The blend cuts the error of the constant baseline by about 3.7 times and of the best simple rule
by about half.

### Leakage-free validation (notebook 03)
With all preprocessing fitted inside each CV fold (instead of on train + test together), the linear
models change by only about 0.0003, so the results above are reliable. A fragile Box-Cox rule
("only columns with more than 5 distinct values") left `PoolArea` untransformed in some folds and
broke SVR for houses with a pool (0.1217); with the fixed rule SVR scores 0.1103. On the same
leakage-free validation the stacking blend reaches 0.1058. Total-area features (TotalSF, TotalBath,
TotalPorchSF) are the most useful engineered features.

Kaggle public leaderboard: …

## Repository
- `notebooks/01_eda_baseline.ipynb`: data description, EDA, missing values, outliers, baseline
- `notebooks/03_validation_features.ipynb`: leakage-free validation, effect of the engineered features, comparison of all models
- `House_Price_2.ipynb`: preprocessing, feature engineering, models, tuning, stacking, submission
- `requirements.txt`: Python packages

## References
Preprocessing and model settings in `House_Price_2.ipynb` are based on the public Kaggle notebook
"Stacked Regressions: Top 4% on LeaderBoard" by Serigne
(https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard).
