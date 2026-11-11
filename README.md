# House Prices: Advanced Regression Techniques — DAM2026 exam project

Competition: [House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Task
Predict the sale price of houses in Ames, Iowa from 79 features describing each house
(size, quality, location, year built, ...). Train: 1 460 houses, test: 1 459 houses.

Target: SalePrice (US dollars). Models are trained on log(1 + SalePrice).

Metric: RMSE between the logarithms of the predicted and actual prices. Errors are measured
relative to the price, so a $20 000 error on a cheap house counts more than on an expensive one.

## Team
| member | role |
|---|---|
| Majba ull Alam Akash | preprocessing, models and weighted ensemble (House_Price_2.ipynb), Kaggle submission |
| Kofi Opoku Ayim | data description, EDA, missing values, outliers, baseline (notebooks/01_eda_baseline.ipynb); README |
| Omisore Omolola Bimpe | leakage-free validation, feature effect, hyperparameter tuning, model comparison (notebook 03) |

## Results (5-fold CV, RMSE on log price)
| model | CV RMSE |
|---|---|
| constant (mean log price) | 0.3988 |
| rule: median by Neighborhood | 0.2690 |
| rule: median by OverallQual | 0.2305 |
| Ridge | 0.1134 |
| Gradient boosting | 0.1124 |
| SVR | 0.1121 |
| ElasticNet | 0.1119 |
| Lasso | 0.1118 |
| Weighted ensemble | 0.1080 |

The models cut the error of the constant baseline by about 3.7 times and of the best simple rule
by about half.

Kaggle public leaderboard: …

## Repository
- notebooks/01_eda_baseline.ipynb: data description, EDA,
