# Demand Forecasting & Inventory Optimization

A time-series machine-learning project for analyzing historical store-item sales, engineering demand features, and comparing baseline and tree-based forecasting models.

> **Current scope:** The repository currently implements sales analysis and demand forecasting experiments. Inventory replenishment optimization, on-hand inventory tracking, supplier lead-time modeling, and a dashboard/API are not yet implemented. The project title describes the broader objective, not a completed inventory optimization system.

## Overview

The current workflow includes:

1. Dataset inspection and data-quality checks.
2. Data cleaning and chronological sorting.
3. Exploratory analysis of sales over time, across stores and items, and by weekday/month.
4. Calendar, lag and rolling-average feature engineering.
5. Naive and seasonal-naive baseline forecasts.
6. Random Forest and XGBoost model training and evaluation.
7. Comparison using MAE, RMSE and WAPE.

## Dataset

The project uses the **Store Item Demand Forecasting Challenge** dataset. The training file, `train.csv`, has four columns:

| Column | Description |
|---|---|
| `date` | Sales observation date |
| `store` | Store identifier |
| `item` | Item/product identifier |
| `sales` | Units sold |

The audited training data contains **913,000 rows**, **10 stores**, and **50 items**, covering **2013-01-01 to 2017-12-31**.

Download the data from [Kaggle — Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) and place the files here:

```text
data/
└── raw/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

The current training notebooks use `data/raw/train.csv`. Dataset files and processed data are excluded from Git by `.gitignore`; download the data separately.

For comprehensive schema definitions, attribute ranges, data quality verification, and feature engineering details, refer to the [Dataset Documentation & Data Dictionary](reports/data_dictionary_and_dataset_documentation.md).

## Repository Structure

```text
.
├── notebooks/
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_forecasting.ipynb
│   ├── 06_random_forest_model.ipynb
│   ├── 07_xgboost_model.ipynb
│   └── 08_model_comparison.ipynb
├── reports/
│   ├── model_comparison.csv
│   ├── random_forest_feature_importance.csv
│   ├── test_baseline_results.csv
│   ├── validation_baseline_results.csv
│   ├── xgboost_feature_importance.csv
│   └── xgboost_test_predictions.csv
├── src/
│   └── test_environment.py
├── data/       # local only; ignored by Git
├── models/     # local only; ignored by Git
├── .gitignore
└── requirements.txt
```

## Notebook Workflow

Run the notebooks in numerical order:

| Notebook | Work performed |
|---|---|
| `01_dataset_understanding.ipynb` | Inspect dimensions, columns, types, summary statistics, missing values, duplicates, stores, items and date range |
| `02_data_cleaning.ipynb` | Parse dates, sort store-item-date records, inspect duplicates, sales values and date coverage; save cleaned data |
| `03_exploratory_data_analysis.ipynb` | Analyze sales distribution, daily/monthly trends, store/item performance, weekday and monthly patterns |
| `04_feature_engineering.ipynb` | Build calendar, lag and rolling-average predictors for each store-item series |
| `05_baseline_forecasting.ipynb` | Create chronological splits and evaluate naive and seasonal-naive baselines |
| `06_random_forest_model.ipynb` | Train/evaluate a Random Forest regressor and export feature importance |
| `07_xgboost_model.ipynb` | Train/evaluate XGBoost, export feature importance and test predictions |
| `08_model_comparison.ipynb` | Consolidate the recorded model metrics |

### Engineered features

- **Calendar:** `year`, `month`, `day`, `day_of_week`, `week_of_year`, `quarter`, `is_weekend`
- **Lags:** `lag_1`, `lag_7`, `lag_14`, `lag_28`
- **Trailing averages:** `rolling_mean_7`, `rolling_mean_28`

Lag and rolling features are calculated independently for each store-item pair. Rolling averages shift sales by one period before applying the rolling window, preventing the current target from entering its own rolling feature. The audited feature dataset contains **899,000 rows**, beginning on 2013-01-29 after the initial history required by the 28-day features is removed.

## Recorded Model Results

The committed `reports/model_comparison.csv` records these **test-set** results:

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| **XGBoost** | **6.260879** | **8.130000** | **10.197908%** |
| Random Forest | 6.453025 | 8.392493 | 10.510881% |
| Seasonal Naive | 9.214098 | 12.113728 | 15.008199% |
| Naive | 11.589739 | 15.863372 | 18.877715% |

Lower is better for all three metrics. XGBoost has the lowest recorded error among the four approaches.

### Chronological partitions

| Partition | Date range |
|---|---|
| Training | Through 2016-12-31 |
| Validation | 2017-01-01 to 2017-06-30 |
| Test | 2017-07-01 to 2017-12-31 |

Baseline validation and test metrics are also saved in `reports/validation_baseline_results.csv` and `reports/test_baseline_results.csv`.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/karanbhilare58/-Demand-Forecasting-Inventory.git
cd -Demand-Forecasting-Inventory
```

### 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install the core notebook dependencies

The committed `requirements.txt` is currently a broad environment export and contains machine-specific local package references. It is not yet a portable install specification. For the notebooks, install the core packages with:

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter ipykernel
```

### 4. Download the dataset

Download `train.csv` from the Kaggle link above and place it at `data/raw/train.csv`.

### 5. Launch Jupyter

```bash
jupyter notebook
```

Open the `notebooks/` directory and run notebooks in numerical order. Their relative paths expect the repository folder structure to be preserved. Create the `data/processed/`, `reports/`, and `models/` directories locally when needed.

## Modeling Notes and Limitations

- Feature generation and model training operate at **store-item-date** granularity. Some EDA and baseline visualizations aggregate sales across stores and items by date.
- The naive baseline uses lag-1 sales; the seasonal-naive baseline uses lag-7 sales.
- The experiments use lag and rolling features derived from historical observed sales. The recorded results should not be interpreted as validation of a recursive multi-day forecasting system.
- The source dataset contains sales history only. It does not provide on-hand inventory, stockout flags, supplier lead times, purchase orders, holding costs or service-level targets.
- Trained `.pkl` and `.joblib` model artifacts are excluded from Git. Recreate them by running the respective model notebooks.
- Notebook files are present in the repository, but run all notebooks from a clean environment before treating the complete workflow as reproducible.

### Next steps toward inventory optimization

1. Make dependencies reproducible and validate all notebooks end to end.
2. Add a reusable preprocessing/feature pipeline under `src/`.
3. Define and validate a multi-step forecasting strategy if multi-day predictions are needed.
4. Obtain inventory, supplier, lead-time, order and stockout data.
5. Define replenishment constraints and policies, including safety stock, service level, review period and order quantity.
6. Implement and validate inventory optimization.
7. Add a dashboard/API after the forecasting and replenishment logic is validated.

## Reports

The `reports/` directory includes:

- `model_comparison.csv` — recorded test metrics for the four approaches.
- `test_baseline_results.csv` — naive and seasonal-naive test metrics.
- `validation_baseline_results.csv` — naive and seasonal-naive validation metrics.
- `random_forest_feature_importance.csv` — Random Forest feature importance.
- `xgboost_feature_importance.csv` — XGBoost feature importance.
- `xgboost_test_predictions.csv` — row-level XGBoost test predictions.

## Author

GitHub: [@karanbhilare58](https://github.com/karanbhilare58)
