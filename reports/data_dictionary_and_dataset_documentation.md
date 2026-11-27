# Dataset Documentation & Data Dictionary

This document details the source, structure, schema, observed statistical properties, and data dictionary for the store-item sales dataset used in this project.

All statistics and boundaries recorded here have been directly verified against the local raw (`data/raw/train.csv`), cleaned (`data/processed/sales_cleaned.csv`), and feature-engineered (`data/processed/sales_features.csv`) datasets.

---

## 1. Dataset Overview & Provenance

- **Origin / Source:** Kaggle Competition: [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data).
- **Primary Training File:** `train.csv` (placed at `data/raw/train.csv`).
- **Domain:** Retail store sales time-series forecasting.
- **Coverage Period:** 2013-01-01 to 2017-12-31 (5 full calendar years, exactly 1,826 consecutive days).
- **Entities Covered:**
  - **Stores:** 10 distinct retail stores (identified by integer IDs 1 through 10).
  - **Items:** 50 distinct merchandise items/products (identified by integer IDs 1 through 50).
  - **Store-Item Combinations:** Exactly 500 distinct series ($10 \times 50$).
- **Total Records:** Exactly 913,000 rows ($500 \text{ series} \times 1,826 \text{ days}$).
- **Granularity:** One record per store, per item, per calendar day (`date`, `store`, `item`).

> **Boundary Note on Source Data:**
> The source dataset provides transaction counts (`sales`) only. It does **not** provide item descriptions, product categories, retail pricing, discounts, promotions, store locations, store square footage, inventory on hand, replenishment orders, supplier lead times, stockout events, or holding costs. No assumptions or fabrications are made regarding these missing operational variables.

---

## 2. Data Quality & Cleaning Audit

Verified data health checks performed on `data/raw/train.csv` and `data/processed/sales_cleaned.csv`:

1. **Missing Values:** Exactly 0 null / NaN values across all columns.
2. **Duplicate Records:** Exactly 0 duplicate records across all columns and across the composite key (`date`, `store`, `item`).
3. **Date Regularity & Completeness:** Every store-item combination possesses exactly 1,826 daily records without gaps or duplicate dates.
4. **Sales Value Range:**
   - Minimum observed sales: `0` (exactly 1 occurrence: store 6, item 4, date 2014-01-15).
   - Maximum observed sales: `231` (store 2, item 15, date 2017-07-23).
   - Negative values: `0` (no negative sales recorded).
   - Mean sales per store-item-day observation: `52.25` units (standard deviation: `28.80`).
5. **Cleaning Methodology Applied (`data/processed/sales_cleaned.csv`):**
   - Parsed `date` column from string/object format to datetime format (`pd.to_datetime`).
   - Confirmed 0 duplicate observations and complete date sequence.
   - Sorted rows chronologically by `store`, `item`, and `date`.
   - Resulting cleaned dataset retains all 913,000 rows with confirmed type alignment and ordering.

---

## 3. Data Dictionary: Original Dataset (Raw & Cleaned)

| Column Name | Storage Type (Raw) | Storage Type (Cleaned) | Range / Cardinality | Description & Inferred Semantics |
|---|---|---|---|---|
| `date` | `object` (string `YYYY-MM-DD`) | `datetime64[ns]` in memory / `object` on reload without `parse_dates` | `2013-01-01` to `2017-12-31` (1,826 unique days) | **Observation Date:** The calendar date on which sales occurred. Consecutive daily reporting. Converted to `datetime64[ns]` in memory during cleaning; serialized as ISO strings in CSV and reloaded as `object` unless `parse_dates=['date']` is specified. |
| `store` | `int64` | `int64` | Integers `1` to `10` (10 unique stores) | **Store Identifier:** Unique anonymous integer identifier for each store location. |
| `item` | `int64` | `int64` | Integers `1` to `50` (50 unique items) | **Item / Product Identifier:** Unique anonymous integer identifier for each item/SKU. |
| `sales` | `int64` | `int64` | `0` to `231` (mean: 52.25, median: 47.00) | **Units Sold:** The target variable representing quantity of units sold for the given store and item on the date. |

### Semantic Classification
- **Observed Values (Factual):** `date`, `store`, `item`, `sales`.
- **Inferred Interpretations:** Daily unit sales are assumed to represent unconstrained demand in this benchmark problem, because inventory stockout indicators and order cancellations are not available in the dataset.

---

## 4. Data Dictionary: Feature-Engineered Dataset (`data/processed/sales_features.csv`)

The feature dataset is constructed from `sales_cleaned.csv` to support machine learning models. Because 28-day lag and rolling-window features require 28 days of prior observations per series, the first 28 days (2013-01-01 through 2013-01-28) are dropped, leaving **899,000 rows** covering **2013-01-29 to 2017-12-31** (1,798 unique days).

| Column Name | Data Type | Value Range | Derivation / Feature Definition |
|---|---|---|---|
| `date` | `object` (`YYYY-MM-DD`) | `2013-01-29` to `2017-12-31` | Observation date. |
| `store` | `int64` | `1` to `10` | Store identifier. |
| `item` | `int64` | `1` to `50` | Item identifier. |
| `sales` | `int64` | `0` to `231` | Target variable: actual sales units sold on `date`. |
| `year` | `int64` | `2013` to `2017` | Calendar feature extracted from `date.year`. |
| `month` | `int64` | `1` to `12` | Calendar feature extracted from `date.month`. |
| `day` | `int64` | `1` to `31` | Calendar day extracted from `date.day`. |
| `day_of_week` | `int64` | `0` to `6` | Day of week (`0` = Monday, `6` = Sunday). |
| `week_of_year` | `int64` | `1` to `53` | ISO week number of the year extracted from `date`. |
| `quarter` | `int64` | `1` to `4` | Calendar quarter (`1` to `4`) extracted from `date.quarter`. |
| `is_weekend` | `int64` | `0` or `1` | Binary flag: `1` if `day_of_week` $\in \{5, 6\}$ (Saturday/Sunday), else `0`. |
| `lag_1` | `float64` | `0.0` to `231.0` | Actual sales for the same `(store, item)` exactly 1 day prior ($t-1$). |
| `lag_7` | `float64` | `0.0` to `231.0` | Actual sales for the same `(store, item)` exactly 7 days prior ($t-7$). Captures day-of-week seasonality. |
| `lag_14` | `float64` | `0.0` to `231.0` | Actual sales for the same `(store, item)` exactly 14 days prior ($t-14$). |
| `lag_28` | `float64` | `0.0` to `231.0` | Actual sales for the same `(store, item)` exactly 28 days prior ($t-28$). Captures 4-week monthly cycle. |
| `rolling_mean_7` | `float64` | `1.14` to `162.71` | 7-day trailing moving average of sales, computed over shifted sales ($t-1$ down to $t-7$) to prevent target leakage. |
| `rolling_mean_28` | `float64` | `1.64` to `147.25` | 28-day trailing moving average of sales, computed over shifted sales ($t-1$ down to $t-28$) to prevent target leakage. |

### Data Leakage Prevention Implementation
- For rolling features (`rolling_mean_7`, `rolling_mean_28`), sales are explicitly shifted by 1 period before applying `.rolling(window=...)`. Thus, target value $y_t$ is strictly excluded from feature values at time $t$.
- Lags and rolling means are grouped strictly by `['store', 'item']` so information never bleeds across different stores or items.
