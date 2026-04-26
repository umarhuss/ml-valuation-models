# Predicting Financial Markets: Stocks and UK Property

Dissertation codebase comparing machine-learning approaches to predicting returns in two distinct asset classes — equity markets and UK residential property — using a shared expanding-window cross-validation framework and SHAP-based interpretability.

## Research Overview

The project asks whether systematic, feature-rich ML models can produce economically meaningful out-of-sample forecasts in markets that are traditionally difficult to predict:

1. **Stocks pipeline** — predicts 12-month excess returns for a basket of ~100 US and UK equities relative to a market benchmark. Models are evaluated on whether top-decile picks outperform bottom-decile picks (decile spread) and by ROC-AUC and Rank IC.

2. **Property pipeline** — predicts next-month local authority (LA) house price growth across England using hedonic and macroeconomic features. Models are evaluated by regression (RMSE, MAE, R²) and binary classification (AUC, Brier score) on whether price growth will be positive.

Both pipelines use **expanding-window cross-validation** (one year held out at a time, training set grows) to respect temporal ordering and prevent lookahead bias.

## Methodology

### Stocks

| Step | What happens |
|------|-------------|
| Data collection | ~100 US/UK tickers via `yfinance`; macro series (VIX, DXY, US10Y, Brent) via `fredapi`; Google Trends + Wikipedia search volumes via `pytrends` |
| Target construction | 12-month excess log return vs. index benchmark; binary label = outperform |
| Features | Price momentum (1m, 3m, 6m, 12m), volatility, macro levels, Google Trends z-scores, Wikipedia spike flags |
| Models | Logistic Regression (L2), XGBoost classifier + regressor |
| Evaluation | ROC-AUC, Rank IC (Spearman), top–bottom decile spread; reported separately for US and UK |
| Interpretability | SHAP TreeExplainer — global feature importance and SHAP family groupings |

### Property

| Step | What happens |
|------|-------------|
| Data collection | HM Land Registry Price Paid Data (PPD); ONSPD postcode-to-LA mapping; ONS rental index (IPHRP); IMD deprivation scores; NAPTAN transport stops; Airbnb listings; OpenStreetMap amenities |
| Feature engineering | Hedonic price indices per LA per month; YoY growth rates; geospatial features (stops per km², fast-food density, Airbnb density); IMD z-scores |
| Targets | Next-month HPI YoY growth (regression) and positive-growth binary flag (classification); two variants — CORE (no rent) and WITHRENT |
| Models | Linear Regression / Logistic Regression (L2), XGBoost regressor + classifier; 1-month feature lag to avoid leakage |
| Evaluation | RMSE, MAE, R², ROC-AUC, Brier score, Average Precision; OOF (out-of-fold) predictions aggregated across all expanding folds |
| Interpretability | SHAP TreeExplainer with static and time-varying context features |

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data processing | pandas, numpy, pyarrow |
| Geospatial | geopandas, shapely |
| Machine learning | scikit-learn, xgboost |
| Interpretability | shap |
| Stock data | yfinance, pandas-datareader |
| Macro data | fredapi |
| Trends data | pytrends |
| Visualisation | matplotlib, seaborn |
| Environment | Python 3.11, conda (see `environment.yml`) |

## Repository Structure

```
.
├── stocks/
│   ├── notebooks/          # 01b–06: end-to-end stocks pipeline
│   └── data/               # not tracked — see Data Sources below
│
├── property/
│   ├── notebooks/          # 01P–08P: end-to-end property pipeline
│   └── data/               # not tracked — see Data Sources below
│
├── common/                 # shared utilities
│   ├── geo_utils.py        # CRS reprojection, spatial joins, distance helpers
│   ├── io_utils.py         # parquet read/write helpers
│   └── model_utils.py      # expanding-fold builder, Rank IC, decile spread, safe AUC
│
├── notebooks/              # top-level / legacy exploration notebooks
├── src/                    # early-stage source modules (cleaning, features, data loading)
├── environment.yml         # conda environment (Python 3.11)
└── .env                    # not tracked — add FRED_API_KEY here
```

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate diss
```

### 2. Add API keys

Create `.env` in the project root:

```
FRED_API_KEY=your_key_here
```

Obtain a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html). Required by `04b_features.ipynb` for macro data.

### 3. Obtain raw data

Data files are not committed to the repository. Download them and place them under the paths the notebooks expect:

#### Stocks pipeline
All data is fetched automatically at runtime (`yfinance`, `fredapi`, `pytrends`). No manual download required.

#### Property pipeline

| File | Source | Place at |
|------|--------|----------|
| Price Paid Data (complete) | [HM Land Registry](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) | `property/data/raw/ppd/` |
| ONSPD postcode directory | [ONS Geography](https://geoportal.statistics.gov.uk/) | `property/data/raw/onspd/` |
| LAD boundaries (GeoPackage) | [ONS Open Geography](https://geoportal.statistics.gov.uk/) | `property/data/raw/boundaries/` |
| IMD deprivation (England) | [MHCLG](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019) | `property/data/raw/imd/` |
| NAPTAN stops | [DfT / NaPTAN](https://www.data.gov.uk/dataset/ff93ffc1-6d58-47f7-8a58-6e6e73e73bee/national-public-transport-access-nodes-naptan) | `property/data/raw/naptan/` |
| Private Rental Market Statistics | [ONS](https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/privaterentalmarketsummarystatisticsengland) | `property/data/raw/rental/` |
| Inside Airbnb (London + Manchester) | [insideairbnb.com](http://insideairbnb.com/get-the-data/) | `property/data/raw/airbnb/` |
| OSM amenities (England) | [Geofabrik / Overpass](https://download.geofabrik.de/europe/great-britain/england.html) | `property/data/raw/amenities/` |

## Running the Pipelines

Notebooks must be run **in order** — each notebook produces outputs consumed by the next.

### Stocks pipeline

```
01b_exploration.ipynb       → initial data exploration
02b_returnsbench.ipynb      → compute monthly returns and benchmark series
03b_target_splits.ipynb     → construct 12-month targets and train/val/test splits
04b_features.ipynb          → feature engineering (momentum, macro, trends)  ← needs FRED_API_KEY
05b.ipynb                   → model training, expanding-window CV, SHAP export
06_stocks_shap_analysis.ipynb → extended SHAP visualisations
```

### Property pipeline

```
01P_data_ingest.ipynb       → load and clean PPD, ONSPD, boundaries
02P_clean_join.ipynb        → join transactions → LA codes → monthly summaries
03P_features_core.ipynb     → hedonic price indices, geospatial and contextual features
04P.ipynb                   → additional feature engineering
05P_models.ipynb            → expanding-window CV, OOF predictions, SHAP export
06P_Model_Analysis.ipynb    → leaderboard and model comparison
07P.ipynb                   → SHAP deep-dive
08P_Dissertation_Figures.ipynb → all figures for the dissertation
```

## Key Outputs

All generated outputs land in `reports/` (created at runtime):

| Output | Path |
|--------|------|
| Stocks CV metrics | `reports/stocks/tables/stocks_model_cv_v3_1b.parquet` |
| Stocks SHAP importance | `reports/stocks/tables/shap_importance_v3_1b_*.parquet` |
| Property OOF predictions | `reports/property/tables/oof_PROP_v1_ALL_property.parquet` |
| Property leaderboard | `reports/property/tables/leaderboard_PROP_v1.parquet` |
| Dissertation figures | `reports/property/figures/` |

## Notes

- **Memory:** the property pipeline processes ~25M land registry transactions; 8–16 GB RAM recommended.
- **Rate limits:** `pytrends` has undocumented rate limits; the feature notebook includes delays between requests.
- **Kernel restarts:** always re-run all cells from the top — variables are not persisted between sessions.
- **Feature variants:** the property pipeline produces two model variants, `CORE` (price-only features) and `WITHRENT` (adds private rental index dynamics). Both are evaluated and compared.

## Data Sources

| Dataset | Provider | Licence |
|---------|----------|---------|
| Price Paid Data | HM Land Registry | [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/) |
| Postcode directory (ONSPD) | Office for National Statistics | Open Government Licence |
| IMD 2019 | Ministry of Housing, Communities & Local Government | Open Government Licence |
| NAPTAN transport stops | Department for Transport | Open Government Licence |
| Private rental index (IPHRP) | ONS | Open Government Licence |
| Airbnb listings | Inside Airbnb | Creative Commons CC0 |
| Stock prices | Yahoo Finance via `yfinance` | Yahoo Terms of Service |
| Macroeconomic series | Federal Reserve (FRED) | Public domain |
| Search trends | Google Trends via `pytrends` | Google Terms of Service |
