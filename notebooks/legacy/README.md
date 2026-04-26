# Legacy Notebooks - Original 10-Stock Universe

This folder contains the **original notebooks** from the initial implementation that used a **10-stock universe**.

## Important Notes

-   **These notebooks are ARCHIVED and NOT actively used**
-   They reference data files **without the `_1b` suffix** (10-stock versions)
-   The **current, active notebooks** are located in: `stocks/notebooks/`
-   Current notebooks use the **100-stock universe** and files with `_1b` suffix

## Notebook Versions

| Legacy (this folder)                       | Current (stocks/notebooks/) |
| ------------------------------------------ | --------------------------- |
| `01_exploration.ipynb`                     | `01b_exploration.ipynb`     |
| `02_returns_benchmark.ipynb`               | `02b_returnsbench.ipynb`    |
| `03_Targets_splits.ipynb`                  | `03b_target_splits.ipynb`   |
| `04_Features.ipynb`                        | `04b_features.ipynb`        |
| `5_baseline.ipynb` / `5_advancement.ipynb` | `05b.ipynb`                 |

## Data Files

-   Legacy notebooks use: `features_stocks_v3.parquet`, `stocks_monthly_targets_splits.parquet` (10 stocks)
-   Current notebooks use: `features_stocks_v3_1b.parquet`, `stocks_monthly_targets_splits_1b.parquet` (100 stocks)
