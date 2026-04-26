"""Common modelling helpers for cross-validation and metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


@dataclass
class FoldSpec:
    train_end: int
    val_year: int


def make_expanding_folds(years: Iterable[int], start_val_year: int, end_val_year: int) -> List[FoldSpec]:
    """Create expanding-window folds from integer years."""
    folds: List[FoldSpec] = []
    for val_year in range(start_val_year, end_val_year + 1):
        folds.append(FoldSpec(train_end=val_year - 1, val_year=val_year))
    return folds


def recency_weights(dates: pd.Series, half_life_months: int = 24) -> np.ndarray:
    """Exponential decay weights based on recency."""
    months_since = (dates.max() - dates).dt.days / 30.44
    lam = np.log(2) / half_life_months
    return np.exp(-lam * months_since)


def rank_ic(y_true: pd.Series, y_score: pd.Series) -> float:
    """Spearman rank correlation."""
    if y_true.nunique() <= 1:
        return float("nan")
    corr, _ = spearmanr(y_true, y_score)
    return float(corr)


def decile_spread(y_true: pd.Series, y_score: pd.Series, n_bins: int = 10) -> float:
    """Top-bottom decile return spread."""
    df = pd.DataFrame({"target": y_true, "score": y_score}).dropna()
    if df.empty:
        return float("nan")
    df["decile"] = pd.qcut(df["score"], q=n_bins, labels=False, duplicates="drop")
    top = df[df["decile"] == df["decile"].max()]["target"].mean()
    bottom = df[df["decile"] == df["decile"].min()]["target"].mean()
    return float(top - bottom)


def safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    """ROC AUC safe wrapper (handles single-class cases)."""
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))
