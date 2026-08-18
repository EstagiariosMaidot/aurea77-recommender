"""Temporal split helpers."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def temporal_cutoff(dates: pd.Series, percentile: float = 0.80) -> datetime:
    """Return the date at the given percentile of `dates`."""
    if dates.empty:
        raise ValueError("Cannot compute cutoff over an empty series.")
    sorted_dates = dates.sort_values().reset_index(drop=True)
    idx = int(np.clip(np.floor(percentile * (len(sorted_dates) - 1)), 0, len(sorted_dates) - 1))
    return sorted_dates.iloc[idx].to_pydatetime()


def temporal_split(
    df: pd.DataFrame, *, cutoff: datetime, date_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows strictly before `cutoff` → train; at/after → test."""
    train = df[df[date_col] < cutoff].copy()
    test = df[df[date_col] >= cutoff].copy()
    return train, test
