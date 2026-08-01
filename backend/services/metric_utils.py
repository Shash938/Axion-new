"""
services/metric_utils.py — Shared Financial Metric Calculations
================================================================
Pure functions for YoY growth, CAGR over N periods, and series alignment.
Used by ratio_calculator, historical_engine, and trend_engine.
"""

import logging
import math
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def yoy_growth(series: List[float]) -> Optional[float]:
    """
    Year-over-year growth between the two most recent values.
    series: newest-first ordering.
    Returns percentage (e.g. 12.5 for 12.5% growth).
    """
    if len(series) < 2:
        return None
    latest, prior = series[0], series[1]
    if prior == 0:
        return None
    return round(((latest - prior) / abs(prior)) * 100, 2)


def cagr(series: List[float], periods: Optional[int] = None) -> Optional[float]:
    """
    Compound annual growth rate over `periods` years (or full series if None).
    series: newest-first. Requires periods+1 data points.
    """
    if not series or len(series) < 2:
        return None

    if periods is not None:
        if len(series) < periods + 1:
            return None
        latest = series[0]
        oldest = series[periods]
        n_years = periods
    else:
        latest = series[0]
        oldest = series[-1]
        n_years = len(series) - 1

    if oldest <= 0:
        return None

    try:
        rate = (math.pow(latest / oldest, 1.0 / n_years) - 1) * 100
        return round(rate, 2)
    except (ZeroDivisionError, ValueError, OverflowError):
        return None


def cagr_3y(series: List[float]) -> Optional[float]:
    return cagr(series, periods=3)


def cagr_5y(series: List[float]) -> Optional[float]:
    return cagr(series, periods=5)


def safe_divide(numerator: float, denominator: float, as_pct: bool = False) -> Optional[float]:
    """Returns None when denominator is zero."""
    if denominator == 0:
        return None
    result = numerator / denominator
    if as_pct:
        result *= 100
    return round(result, 2)


def align_series(
    *series_lists: List[float],
) -> Tuple[List[List[float]], int]:
    """
    Truncates multiple series to the length of the shortest one.
    Returns aligned copies and the common length.
    """
    if not series_lists:
        return [], 0
    min_len = min(len(s) for s in series_lists if s)
    if min_len == 0:
        return [[] for _ in series_lists], 0
    return [s[:min_len] for s in series_lists], min_len


def format_inr_crores(value: float) -> str:
    """Formats INR Crores for display in explanations."""
    if abs(value) >= 100000:
        return f"₹{value / 100000:.2f}L Cr"
    if abs(value) >= 1000:
        return f"₹{value / 1000:.1f}K Cr"
    return f"₹{value:,.0f} Cr"
