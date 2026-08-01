"""
services/trend_engine.py — Trend Classification Engine
======================================================
Classifies financial metric trends from historical series.
"""

import logging
from enum import Enum
from typing import List, Optional

from services.metric_utils import cagr, yoy_growth

logger = logging.getLogger(__name__)


class TrendDirection(str, Enum):
    GROWING = "Growing"
    DECLINING = "Declining"
    STABLE = "Stable"
    ACCELERATING = "Accelerating"
    DECELERATING = "Decelerating"
    EXPANDING = "Expanding"
    CONTRACTING = "Contracting"
    INCREASING = "Increasing"
    DECREASING = "Decreasing"
    FLAT = "Flat"
    IMPROVING = "Improving"
    WEAKENING = "Weakening"
    UNAVAILABLE = "Unavailable"


class TrendEngine:
    """Classifies trends for growth, margin, debt, and cash flow metrics."""

    STABLE_THRESHOLD_PCT = 3.0

    def classify_growth(self, series: List[float]) -> TrendDirection:
        """Revenue, profit, EPS, dividend growth trends."""
        if len(series) < 2:
            return TrendDirection.UNAVAILABLE

        yoy = yoy_growth(series)
        full_cagr = cagr(series)

        if yoy is None:
            return TrendDirection.UNAVAILABLE

        if abs(yoy) < self.STABLE_THRESHOLD_PCT:
            return TrendDirection.STABLE

        if len(series) >= 3:
            recent_yoy = yoy
            prior_yoy = yoy_growth(series[1:])
            if prior_yoy is not None:
                if recent_yoy > prior_yoy + 2 and recent_yoy > 0:
                    return TrendDirection.ACCELERATING
                if recent_yoy < prior_yoy - 2 and recent_yoy > 0:
                    return TrendDirection.DECELERATING

        if full_cagr is not None and full_cagr > self.STABLE_THRESHOLD_PCT:
            return TrendDirection.GROWING
        if full_cagr is not None and full_cagr < -self.STABLE_THRESHOLD_PCT:
            return TrendDirection.DECLINING
        return TrendDirection.STABLE

    def classify_margin(self, series: List[float]) -> TrendDirection:
        """Operating / net margin trends."""
        if len(series) < 2:
            return TrendDirection.UNAVAILABLE

        yoy = yoy_growth(series)
        if yoy is None:
            return TrendDirection.UNAVAILABLE

        if abs(yoy) < 1.0:
            return TrendDirection.STABLE
        if yoy > 0:
            return TrendDirection.EXPANDING
        return TrendDirection.CONTRACTING

    def classify_debt(self, series: List[float]) -> TrendDirection:
        """Debt level trends."""
        if len(series) < 2:
            return TrendDirection.UNAVAILABLE

        yoy = yoy_growth(series)
        if yoy is None:
            return TrendDirection.UNAVAILABLE

        if abs(yoy) < self.STABLE_THRESHOLD_PCT:
            return TrendDirection.FLAT
        if yoy > 0:
            return TrendDirection.INCREASING
        return TrendDirection.DECREASING

    def classify_cash_flow(self, series: List[float]) -> TrendDirection:
        """Operating / free cash flow trends."""
        if len(series) < 2:
            return TrendDirection.UNAVAILABLE

        yoy = yoy_growth(series)
        if yoy is None:
            return TrendDirection.UNAVAILABLE

        if abs(yoy) < self.STABLE_THRESHOLD_PCT:
            return TrendDirection.STABLE
        if yoy > 0:
            return TrendDirection.IMPROVING
        return TrendDirection.WEAKENING

    def classify_for_metric(self, metric_key: str, series: List[float]) -> TrendDirection:
        """Dispatches to the appropriate classifier by metric type."""
        margin_keys = {"operating_margin", "net_margin", "fcf_margin"}
        debt_keys = {"debt", "debt_to_equity", "total_debt"}
        cash_keys = {"operating_cash_flow", "free_cash_flow", "cash_flow_growth"}

        if metric_key in margin_keys:
            return self.classify_margin(series)
        if metric_key in debt_keys:
            return self.classify_debt(series)
        if metric_key in cash_keys:
            return self.classify_cash_flow(series)
        return self.classify_growth(series)
