"""
services/consistency_engine.py — Historical Consistency Engine
===============================================================
Evaluates the stability and trend quality of a company's fundamentals
over the available 3–5 year history.

Methodology redesign:
    Previous version used coefficient of variation (CV) alone, which
    incorrectly penalised companies on an improving trajectory.

    New approach:
        - Stability score    : rewards low volatility (CV-based)
        - Trend direction    : rewards improving trends separately
        - Growth CAGR quality: long-term compounding of revenue/EPS
        - Cyclical mode      : lenient CV thresholds for capital-intensive
                               industries where volatility is structural
"""

import logging
import statistics
from typing import List, Optional

import numpy as np

from services.data_cleaner import CleanedFinancialData
from services.ratio_calculator import CalculatedRatios

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _safe_cv(values: List[float]) -> Optional[float]:
    """Coefficient of variation (σ / |μ|). None if not enough data or μ ≈ 0."""
    if len(values) < 2:
        return None
    mean = float(np.mean(values))
    if abs(mean) < 0.001:
        return None
    return float(np.std(values, ddof=1)) / abs(mean)


def _trend_slope_normalized(values: List[float]) -> Optional[float]:
    """
    Computes a normalized linear slope:
        slope = (last − first) / ((n−1) × |mean|)
    Returns None if insufficient data or mean ≈ 0.
    Oldest values come first (series reversed before calling).
    """
    if len(values) < 2:
        return None
    mean = float(np.mean(values))
    if abs(mean) < 0.001:
        return None
    n = len(values)
    slope = (values[-1] - values[0]) / ((n - 1) * abs(mean))
    return slope


def _score_cv(cv: float, is_cyclical: bool = False) -> float:
    """CV → stability score 0–10. Cyclical industries get lenient tiers."""
    if is_cyclical:
        if cv < 0.15:  return 10.0
        if cv < 0.30:  return 8.5
        if cv < 0.50:  return 7.0
        if cv < 0.75:  return 5.5
        if cv < 1.10:  return 4.0
        return 2.5
    else:
        if cv < 0.10:  return 10.0
        if cv < 0.20:  return 8.5
        if cv < 0.35:  return 7.0
        if cv < 0.55:  return 5.0
        if cv < 0.80:  return 3.0
        return 1.0


def _score_trend(slope: Optional[float]) -> float:
    """Normalized slope → trend score 0–10. Positive slope is rewarded."""
    if slope is None:
        return 5.0  # neutral if unknown
    if slope > 0.20:    return 10.0
    if slope > 0.10:    return 9.0
    if slope > 0.03:    return 8.0
    if slope > -0.03:   return 6.5   # roughly flat
    if slope > -0.08:   return 4.5
    if slope > -0.15:   return 2.5
    return 1.0


def _score_growth_cagr(cagr_pct: Optional[float], is_cyclical: bool = False) -> Optional[float]:
    """Maps a multi-year CAGR to a consistency quality score 0–10."""
    if cagr_pct is None:
        return None
    if is_cyclical:
        if cagr_pct > 12:   return 10.0
        if cagr_pct > 6:    return 8.0
        if cagr_pct > 0:    return 6.5
        if cagr_pct > -5:   return 5.0
        return 2.5
    else:
        if cagr_pct > 15:   return 10.0
        if cagr_pct > 10:   return 8.5
        if cagr_pct > 5:    return 7.0
        if cagr_pct > 0:    return 5.5
        if cagr_pct > -5:   return 3.5
        return 1.5


# ─────────────────────────────────────────────────────────────────────
# ConsistencyEngine
# ─────────────────────────────────────────────────────────────────────

class ConsistencyEngine:
    """
    Evaluates historical consistency and trend quality of fundamentals.

    Scoring model (6 sub-components, dynamically weighted by data availability):

        ROE stability        (CV-based)               20%
        ROE trend direction  (slope-based)             15%
        Margin stability     (CV-based)                20%
        Profit growth CAGR   (level-based)             20%
        Revenue growth CAGR  (level-based)             15%
        EPS growth consistency (positive YoY count)    10%

    Cyclical mode:
        When is_cyclical=True, CV thresholds are relaxed and CAGR tiers
        are adjusted because volatility is structural, not a quality failure.
    """

    def evaluate(
        self,
        data: CleanedFinancialData,
        ratios: CalculatedRatios,
        is_cyclical: bool = False,
    ) -> float:
        """
        Returns a Consistency Score (0–10).

        A company with steadily improving fundamentals scores higher than a
        company that is merely stable at a mediocre level.
        """
        weights: List[float] = []
        scores: List[float] = []

        # ── 1. ROE Stability (CV) ──────────────────────────────────────
        roe_history = self._extract_roe_history(data)
        if len(roe_history) >= 3:
            cv = _safe_cv(roe_history)
            if cv is not None:
                weights.append(0.20)
                scores.append(_score_cv(cv, is_cyclical))

        # ── 2. ROE Trend Direction ─────────────────────────────────────
        # Reverse newest-first list so slope reads oldest→newest
        if len(roe_history) >= 2:
            slope = _trend_slope_normalized(list(reversed(roe_history)))
            weights.append(0.15)
            scores.append(_score_trend(slope))

        # ── 3. Margin Stability (CV) ───────────────────────────────────
        op_margins = self._extract_margin_history(data)
        if len(op_margins) >= 3:
            cv_m = _safe_cv(op_margins)
            if cv_m is not None:
                weights.append(0.20)
                scores.append(_score_cv(cv_m, is_cyclical))
        elif len(op_margins) >= 2:
            # At least check whether margins are stable directionally
            slope_m = _trend_slope_normalized(list(reversed(op_margins)))
            weights.append(0.10)  # reduced weight with limited history
            scores.append(_score_trend(slope_m))

        # ── 4. Profit Growth CAGR ──────────────────────────────────────
        s = _score_growth_cagr(ratios.profit_growth, is_cyclical)
        if s is not None:
            weights.append(0.20)
            scores.append(s)

        # ── 5. Revenue Growth CAGR ─────────────────────────────────────
        s = _score_growth_cagr(ratios.revenue_growth, is_cyclical)
        if s is not None:
            weights.append(0.15)
            scores.append(s)

        # ── 6. EPS Consistency (fraction of years with YoY growth) ─────
        eps_series = data.eps_series  # newest first
        if len(eps_series) >= 3:
            # Count years with year-over-year EPS growth
            yoy_positive = sum(
                1 for i in range(len(eps_series) - 1)
                if eps_series[i + 1] != 0 and (eps_series[i] - eps_series[i + 1]) > 0
            )
            total_comparisons = len(eps_series) - 1
            fraction = yoy_positive / total_comparisons
            # Also check if EPS trend is improving
            eps_trend = _trend_slope_normalized(list(reversed(eps_series)))
            base_score = fraction * 10.0
            trend_bonus = max(_score_trend(eps_trend) - 6.5, 0) * 0.3  # partial trend bonus
            weights.append(0.10)
            scores.append(min(base_score + trend_bonus, 10.0))

        # ── 7. Cash flow durability and direction ─────────────────────
        ocf_series = data.operating_cash_flow_series
        if len(ocf_series) >= 3:
            cv_ocf = _safe_cv(ocf_series)
            if cv_ocf is not None:
                weights.append(0.10)
                scores.append(_score_cv(cv_ocf, is_cyclical))
            slope_ocf = _trend_slope_normalized(list(reversed(ocf_series)))
            weights.append(0.05)
            scores.append(_score_trend(slope_ocf))

        # ── 8. Free cash flow consistency ─────────────────────────────
        fcf_series = self._extract_fcf_history(data)
        if len(fcf_series) >= 3:
            positive_fraction = sum(1 for value in fcf_series if value > 0) / len(fcf_series)
            weights.append(0.10)
            scores.append(min(positive_fraction * 10.0 + 1.0, 10.0))

        if not scores:
            logger.debug("ConsistencyEngine: insufficient data, returning neutral 5.0")
            return 5.0

        total_w = sum(weights)
        raw = sum(s * w for s, w in zip(scores, weights)) / total_w
        result = round(min(max(raw, 0.0), 10.0), 2)
        logger.debug(
            "ConsistencyEngine: score=%.2f, dims=%d, cyclical=%s",
            result, len(scores), is_cyclical,
        )
        return result

    # ──────────────────────────────────────────────────────────────────
    # Series extractors
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_roe_history(data: CleanedFinancialData) -> List[float]:
        """Compute ROE for each year where equity > 0 (newest first)."""
        history: List[float] = []
        for np_, eq in zip(data.net_profit_series, data.total_equity_series):
            if eq > 0:
                history.append((np_ / eq) * 100.0)
        return history

    @staticmethod
    def _extract_margin_history(data: CleanedFinancialData) -> List[float]:
        """Compute operating margin for each year where revenue > 0 (newest first)."""
        history: List[float] = []
        for op, rev in zip(data.operating_income_series, data.revenue_series):
            if rev > 0:
                history.append((op / rev) * 100.0)
        return history

    @staticmethod
    def _extract_fcf_history(data: CleanedFinancialData) -> List[float]:
        """Compute free cash flow by year (newest first)."""
        history: List[float] = []
        for idx, ocf in enumerate(data.operating_cash_flow_series):
            capex = abs(data.capex_series[idx]) if idx < len(data.capex_series) else 0.0
            history.append(ocf - capex)
        return history
