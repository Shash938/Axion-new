"""
services/qualitative_engine.py — Qualitative Assessment Engine
===============================================================
Evaluates Moat, Capital Allocation, and Earnings Quality from
multi-year financial series rather than single-year snapshots.

Design philosophy:
    - Long-term consistency is rewarded over a single exceptional year.
    - Each dimension uses 3–5 years of history where available.
    - Scores degrade gracefully when history is short (< 3 years).
    - No hardcoded thresholds are sector-specific here; the scoring engine
      passes sector context via the SectorType when relevant.
"""

import logging
import statistics
from typing import List, Optional, Tuple

from services.data_cleaner import CleanedFinancialData
from services.ratio_calculator import CalculatedRatios

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────

def _safe_cv(values: List[float]) -> Optional[float]:
    """Coefficient of variation (std / mean). None if mean ≈ 0 or < 2 points."""
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    if abs(mean) < 0.001:
        return None
    return statistics.stdev(values) / abs(mean)


def _trend_slope_score(values: List[float]) -> float:
    """
    Returns a score 0–10 reflecting whether values trend up, flat, or down.
    Uses a simple linear regression slope relative to the mean magnitude.
    Scores:
        > +15% relative slope → 10 (strongly improving)
        > +5%                 → 8
        > 0%                  → 6 (flat / stable)
        > -5%                 → 4 (mild decline)
        > -15%                → 2 (deteriorating)
        worse                 → 0
    Requires at least 2 points; returns 5.0 (neutral) if unavailable.
    """
    n = len(values)
    if n < 2:
        return 5.0
    # Compute mean
    mean = statistics.mean(values)
    if abs(mean) < 0.001:
        return 5.0
    # Simple slope: (last - first) / (n - 1)  relative to mean
    slope = (values[-1] - values[0]) / ((n - 1) * abs(mean))
    if slope > 0.15:   return 10.0
    if slope > 0.05:   return 8.0
    if slope > -0.01:  return 6.0   # roughly flat
    if slope > -0.05:  return 4.0
    if slope > -0.15:  return 2.0
    return 0.0


def _series_zip(s1: List[float], s2: List[float]) -> List[Tuple[float, float]]:
    """Zip two newest-first series, taking only aligned pairs with s2 > 0."""
    return [(a, b) for a, b in zip(s1, s2) if b != 0]


def _positive_fraction(values: List[float]) -> float:
    """Fraction of values that are > 0. Returns 0 if empty."""
    if not values:
        return 0.0
    return sum(1 for v in values if v > 0) / len(values)


# ─────────────────────────────────────────────────────────────────────
# QualitativeEngine
# ─────────────────────────────────────────────────────────────────────

class QualitativeEngine:
    """
    Evaluates three qualitative dimensions using multi-year financial data:

    1. Moat Score           — Pricing power and competitive durability
    2. Capital Allocation   — Management's reinvestment effectiveness
    3. Earnings Quality     — Whether reported profits are backed by cash
    """

    # ──────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────

    def evaluate_moat(
        self,
        data: CleanedFinancialData,
        ratios: CalculatedRatios,
        is_cyclical: bool = False,
    ) -> float:
        """
        Moat Score (0–10) — Inferred from multi-year consistency of high returns.

        A genuine moat manifests as:
          - Persistently high ROE (≥ 15–20%) across multiple years
          - Stable or expanding operating margins
          - Consistent FCF generation relative to revenue
          - Sustained high ROCE

        Dimensions and weights:
          ROE mean level          25%  — absolute quality anchor
          ROE stability           20%  — consistency vs variance
          Operating margin mean   20%  — pricing power proxy
          Margin stability        15%  — durable advantage indicator
          FCF/Revenue consistency 15%  — self-funding quality
          ROCE mean (bonus)        5%  — capital efficiency confirmation
        """
        weights = []
        scores = []

        # ── 1. ROE mean level ─────────────────────────────────────────
        roe_history = self._roe_series(data)
        if roe_history:
            mean_roe = statistics.mean(roe_history)
            s = self._score_roe_mean(mean_roe)
            weights.append(0.25); scores.append(s)

            # ── 2. ROE stability (inverse CV) ─────────────────────────
            cv = _safe_cv(roe_history)
            if cv is not None:
                s2 = self._score_cv(cv, is_cyclical)
                weights.append(0.20); scores.append(s2)
        else:
            # Fallback to latest-year ROE from ratios
            if ratios.roe is not None:
                weights.append(0.25); scores.append(self._score_roe_mean(ratios.roe))

        # ── 3. Operating margin mean ──────────────────────────────────
        margin_history = self._margin_series(data)
        if margin_history:
            mean_margin = statistics.mean(margin_history)
            s = self._score_margin_mean(mean_margin)
            weights.append(0.20); scores.append(s)
            weights.append(0.05); scores.append(_trend_slope_score(list(reversed(margin_history))))

            # ── 4. Margin stability ───────────────────────────────────
            cv_m = _safe_cv(margin_history)
            if cv_m is not None:
                s2 = self._score_cv(cv_m, is_cyclical)
                weights.append(0.15); scores.append(s2)
        elif ratios.operating_margin is not None:
            weights.append(0.20)
            scores.append(self._score_margin_mean(ratios.operating_margin))

        # ── 5. FCF / Revenue consistency ─────────────────────────────
        fcf_rev_history = self._fcf_revenue_series(data)
        if fcf_rev_history:
            mean_fcf_rev = statistics.mean(fcf_rev_history)
            pos_frac = _positive_fraction(fcf_rev_history)
            # Combine: average FCF margin quality + fraction of positive years
            s = (self._score_fcf_margin(mean_fcf_rev) * 0.6
                 + pos_frac * 10.0 * 0.4)
            weights.append(0.15); scores.append(min(s, 10.0))
            cv_fcf = _safe_cv(fcf_rev_history)
            if cv_fcf is not None:
                weights.append(0.05); scores.append(self._score_cv(cv_fcf, is_cyclical))
        elif ratios.fcf_margin is not None:
            weights.append(0.15)
            scores.append(self._score_fcf_margin(ratios.fcf_margin))

        # ── 6. ROCE mean (bonus) ──────────────────────────────────────
        roce_history = self._roce_series(data)
        if roce_history:
            mean_roce = statistics.mean(roce_history)
            s = self._score_roce_mean(mean_roce)
            weights.append(0.05); scores.append(s)
        elif ratios.roce is not None:
            weights.append(0.05)
            scores.append(self._score_roce_mean(ratios.roce))

        if not weights:
            logger.debug("QualitativeEngine: no data for moat, returning neutral 5.0")
            return 5.0

        total_w = sum(weights)
        raw = sum(s * w for s, w in zip(scores, weights)) / total_w
        result = round(min(max(raw, 0.0), 10.0), 2)
        logger.debug("QualitativeEngine: Moat = %.2f (dims=%d)", result, len(weights))
        return result

    def evaluate_earnings_quality(
        self,
        data: CleanedFinancialData,
        ratios: CalculatedRatios,
    ) -> float:
        """
        Earnings Quality Score (0–10).

        High-quality earnings are fully backed by cash.
        Indicators:
          OCF/NI ratio (multi-year avg)   35% — cash backing of reported profit
          FCF consistency (% positive yr)  25% — free cash flow durability
          Accrual quality                  25% — low accruals signal real profits
          Profitability persistence        15% — consistent positive NI over years

        OCF/NI > 1.0 means more cash than reported profit (excellent).
        OCF/NI 0.8–1.0 means reasonable quality.
        OCF/NI < 0.5 signals earnings inflation risk.
        """
        weights = []
        scores = []

        # ── 1. OCF / Net Income (multi-year average) ──────────────────
        ocf_ni_ratios = []
        for ocf, ni in _series_zip(
            data.operating_cash_flow_series,
            data.net_profit_series,
        ):
            if ni != 0:
                ocf_ni_ratios.append(ocf / ni)

        if ocf_ni_ratios:
            mean_ratio = statistics.mean(ocf_ni_ratios)
            s = self._score_ocf_ni(mean_ratio)
            weights.append(0.35); scores.append(s)
            cv_ocf_ni = _safe_cv(ocf_ni_ratios)
            if cv_ocf_ni is not None:
                weights.append(0.10); scores.append(self._score_cv(cv_ocf_ni))
        
        # ── 2. FCF consistency (fraction of years with positive FCF) ──
        fcf_series = self._fcf_series(data)
        if len(fcf_series) >= 2:
            pos_frac = _positive_fraction(fcf_series)
            # 100% positive = 10, 80% = 8, 60% = 6, 40% = 4, else = 2
            s = min(pos_frac * 10.0 + 1.0, 10.0) if pos_frac > 0.3 else pos_frac * 8.0
            weights.append(0.25); scores.append(round(s, 1))

        # ── 3. Accrual quality (approximation via working capital change)
        accrual_score = self._accrual_quality_score(data)
        if accrual_score is not None:
            weights.append(0.25); scores.append(accrual_score)

        # ── 4. Profitability persistence (fraction of years profitable) ─
        if data.net_profit_series:
            profitable_frac = _positive_fraction(data.net_profit_series)
            s = profitable_frac * 10.0
            weights.append(0.15); scores.append(s)
            if len(data.net_profit_series) >= 2:
                weights.append(0.10)
                scores.append(_trend_slope_score(list(reversed(data.net_profit_series))))

        if not weights:
            return 5.0

        total_w = sum(weights)
        raw = sum(s * w for s, w in zip(scores, weights)) / total_w
        result = round(min(max(raw, 0.0), 10.0), 2)
        logger.debug("QualitativeEngine: EarningsQuality = %.2f (dims=%d)", result, len(weights))
        return result

    def evaluate_capital_allocation(
        self,
        data: CleanedFinancialData,
        ratios: CalculatedRatios,
    ) -> float:
        """
        Capital Allocation Score (0–10).

        Great capital allocation means management consistently deploys capital
        into projects earning above the cost of capital.

        Dimensions:
          ROCE mean (5Y)           30% — average return on all capital deployed
          ROCE trend (direction)   20% — improving allocation over time
          FCF margin mean          25% — surplus cash generation after reinvestment
          FCF trend (direction)    15% — improving cash generation over time
          Reinvestment efficiency  10% — growth achieved per unit of CapEx
        """
        weights = []
        scores = []

        # ── 1. ROCE mean ──────────────────────────────────────────────
        roce_history = self._roce_series(data)
        if roce_history:
            mean_roce = statistics.mean(roce_history)
            s = self._score_roce_mean(mean_roce)
            weights.append(0.30); scores.append(s)

            # ── 2. ROCE trend ─────────────────────────────────────────
            # History is newest-first; reverse for trend calculation
            trend_score = _trend_slope_score(list(reversed(roce_history)))
            weights.append(0.20); scores.append(trend_score)
        elif ratios.roce is not None:
            weights.append(0.30)
            scores.append(self._score_roce_mean(ratios.roce))

        # ── 3. FCF margin mean ────────────────────────────────────────
        fcf_rev_history = self._fcf_revenue_series(data)
        if fcf_rev_history:
            mean_fcf_rev = statistics.mean(fcf_rev_history)
            s = self._score_fcf_margin(mean_fcf_rev)
            weights.append(0.25); scores.append(s)

            # ── 4. FCF trend ──────────────────────────────────────────
            trend_score = _trend_slope_score(list(reversed(fcf_rev_history)))
            weights.append(0.15); scores.append(trend_score)
        elif ratios.fcf_margin is not None:
            weights.append(0.25)
            scores.append(self._score_fcf_margin(ratios.fcf_margin))

        # ── 5. Reinvestment efficiency: revenue CAGR vs CapEx intensity ─
        reinvest_score = self._reinvestment_efficiency_score(data, ratios)
        if reinvest_score is not None:
            weights.append(0.10); scores.append(reinvest_score)

        shareholder_return_score = self._shareholder_return_score(data)
        if shareholder_return_score is not None:
            weights.append(0.08); scores.append(shareholder_return_score)

        cash_deployment_score = self._cash_deployment_score(data)
        if cash_deployment_score is not None:
            weights.append(0.07); scores.append(cash_deployment_score)

        if not weights:
            return 5.0

        total_w = sum(weights)
        raw = sum(s * w for s, w in zip(scores, weights)) / total_w
        result = round(min(max(raw, 0.0), 10.0), 2)
        logger.debug("QualitativeEngine: CapAlloc = %.2f (dims=%d)", result, len(weights))
        return result

    # ──────────────────────────────────────────────────────────────────
    # SERIES EXTRACTORS
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _roe_series(data: CleanedFinancialData) -> List[float]:
        """Historical ROE series (newest first), computed from NP/Equity."""
        result = []
        for np_, eq in _series_zip(data.net_profit_series, data.total_equity_series):
            result.append((np_ / eq) * 100.0)
        return result

    @staticmethod
    def _roce_series(data: CleanedFinancialData) -> List[float]:
        """Historical ROCE series (newest first), from EBIT/CapitalEmployed."""
        result = []
        for ebit, ce in _series_zip(data.ebit_series, data.capital_employed_series):
            result.append((ebit / ce) * 100.0)
        return result

    @staticmethod
    def _margin_series(data: CleanedFinancialData) -> List[float]:
        """Historical operating margin series (newest first)."""
        result = []
        for op, rev in _series_zip(data.operating_income_series, data.revenue_series):
            result.append((op / rev) * 100.0)
        return result

    @staticmethod
    def _fcf_revenue_series(data: CleanedFinancialData) -> List[float]:
        """Historical FCF/Revenue series (newest first)."""
        result = []
        capex = data.capex_series or []
        for i, (ocf, rev) in enumerate(_series_zip(
            data.operating_cash_flow_series, data.revenue_series
        )):
            cap = abs(capex[i]) if i < len(capex) else 0.0
            fcf = ocf - cap
            result.append((fcf / rev) * 100.0)
        return result

    @staticmethod
    def _fcf_series(data: CleanedFinancialData) -> List[float]:
        """Absolute FCF series (newest first)."""
        result = []
        capex = data.capex_series or []
        for i, ocf in enumerate(data.operating_cash_flow_series):
            cap = abs(capex[i]) if i < len(capex) else 0.0
            result.append(ocf - cap)
        return result

    # ──────────────────────────────────────────────────────────────────
    # ACCRUAL QUALITY
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _accrual_quality_score(data: CleanedFinancialData) -> Optional[float]:
        """
        Approximates accrual ratio using Balance Sheet Method:
          Accrual Ratio = (ΔNet Operating Assets) / Average Assets
          ΔNet Operating Assets ≈ Δ(CurrentAssets - CurrentLiabilities)

        A low absolute accrual ratio (< 5%) indicates high earnings quality.
        A high ratio (> 15%) signals potential earnings manipulation.
        """
        ca = data.current_assets_series
        cl = data.current_liabilities_series
        np_ = data.net_profit_series
        rev = data.revenue_series

        if len(ca) < 2 or len(cl) < 2 or not np_ or not rev:
            return None

        min_len = min(len(ca), len(cl), len(np_), len(rev))
        if min_len < 2:
            return None

        # Compute change in net current assets over available years
        accrual_ratios = []
        for i in range(min_len - 1):
            nca_t = ca[i] - cl[i]      # latest year NOA
            nca_t1 = ca[i + 1] - cl[i + 1]  # prior year NOA
            delta = nca_t - nca_t1     # increase in NOA
            avg_revenue = (rev[i] + rev[i + 1]) / 2.0
            if avg_revenue > 0:
                accrual_ratios.append(abs(delta) / avg_revenue * 100.0)

        if not accrual_ratios:
            return None

        mean_accrual = statistics.mean(accrual_ratios)
        # Score: lower accrual = higher quality
        if mean_accrual < 3.0:   return 10.0
        if mean_accrual < 5.0:   return 8.5
        if mean_accrual < 8.0:   return 7.0
        if mean_accrual < 12.0:  return 5.5
        if mean_accrual < 18.0:  return 3.5
        return 1.0

    # ──────────────────────────────────────────────────────────────────
    # REINVESTMENT EFFICIENCY
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _reinvestment_efficiency_score(
        data: CleanedFinancialData,
        ratios: CalculatedRatios,
    ) -> Optional[float]:
        """
        Measures revenue growth achieved relative to CapEx intensity.
        CapEx intensity = Total CapEx (5Y sum) / Average Revenue
        Revenue CAGR / CapEx intensity → high ratio = efficient reinvestment
        """
        capex = data.capex_series
        rev = data.revenue_series
        if not capex or len(rev) < 2:
            return None

        total_capex = sum(abs(c) for c in capex)
        avg_rev = statistics.mean(rev)
        if avg_rev <= 0:
            return None

        capex_intensity = total_capex / (avg_rev * len(capex))  # fraction

        rev_cagr = ratios.revenue_growth  # already computed
        if rev_cagr is None:
            return None

        # Efficiency = growth per unit capex intensity
        if capex_intensity <= 0:
            return 7.0  # Asset-light — good by default

        efficiency = rev_cagr / (capex_intensity * 100.0)

        # Score relative to efficiency ratio
        if efficiency > 2.0:  return 10.0
        if efficiency > 1.0:  return 8.0
        if efficiency > 0.5:  return 6.5
        if efficiency > 0.2:  return 5.0
        return 3.0

    @staticmethod
    def _shareholder_return_score(data: CleanedFinancialData) -> Optional[float]:
        """Scores recurring dividend evidence without assuming buybacks."""
        if not data.dividend_per_share_series:
            return None
        positive_fraction = _positive_fraction(data.dividend_per_share_series)
        trend_score = _trend_slope_score(list(reversed(data.dividend_per_share_series)))
        return round((positive_fraction * 10.0 * 0.6) + (trend_score * 0.4), 2)

    @staticmethod
    def _cash_deployment_score(data: CleanedFinancialData) -> Optional[float]:
        """Scores cash deployment from cash, debt, and near-term reinvestment capacity."""
        if not data.cash_and_equivalents_series:
            return None
        cash = data.cash_and_equivalents_series[0]
        debt = data.total_debt_series[0] if data.total_debt_series else 0.0
        capex = abs(data.capex_series[0]) if data.capex_series else 0.0
        if debt <= 0 and cash > 0:
            return 8.0
        if debt <= 0:
            return None
        cash_to_debt = cash / debt
        if cash_to_debt >= 0.75:
            base = 9.0
        elif cash_to_debt >= 0.40:
            base = 7.5
        elif cash_to_debt >= 0.20:
            base = 6.0
        elif cash_to_debt >= 0.10:
            base = 4.5
        else:
            base = 3.0
        if capex > 0 and cash >= capex:
            base = min(base + 0.75, 10.0)
        return round(base, 2)

    # ──────────────────────────────────────────────────────────────────
    # SCORING TIER HELPERS
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_roe_mean(roe: float) -> float:
        """Maps mean historical ROE to a 0–10 score."""
        if roe >= 25:  return 10.0
        if roe >= 20:  return 8.5
        if roe >= 15:  return 7.0
        if roe >= 10:  return 5.5
        if roe >= 5:   return 3.5
        if roe >= 0:   return 2.0
        return 0.0  # negative mean ROE

    @staticmethod
    def _score_roce_mean(roce: float) -> float:
        """Maps mean historical ROCE to a 0–10 score."""
        if roce >= 20:  return 10.0
        if roce >= 15:  return 8.5
        if roce >= 12:  return 7.0
        if roce >= 8:   return 5.5
        if roce >= 5:   return 3.5
        if roce >= 0:   return 2.0
        return 0.0

    @staticmethod
    def _score_margin_mean(margin: float) -> float:
        """Maps mean historical operating margin to a 0–10 score."""
        if margin >= 25:  return 10.0
        if margin >= 20:  return 8.5
        if margin >= 15:  return 7.0
        if margin >= 10:  return 5.5
        if margin >= 5:   return 3.5
        if margin >= 0:   return 2.0
        return 0.0

    @staticmethod
    def _score_fcf_margin(fcf_pct: float) -> float:
        """Maps mean historical FCF margin to a 0–10 score."""
        if fcf_pct >= 15:  return 10.0
        if fcf_pct >= 10:  return 8.5
        if fcf_pct >= 7:   return 7.0
        if fcf_pct >= 4:   return 5.5
        if fcf_pct >= 1:   return 4.0
        if fcf_pct >= 0:   return 2.5
        return 1.0  # mildly negative FCF (not catastrophic)

    @staticmethod
    def _score_cv(cv: float, is_cyclical: bool = False) -> float:
        """
        Maps coefficient of variation to a stability score 0–10.
        Cyclical industries use more lenient thresholds.
        """
        if is_cyclical:
            # More lenient — high CV is expected in cyclical sectors
            if cv < 0.15:  return 10.0
            if cv < 0.30:  return 8.5
            if cv < 0.50:  return 7.0
            if cv < 0.80:  return 5.5
            if cv < 1.20:  return 4.0
            return 2.5
        else:
            if cv < 0.10:  return 10.0
            if cv < 0.20:  return 8.5
            if cv < 0.35:  return 7.0
            if cv < 0.55:  return 5.0
            if cv < 0.80:  return 3.0
            return 1.0

    @staticmethod
    def _score_ocf_ni(ratio: float) -> float:
        """Maps OCF/Net Income ratio to an earnings quality score 0–10."""
        if ratio >= 1.3:   return 10.0   # Exceptional — more cash than profit
        if ratio >= 1.1:   return 9.0
        if ratio >= 0.9:   return 7.5    # Good — profit mostly backed by cash
        if ratio >= 0.7:   return 6.0
        if ratio >= 0.5:   return 4.0    # Moderate accruals
        if ratio >= 0.3:   return 2.5    # High accruals — caution
        return 1.0                        # Very low cash backing
