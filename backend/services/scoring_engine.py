"""
services/scoring_engine.py — Multi-Dimensional Scoring Engine (Redesigned)
===========================================================================
Architecture:
    This engine replaces the old single-axis "ratio → threshold → score"
    approach with a multi-dimensional weighted model:

    Business Quality Score (drives Grade & BUY/HOLD/SELL):
        1. Financial Quality     40%   (ROE, ROCE, Margins, Cash Flow)
        2. Historical Consistency 15%   (Stability of ROE, Margins, Profits)
        3. Moat / Competitive Edge 8%   (Inferred from ROE & Margin sustainability)
        4. Earnings Quality      12%   (OCF vs Net Income, FCF Consistency)
        5. Capital Allocation    10%   (ROCE sustainability, FCF generation)
        6. Risk Profile          15%   (Debt, Coverage, Liquidity)
        ──────────────────────────────
        TOTAL                   100%

    Valuation Score (separate, independent of Business Quality):
        PE, PB, EV/EBITDA, PEG — scored separately
        NOT blended into Business Quality Score

    Industry Relative Performance:
        Architecture is modular and ready for plug-in.
        Currently 0% weight. Will be enabled when real peer data exists.

    Final Recommendation Logic:
        BUY   = Business Quality ≥ 6.5 AND Valuation ≥ 5.5 (not overpriced)
              OR Business Quality ≥ 8.5 (exceptional business, accept any valuation)
        HOLD  = Business Quality ≥ 5.0 OR Valuation ≥ 6.0
        SELL  = Business Quality < 5.0 AND Valuation < 4.0

Grading (on Business Quality Score):
    S+  ≥ 9.0   Exceptional
    S   ≥ 8.2   Excellent
    A+  ≥ 7.6   Very Strong
    A   ≥ 7.2   Strong
    B+  ≥ 6.4   Good
    B   ≥ 5.5   Average/Good
    C   ≥ 4.0   Below Average
    D   ≥ 2.5   Weak
    F   < 2.5   Poor/High Risk

Confidence Score:
    Computed as % of key data points successfully retrieved.
    Penalizes missing income statement, balance sheet, cash flow.
"""

import logging
from typing import Dict, List, Optional, Tuple

from config.scoring_rules import ALL_SCORING_RULES, RULES_BY_KEY, MetricRule
from models.fundamental import FundamentalScore, Grade, MetricScore, PeerMetrics, Recommendation
from services.consistency_engine import ConsistencyEngine
from services.data_cleaner import CleanedFinancialData
from services.qualitative_engine import QualitativeEngine
from services.ratio_calculator import CalculatedRatios
from services.sector_engine import SectorEngine
from services.valuation_engine import ValuationEngine

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# GRADE TABLE  (on Business Quality Score, highest first)
# ─────────────────────────────────────────────────────────────
_GRADE_MAP: List[Tuple[float, Grade]] = [
    (9.0, Grade.S_PLUS),
    (8.2, Grade.S),
    (7.6, Grade.A_PLUS),
    (7.2, Grade.A),
    (6.4, Grade.B_PLUS),
    (5.5, Grade.B),
    (4.0, Grade.C),
    (2.5, Grade.D),
    (0.0, Grade.F),
]


# ─────────────────────────────────────────────────────────────
# FINANCIAL QUALITY DIMENSION WEIGHTS
# These are the weights used WITHIN the Financial Quality sub-score.
# ─────────────────────────────────────────────────────────────
_FQ_METRIC_WEIGHTS: Dict[str, float] = {
    # Profitability (40% of Financial Quality)
    "roe":              0.16,
    "roce":             0.14,
    "operating_margin": 0.10,
    # Cash Quality (30%)
    "fcf_margin":       0.14,
    "net_margin":       0.10,
    "interest_coverage":0.06,
    # Growth (30%)
    "revenue_growth":   0.10,
    "profit_growth":    0.10,
    "eps_growth":       0.10,
}

# ─────────────────────────────────────────────────────────────
# BUSINESS QUALITY DIMENSION WEIGHTS
# ─────────────────────────────────────────────────────────────
_BQ_DIMENSION_WEIGHTS = {
    "financial_quality":    0.40,
    "consistency":          0.15,
    "risk":                 0.10,
    "earnings_quality":     0.10,
    "capital_allocation":   0.10,
    "moat":                 0.05,
    "industry_relative":    0.00,
}


class ScoringEngine:
    """
    Multi-dimensional scoring engine that evaluates business quality,
    valuation, and risk as independent dimensions.
    """

    def __init__(
        self,
        sector_engine: Optional[SectorEngine] = None,
        peer_engine=None,
    ) -> None:
        self._sector = sector_engine or SectorEngine()
        if peer_engine is None:
            from services.peer_engine import PeerAnalysisEngine
            peer_engine = PeerAnalysisEngine()
        self._peer_engine = peer_engine
        self._valuation_eng = ValuationEngine()
        self._consistency_eng = ConsistencyEngine()
        self._qualitative_eng = QualitativeEngine()
        logger.info("ScoringEngine (multi-dimensional) initialised.")

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def score(
        self,
        ratios: CalculatedRatios,
        cleaned_data: Optional[CleanedFinancialData] = None,
    ) -> Tuple[List[MetricScore], FundamentalScore]:
        ratio_only_mode = cleaned_data is None
        if cleaned_data is None:
            cleaned_data = CleanedFinancialData(ticker="", exchange="")

        rules, sector_profile, excluded = self._sector.get_effective_rules(cleaned_data)
        ratio_map = self._build_ratio_map(ratios, cleaned_data)

        # ── Step 1: Score every individual metric (for display & FQ dimension)
        metric_scores = self._score_all_metrics(rules, ratio_map, excluded, cleaned_data)

        # ── Step 2: Calculate each Business Quality dimension
        fq_score = self._compute_financial_quality(ratio_map, rules, excluded)
        is_cyclical = bool(getattr(sector_profile.metric_profile, "cyclical", False))
        consistency_score = self._consistency_eng.evaluate(cleaned_data, ratios, is_cyclical=is_cyclical)
        moat_score = self._qualitative_eng.evaluate_moat(cleaned_data, ratios, is_cyclical=is_cyclical)
        eq_score = self._qualitative_eng.evaluate_earnings_quality(cleaned_data, ratios)
        ca_score = self._qualitative_eng.evaluate_capital_allocation(cleaned_data, ratios)
        risk_score = self._compute_risk_score(ratio_map)
        if ratio_only_mode:
            consistency_score = fq_score
            if not any((cleaned_data.operating_cash_flow_series, cleaned_data.net_profit_series)):
                eq_score = fq_score
            if not any((cleaned_data.capital_employed_series, cleaned_data.operating_cash_flow_series)):
                ca_score = fq_score
            if not any((cleaned_data.total_equity_series, cleaned_data.operating_income_series)):
                moat_score = fq_score
            if all(ratio_map.get(key) is None for key in ("debt_to_equity", "current_ratio", "interest_coverage")):
                risk_score = fq_score

        # ── Step 3: Weighted Business Quality Score
        bq_score = self._weighted_dimension_score({
            "financial_quality": fq_score,
            "consistency": consistency_score,
            "risk": risk_score,
            "earnings_quality": eq_score,
            "capital_allocation": ca_score,
            "moat": moat_score,
            "industry_relative": None,
        })
        bq_score = min(bq_score, 10.0)

        # ── Step 4: Independent Valuation Score
        valuation_score = self._valuation_eng.evaluate(
            ratios, sector_profile=sector_profile, cleaned_data=cleaned_data
        )

        # ── Step 5: Grade (on Business Quality only)
        grade = self._derive_grade(bq_score)

        # ── Step 6: Recommendation (considers both dimensions)
        recommendation = self._derive_recommendation(bq_score, valuation_score, risk_score)

        # ── Step 7: Confidence Score
        confidence = self._compute_confidence(cleaned_data, ratios)

        # ── Step 8: Coverage stats (for backward-compat display)
        metrics_evaluated = sum(1 for ms in metric_scores if ms.data_available and not ms.informational)
        metrics_total = sum(1 for ms in metric_scores if not ms.informational)
        coverage_pct = round((metrics_evaluated / metrics_total) * 100, 1) if metrics_total else 0.0
        if metrics_evaluated == 0:
            bq_score = 0.0

        fundamental_score = FundamentalScore(
            total_score=bq_score,
            business_quality_score=bq_score,
            valuation_score=round(valuation_score, 2),
            risk_score=round(risk_score, 2),
            financial_quality_score=round(fq_score, 2),
            consistency_score=round(consistency_score, 2),
            moat_score=round(moat_score, 2),
            earnings_quality_score=round(eq_score, 2),
            capital_allocation_score=round(ca_score, 2),
            industry_relative_score=None,
            absolute_total_score=bq_score,
            relative_total_score=None,     # future peer data plug-in
            hybrid_total_score=bq_score,
            grade=grade,
            recommendation=recommendation,
            metrics_evaluated=metrics_evaluated,
            metrics_total=metrics_total,
            coverage_pct=coverage_pct,
            confidence_score=round(confidence, 1),
            data_quality_notes=self._build_data_quality_notes(cleaned_data, ratios),
        )

        metric_scores.sort(key=lambda ms: ms.weight, reverse=True)

        logger.info(
            "ScoringEngine: %s → BQ=%.2f/10 (%s) | Val=%.2f | Risk=%.2f | "
            "FQ=%.2f | Consistency=%.2f | Moat=%.2f | EQ=%.2f | CapAlloc=%.2f | "
            "conf=%.0f%% | rec=%s",
            cleaned_data.ticker,
            bq_score, grade.value,
            valuation_score, risk_score,
            fq_score, consistency_score, moat_score, eq_score, ca_score,
            confidence,
            recommendation.value,
        )

        return metric_scores, fundamental_score

    # ─────────────────────────────────────────────────────────
    # DIMENSION CALCULATIONS
    # ─────────────────────────────────────────────────────────

    def _compute_financial_quality(
        self,
        ratio_map: Dict[str, Optional[float]],
        rules: List[MetricRule],
        excluded: set,
    ) -> float:
        """
        Financial Quality sub-score using sector-adjusted rules,
        renormalized to the FQ metric weight table.
        """
        rules_by_key = {r.key: r for r in rules}
        total_weight = 0.0
        weighted_sum = 0.0

        for key, fq_weight in _FQ_METRIC_WEIGHTS.items():
            if key in excluded:
                continue
            raw = ratio_map.get(key)
            if raw is None:
                continue
            rule = rules_by_key.get(key)
            if rule is None:
                continue
            score = self._apply_tiers(rule, raw)
            weighted_sum += score * fq_weight
            total_weight += fq_weight

        if total_weight == 0:
            return 5.0
        return round(min(weighted_sum / total_weight, 10.0), 2)

    def _compute_risk_score(self, ratio_map: Dict[str, Optional[float]]) -> float:
        """
        Risk sub-score (higher = lower risk = better).
        Dimensions: Debt/Equity, Current Ratio, Interest Coverage.
        """
        scores = []

        de = ratio_map.get("debt_to_equity")
        if de is not None:
            if de == 0:    scores.append(10.0)
            elif de < 0.3: scores.append(10.0)
            elif de < 0.5: scores.append(8.5)
            elif de < 0.8: scores.append(7.5)
            elif de < 1.2: scores.append(6.0)
            elif de < 2.0: scores.append(4.0)
            elif de < 3.0: scores.append(2.0)
            else:          scores.append(0.5)

        cr = ratio_map.get("current_ratio")
        if cr is not None:
            if cr >= 3.0:  scores.append(10.0)
            elif cr >= 2.0: scores.append(8.5)
            elif cr >= 1.5: scores.append(7.0)
            elif cr >= 1.2: scores.append(5.5)
            elif cr >= 1.0: scores.append(4.0)
            elif cr >= 0.8: scores.append(2.0)
            else:           scores.append(0.5)

        ic = ratio_map.get("interest_coverage")
        if ic is not None:
            if ic >= 99:   scores.append(10.0)   # No debt sentinel
            elif ic >= 15: scores.append(10.0)
            elif ic >= 8:  scores.append(8.5)
            elif ic >= 4:  scores.append(7.0)
            elif ic >= 2:  scores.append(5.0)
            elif ic >= 1:  scores.append(3.0)
            else:          scores.append(0.0)

        if not scores:
            return 5.0
        import statistics
        return round(min(statistics.mean(scores), 10.0), 2)

    # ─────────────────────────────────────────────────────────
    # INDIVIDUAL METRIC SCORING (for display list only)
    # ─────────────────────────────────────────────────────────

    def _score_all_metrics(
        self,
        rules: List[MetricRule],
        ratio_map: Dict[str, Optional[float]],
        excluded: set,
        cleaned_data: CleanedFinancialData,
    ) -> List[MetricScore]:
        metric_scores: List[MetricScore] = []
        metadata_only = not any((
            cleaned_data.revenue_series,
            cleaned_data.net_profit_series,
            cleaned_data.total_equity_series,
            cleaned_data.operating_cash_flow_series,
        ))

        for rule in rules:
            if metadata_only and rule.key in excluded:
                continue
            raw_value = ratio_map.get(rule.key)
            is_informational = rule.weight == 0.0

            if raw_value is None:
                abs_score = 0.0
                data_available = False
            else:
                abs_score = self._apply_tiers(rule, raw_value)
                abs_score = self._apply_size_context(rule, raw_value, abs_score, cleaned_data)
                data_available = True

            # Attempt relative scoring via peer engine (no-op until peer data exists)
            rel_score = None
            peer_metrics_obj = None
            if rule.is_relative_eligible and data_available:
                peer_values = self._peer_engine.extract_peer_values(rule.key)
                pm = self._peer_engine.evaluate(raw_value, peer_values, lower_is_better=rule.lower_is_better)
                if pm is not None:
                    rel_score = pm.percentile / 10.0
                    peer_metrics_obj = pm

            hyb_score = abs_score if rel_score is None else (0.65 * abs_score + 0.35 * rel_score)
            final_score = round(hyb_score, 2) if data_available else 0.0

            # Weight shown in MetricScore is the FQ-dimension weight (for display relevance)
            display_weight = _FQ_METRIC_WEIGHTS.get(rule.key, rule.weight)

            metric_scores.append(
                MetricScore(
                    metric_name=rule.display_name,
                    metric_key=rule.key,
                    raw_value=raw_value,
                    raw_value_unit=rule.display_unit,
                    score=final_score,
                    absolute_score=round(abs_score, 2) if data_available else 0.0,
                    relative_score=round(rel_score, 2) if rel_score is not None else None,
                    hybrid_score=round(hyb_score, 2) if data_available else None,
                    peer_metrics=peer_metrics_obj,
                    weight=display_weight,
                    weighted_score=round(final_score * display_weight, 4),
                    explanation="",
                    data_available=data_available,
                    informational=is_informational,
                )
            )

        return metric_scores

    @staticmethod
    def _weighted_dimension_score(dimension_scores: Dict[str, Optional[float]]) -> float:
        active = [
            (score, _BQ_DIMENSION_WEIGHTS.get(key, 0.0))
            for key, score in dimension_scores.items()
            if score is not None and _BQ_DIMENSION_WEIGHTS.get(key, 0.0) > 0
        ]
        total_weight = sum(weight for _, weight in active)
        if total_weight <= 0:
            return 0.0
        return round(sum(score * weight for score, weight in active) / total_weight, 2)

    @staticmethod
    def _apply_size_context(
        rule: MetricRule,
        raw_value: float,
        score: float,
        data: CleanedFinancialData,
    ) -> float:
        """Large mature companies get modest credit for positive growth off a large base."""
        if not rule.is_growth_metric or raw_value < 0 or data.market_cap is None:
            return score
        if data.market_cap >= 100_000 and raw_value >= 3.0:
            return min(score + 1.0, 10.0)
        if data.market_cap >= 20_000 and raw_value >= 5.0:
            return min(score + 0.5, 10.0)
        return score

    # ─────────────────────────────────────────────────────────
    # GRADING & RECOMMENDATION
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _derive_grade(bq_score: float) -> Grade:
        for threshold, grade in _GRADE_MAP:
            if bq_score >= threshold:
                return grade
        return Grade.F

    @staticmethod
    def _derive_recommendation(bq_score: float, val_score: float, risk_score: float) -> Recommendation:
        """
        Recommendation is a function of Business Quality, Valuation, and Risk.
        The model remains constructive only when quality is strong and the risk/valuation
        balance is supportive rather than merely average.
        """
        if risk_score < 3.0:
            return Recommendation.SELL if bq_score < 7.0 else Recommendation.HOLD
        if bq_score >= 8.5 and risk_score >= 5.0:
            return Recommendation.BUY
        if bq_score >= 7.5 and val_score >= 6.0 and risk_score >= 5.5:
            return Recommendation.BUY
        if bq_score >= 5.0 or (val_score >= 6.0 and risk_score >= 4.0):
            return Recommendation.HOLD
        return Recommendation.SELL

    # ─────────────────────────────────────────────────────────
    # CONFIDENCE SCORE
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(data: CleanedFinancialData, ratios: CalculatedRatios) -> float:
        """
        Confidence = % of critical data points available.
        Critical data:
          - Revenue, Net Profit, Operating Income (income statement)
          - Total Equity, Total Debt, Current Assets (balance sheet)
          - Operating Cash Flow, CapEx (cash flow)
          - EPS, EBIT (for coverage)
        Bonus: Market Cap, Current Price available (for valuation)
        """
        checks = [
            bool(data.revenue_series),
            bool(data.net_profit_series),
            bool(data.operating_income_series),
            bool(data.ebit_series),
            bool(data.eps_series),
            bool(data.total_equity_series),
            bool(data.total_debt_series),
            bool(data.current_assets_series),
            bool(data.current_liabilities_series),
            bool(data.operating_cash_flow_series),
            bool(data.capex_series),
            bool(data.capital_employed_series),
            bool(data.book_value_per_share_series),
        ]
        # Valuation data availability (bonus items)
        valuation_checks = [
            data.current_price is not None,
            data.market_cap is not None,
            bool(data.dividend_per_share_series),
            bool(data.cash_and_equivalents_series),
        ]

        core_score = sum(checks) / len(checks)
        val_score = sum(valuation_checks) / len(valuation_checks)

        # Core data weighted 80%, valuation 20%
        confidence = (core_score * 0.80 + val_score * 0.20) * 100
        return round(confidence, 1)

    @staticmethod
    def _build_data_quality_notes(data: CleanedFinancialData, ratios: CalculatedRatios) -> List[str]:
        """Explains which exact line items blocked score-critical calculations."""
        notes = list(dict.fromkeys(data.warnings))
        available = {
            "Revenue": bool(data.revenue_series),
            "Net profit": bool(data.net_profit_series),
            "Shareholder equity": bool(data.total_equity_series),
            "EBIT": bool(data.ebit_series),
            "Capital employed": bool(data.capital_employed_series),
            "Operating income": bool(data.operating_income_series),
            "Total debt": bool(data.total_debt_series),
            "Current assets": bool(data.current_assets_series),
            "Current liabilities": bool(data.current_liabilities_series),
            "Interest expense": bool(data.interest_expense_series),
            "Operating cash flow": bool(data.operating_cash_flow_series),
            "CapEx": bool(data.capex_series),
        }
        requirements = [
            ("Return on Equity", ratios.roe, ["Net profit", "Shareholder equity"]),
            ("ROCE", ratios.roce, ["EBIT", "Capital employed"]),
            ("Operating margin", ratios.operating_margin, ["Operating income", "Revenue"]),
            ("Net margin", ratios.net_margin, ["Net profit", "Revenue"]),
            ("Debt to equity", ratios.debt_to_equity, ["Total debt", "Shareholder equity"]),
            ("Current ratio", ratios.current_ratio, ["Current assets", "Current liabilities"]),
            ("Interest coverage", ratios.interest_coverage, ["EBIT", "Interest expense"]),
            ("FCF margin", ratios.fcf_margin, ["Operating cash flow", "CapEx", "Revenue"]),
        ]
        for metric_name, value, fields in requirements:
            if value is not None:
                continue
            missing = [field for field in fields if not available.get(field, False)]
            if missing:
                notes.append(
                    f"{metric_name} could not be calculated because {', '.join(missing)} "
                    "was missing from the parsed financial statements."
                )
        return list(dict.fromkeys(notes))

    # ─────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _apply_tiers(rule: MetricRule, raw_value: float) -> float:
        """Applies the tier table from a MetricRule to get a 0–10 score."""
        for threshold, tier_score in rule.tiers:
            if threshold is None:
                return tier_score
            if rule.lower_is_better:
                if raw_value < threshold:
                    return tier_score
            else:
                if raw_value >= threshold:
                    return tier_score
        return 0.0

    @staticmethod
    def _build_ratio_map(
        ratios: CalculatedRatios, data: CleanedFinancialData
    ) -> Dict[str, Optional[float]]:
        latest_val = lambda lst: lst[0] if lst else None
        return {
            "revenue_growth":    ratios.revenue_growth,
            "profit_growth":     ratios.profit_growth,
            "eps_growth":        ratios.eps_growth,
            "cash_flow_growth":  ratios.cash_flow_growth,
            "book_value_growth": ratios.book_value_growth,
            "dividend_growth":   ratios.dividend_growth,
            "roe":               ratios.roe,
            "roce":              ratios.roce,
            "operating_margin":  ratios.operating_margin,
            "net_margin":        ratios.net_margin,
            "debt_to_equity":    ratios.debt_to_equity,
            "current_ratio":     ratios.current_ratio,
            "interest_coverage": ratios.interest_coverage,
            "fcf_margin":        ratios.fcf_margin,
            "free_cash_flow":    ratios.free_cash_flow,
            "revenue":           latest_val(data.revenue_series),
            "net_profit":        latest_val(data.net_profit_series),
            "eps":               latest_val(data.eps_series),
            "operating_income":  latest_val(data.operating_income_series),
            "ebit":              latest_val(data.ebit_series),
            "debt":              latest_val(data.total_debt_series),
            "equity":            latest_val(data.total_equity_series),
            "current_assets":    latest_val(data.current_assets_series),
            "current_liabilities": latest_val(data.current_liabilities_series),
            "operating_cash_flow": latest_val(data.operating_cash_flow_series),
            "capex":             latest_val(data.capex_series),
            "interest_expense":  latest_val(data.interest_expense_series),
            "book_value":        latest_val(data.book_value_per_share_series),
            "dividend":          latest_val(data.dividend_per_share_series),
            "capital_employed":  latest_val(data.capital_employed_series),
            "market_cap":        data.market_cap,
            "current_price":     data.current_price,
        }
