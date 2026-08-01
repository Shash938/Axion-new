"""
services/dashboard_engine.py — Research Dashboard Generator
===========================================================
Builds financial health narrative, summary paragraph, growth drivers,
risk factors, and category assessments from calculated metrics.
"""

import logging
from typing import Dict, List, Optional, Set

from models.fundamental import CategoryAssessment, DashboardSummary, MetricScore
from services.data_cleaner import CleanedFinancialData
from services.metric_utils import yoy_growth
from services.ratio_calculator import CalculatedRatios

logger = logging.getLogger(__name__)

_PROFITABILITY_KEYS = {"roe", "roce", "operating_margin", "net_margin"}
_LIQUIDITY_KEYS = {"current_ratio"}
_LEVERAGE_KEYS = {"debt_to_equity", "interest_coverage"}
_CASH_KEYS = {"fcf_margin", "cash_flow_growth"}
_GROWTH_KEYS = {"revenue_growth", "profit_growth", "eps_growth", "book_value_growth", "dividend_growth"}

# Weighted models per category (metric_key → weight within that category)
_CATEGORY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Profitability": {
        "roe":              0.40,
        "roce":             0.35,
        "operating_margin": 0.15,
        "net_margin":       0.10,
    },
    "Growth": {
        "revenue_growth":   0.35,
        "eps_growth":       0.35,
        "profit_growth":    0.20,
        "book_value_growth":0.05,
        "dividend_growth":  0.05,
    },
    "Liquidity": {
        "current_ratio":    1.00,
    },
    "Leverage": {
        "debt_to_equity":   0.55,
        "interest_coverage":0.45,
    },
    "Cash Generation": {
        "fcf_margin":       0.70,
        "cash_flow_growth": 0.30,
    },
}


class DashboardEngine:
    """Generates structured dashboard content from scored metrics and raw financials."""

    def build(
        self,
        metric_scores: List[MetricScore],
        cleaned_data: CleanedFinancialData,
        ratios: CalculatedRatios,
        total_score: float,
        grade: str,
        strengths: List[str],
        weaknesses: List[str],
    ) -> DashboardSummary:
        score_map = {ms.metric_key: ms for ms in metric_scores}
        effective_weaknesses = weaknesses or self._comparative_weaknesses(metric_scores)

        return DashboardSummary(
            financial_health=self._build_financial_health(
                cleaned_data, score_map, ratios, total_score, grade, strengths, effective_weaknesses
            ),
            financial_summary=self._build_financial_summary(cleaned_data, score_map, ratios),
            strengths=strengths,
            weaknesses=effective_weaknesses,
            growth_drivers=self._dedupe_insights(self._identify_growth_drivers(cleaned_data, score_map, ratios)),
            risk_factors=self._dedupe_insights(self._identify_risk_factors(cleaned_data, score_map, ratios)),
            profitability=self._assess_category("Profitability", _PROFITABILITY_KEYS, score_map, cleaned_data, ratios),
            liquidity=self._assess_category("Liquidity", _LIQUIDITY_KEYS, score_map, cleaned_data, ratios),
            leverage=self._assess_category("Leverage", _LEVERAGE_KEYS, score_map, cleaned_data, ratios, invert=True),
            cash_generation=self._assess_category("Cash Generation", _CASH_KEYS, score_map, cleaned_data, ratios),
            growth=self._assess_category("Growth", _GROWTH_KEYS, score_map, cleaned_data, ratios),
            financial_quality=self._assess_financial_quality(total_score, grade, score_map),
        )

    @staticmethod
    def _comparative_weaknesses(metric_scores: List[MetricScore]) -> List[str]:
        candidates = [
            ms for ms in metric_scores
            if ms.data_available and not ms.informational
        ]
        if not candidates:
            return ["Insufficient scored metrics to identify a weakest financial area."]
        weakest = min(candidates, key=lambda metric: metric.score)
        if weakest.score >= 6.5:
            return [f"{weakest.metric_name} is the comparatively softest area despite an acceptable score."]
        return [f"{weakest.metric_name} is the weakest scored financial area."]

    @staticmethod
    def _dedupe_insights(items: List[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for item in items:
            key = item.strip().lower()
            if key and key not in seen:
                result.append(item)
                seen.add(key)
        return result

    # ------------------------------------------------------------------
    # Overall narratives
    # ------------------------------------------------------------------

    def _build_financial_health(
        self,
        data: CleanedFinancialData,
        score_map: Dict[str, MetricScore],
        ratios: CalculatedRatios,
        total_score: float,
        grade: str,
        strengths: List[str],
        weaknesses: List[str],
    ) -> str:
        name = data.company_name
        parts: List[str] = []

        prof_signals = self._collect_profitability_signals(score_map, ratios)
        if prof_signals:
            parts.append(prof_signals)

        growth_signals = self._collect_growth_signals(data, score_map, ratios)
        if growth_signals:
            parts.append(growth_signals)

        leverage_signal = self._collect_leverage_signal(score_map, ratios)
        if leverage_signal:
            parts.append(leverage_signal)

        cash_signal = self._collect_cash_signal(score_map, ratios, data)
        if cash_signal:
            parts.append(cash_signal)

        if parts:
            narrative = " ".join(parts)
            if not narrative.endswith("."):
                narrative += "."
            health = "financially healthy" if total_score >= 7 else "moderately healthy" if total_score >= 5.5 else "financially stressed"
            return f"{name} {narrative} Overall, the company appears {health} with a fundamental score of {total_score:.1f}/10 (Grade {grade})."

        return (
            f"{name} scores {total_score:.1f}/10 (Grade {grade}) based on available financial data. "
            f"Limited metric coverage prevents a more detailed health assessment."
        )

    def _build_financial_summary(
        self,
        data: CleanedFinancialData,
        score_map: Dict[str, MetricScore],
        ratios: CalculatedRatios,
    ) -> str:
        sentences: List[str] = []

        rev_trend = self._series_trend_sentence(
            "Revenue", data.revenue_series, ratios.revenue_growth, "₹ Cr"
        )
        if rev_trend:
            sentences.append(rev_trend)

        profit_trend = self._series_trend_sentence(
            "Net profit", data.net_profit_series, ratios.profit_growth, "₹ Cr"
        )
        if profit_trend:
            sentences.append(profit_trend)

        cf_trend = self._series_trend_sentence(
            "Operating cash flow", data.operating_cash_flow_series, ratios.cash_flow_growth, "₹ Cr"
        )
        if cf_trend:
            sentences.append(cf_trend)

        if ratios.debt_to_equity is not None:
            debt_level = "minimal" if ratios.debt_to_equity < 0.3 else "moderate" if ratios.debt_to_equity < 1.0 else "elevated"
            sentences.append(
                f"Debt-to-equity stands at {ratios.debt_to_equity:.2f}x, indicating {debt_level} leverage."
            )
        elif data.total_debt_series:
            sentences.append(
                f"Total debt is ₹{data.total_debt_series[0]:,.0f} Cr; equity data was insufficient to compute D/E."
            )

        margin_parts = []
        if ratios.operating_margin is not None:
            margin_parts.append(f"operating margin of {ratios.operating_margin:.1f}%")
        if ratios.net_margin is not None:
            margin_parts.append(f"net margin of {ratios.net_margin:.1f}%")
        if margin_parts:
            sentences.append(f"Margins show {' and '.join(margin_parts)}.")

        return_parts = []
        if ratios.roe is not None:
            return_parts.append(f"ROE of {ratios.roe:.1f}%")
        if ratios.roce is not None:
            return_parts.append(f"ROCE of {ratios.roce:.1f}%")
        if return_parts:
            sentences.append(f"Returns are {' and '.join(return_parts)}.")

        if data.market_cap and data.market_cap > 0:
            sentences.append(f"Market capitalisation is ₹{data.market_cap:,.0f} Cr.")

        if not sentences:
            return (
                "Insufficient financial statement data to produce a summary. "
                "Revenue, profit, or balance sheet figures may be missing from the source."
            )

        return " ".join(sentences)

    # ------------------------------------------------------------------
    # Growth drivers & risk factors
    # ------------------------------------------------------------------

    def _identify_growth_drivers(
        self,
        data: CleanedFinancialData,
        score_map: Dict[str, MetricScore],
        ratios: CalculatedRatios,
    ) -> List[str]:
        drivers: List[str] = []

        if self._metric_strong(score_map, "revenue_growth") or (ratios.revenue_growth or 0) >= 8:
            drivers.append("Consistent revenue growth")
        elif data.revenue_series and len(data.revenue_series) >= 2 and yoy_growth(data.revenue_series) and yoy_growth(data.revenue_series) > 5:
            drivers.append("Recent revenue acceleration")

        if self._margin_expanding(data.operating_income_series, data.revenue_series):
            drivers.append("Expanding operating margins")
        elif self._metric_strong(score_map, "operating_margin"):
            drivers.append("Strong operating margins")

        if self._metric_strong(score_map, "roe") or (ratios.roe or 0) >= 18:
            drivers.append("High return on equity (ROE)")
        if self._metric_strong(score_map, "roce") or (ratios.roce or 0) >= 18:
            drivers.append("Efficient capital employment (ROCE)")

        if self._metric_strong(score_map, "cash_flow_growth") or self._metric_strong(score_map, "fcf_margin"):
            drivers.append("Strong cash flow generation")
        elif ratios.fcf_margin is not None and ratios.fcf_margin >= 10:
            drivers.append("Healthy free cash flow margins")

        if ratios.debt_to_equity is not None and ratios.debt_to_equity < 0.5:
            drivers.append("Conservative debt profile")
        elif ratios.interest_coverage is not None and ratios.interest_coverage >= 10:
            drivers.append("Comfortable interest coverage")

        if self._metric_strong(score_map, "eps_growth") or (ratios.eps_growth or 0) >= 8:
            drivers.append("Growing earnings per share (EPS)")
        if self._metric_strong(score_map, "book_value_growth") or (ratios.book_value_growth or 0) >= 8:
            drivers.append("Increasing book value per share")
        if self._metric_strong(score_map, "dividend_growth") or (ratios.dividend_growth or 0) >= 5:
            drivers.append("Growing dividend payouts")
        if self._metric_strong(score_map, "profit_growth"):
            drivers.append("Sustained profit growth")

        if not drivers:
            return ["No significant growth drivers identified from available metrics."]

        return drivers[:8]

    def _identify_risk_factors(
        self,
        data: CleanedFinancialData,
        score_map: Dict[str, MetricScore],
        ratios: CalculatedRatios,
    ) -> List[str]:
        risks: List[str] = []

        if self._metric_weak(score_map, "revenue_growth") or (ratios.revenue_growth is not None and ratios.revenue_growth < 0):
            risks.append("Declining or weak revenue growth")
        elif data.revenue_series and len(data.revenue_series) >= 2:
            yoy = yoy_growth(data.revenue_series)
            if yoy is not None and yoy < -5:
                risks.append("Recent revenue decline")

        if self._metric_weak(score_map, "cash_flow_growth"):
            risks.append("Falling or stagnant cash flow growth")
        elif data.operating_cash_flow_series and len(data.operating_cash_flow_series) >= 2:
            yoy = yoy_growth(data.operating_cash_flow_series)
            if yoy is not None and yoy < -10:
                risks.append("Operating cash flow declined year-over-year")

        if ratios.debt_to_equity is not None and ratios.debt_to_equity > 1.5:
            risks.append("High debt-to-equity ratio")
        elif self._metric_weak(score_map, "debt_to_equity"):
            risks.append("Elevated leverage")

        if ratios.interest_coverage is not None and ratios.interest_coverage < 3 and ratios.interest_coverage < 99:
            risks.append("Low interest coverage")
        elif self._metric_weak(score_map, "interest_coverage"):
            risks.append("Weak ability to cover interest expenses")

        if ratios.free_cash_flow is not None and ratios.free_cash_flow < 0:
            risks.append("Negative free cash flow")
        elif self._metric_weak(score_map, "fcf_margin"):
            risks.append("Weak cash conversion (FCF margin)")

        if self._margin_contracting(data.operating_income_series, data.revenue_series):
            risks.append("Shrinking operating margins")
        elif self._metric_weak(score_map, "operating_margin"):
            risks.append("Below-average operating margins")

        if self._metric_weak(score_map, "eps_growth") or (ratios.eps_growth is not None and ratios.eps_growth < 0):
            risks.append("Weak or negative EPS growth")
        if self._metric_weak(score_map, "profit_growth"):
            risks.append("Profit growth lagging expectations")
        if self._metric_weak(score_map, "current_ratio"):
            risks.append("Liquidity concerns (low current ratio)")
        if self._metric_weak(score_map, "roe"):
            risks.append("Low return on equity")
        if self._metric_weak(score_map, "net_margin"):
            risks.append("Thin net profit margins")

        if not risks:
            return ["No major financial risk factors identified from available metrics."]

        return risks[:8]

    # ------------------------------------------------------------------
    # Category assessments
    # ------------------------------------------------------------------

    def _assess_category(
        self,
        name: str,
        keys: Set[str],
        score_map: Dict[str, MetricScore],
        data: CleanedFinancialData,
        ratios: CalculatedRatios,
        invert: bool = False,
    ) -> CategoryAssessment:
        available = [
            score_map[k] for k in keys
            if k in score_map and score_map[k].data_available and not score_map[k].informational
        ]
        if not available:
            missing = ", ".join(sorted(keys))
            return CategoryAssessment(
                grade="N/A",
                explanation=f"Insufficient data to assess {name.lower()}. Required metrics ({missing}) were unavailable.",
            )

        # Use weighted model if available, fallback to simple average
        weight_model = _CATEGORY_WEIGHTS.get(name, {})
        if weight_model:
            total_w = sum(weight_model.get(ms.metric_key, 0) for ms in available)
            if total_w > 0:
                weighted_sum = sum(
                    ms.score * weight_model.get(ms.metric_key, 0)
                    for ms in available
                )
                avg = round(weighted_sum / total_w, 1)
            else:
                avg = round(sum(ms.score for ms in available) / len(available), 1)
        else:
            avg = round(sum(ms.score for ms in available) / len(available), 1)

        grade = self._score_to_grade(avg)
        explanation = self._category_explanation(name, keys, score_map, data, ratios, avg, invert)
        return CategoryAssessment(score=avg, grade=grade, explanation=explanation)

    def _assess_financial_quality(
        self,
        total_score: float,
        grade: str,
        score_map: Dict[str, MetricScore],
    ) -> CategoryAssessment:
        weighted_available = [
            ms for ms in score_map.values()
            if ms.data_available and not ms.informational
        ]
        if not weighted_available:
            return CategoryAssessment(
                grade="N/A",
                explanation="Overall financial quality cannot be assessed due to insufficient scored metrics.",
            )

        if total_score >= 9.0:
            explanation = "Exceptional balance across profitability, growth, leverage, and cash generation."
        elif total_score >= 8.0:
            explanation = "High-quality financial profile with strong metrics across most categories."
        elif total_score >= 7.0:
            explanation = "Good overall financial quality with manageable weaknesses."
        elif total_score >= 5.5:
            explanation = "Average financial quality — strengths and weaknesses are balanced."
        elif total_score >= 4.0:
            explanation = "Below-average financial quality with multiple areas needing improvement."
        else:
            explanation = "Poor financial quality with significant fundamental weaknesses."

        weak_count = sum(1 for ms in weighted_available if ms.score <= 3)
        if weak_count >= 3:
            explanation += f" {weak_count} metrics scored poorly (≤3/10)."

        return CategoryAssessment(score=round(total_score, 1), grade=grade, explanation=explanation)

    def _category_explanation(
        self,
        name: str,
        keys: Set[str],
        score_map: Dict[str, MetricScore],
        data: CleanedFinancialData,
        ratios: CalculatedRatios,
        avg: float,
        invert: bool,
    ) -> str:
        strong = [score_map[k].metric_name for k in keys if k in score_map and score_map[k].data_available and score_map[k].score >= 7]
        weak = [score_map[k].metric_name for k in keys if k in score_map and score_map[k].data_available and score_map[k].score <= 3]

        quality = "strong" if avg >= 7.5 else "adequate" if avg >= 5.5 else "weak"
        if invert and name == "Leverage":
            quality = "conservative" if avg >= 7.5 else "moderate" if avg >= 5.5 else "stressed"

        parts = [f"{name} is {quality} (avg {avg:.1f}/10)."]
        if strong:
            parts.append(f"Standouts: {', '.join(strong[:3])}.")
        if weak:
            parts.append(f"Concerns: {', '.join(weak[:3])}.")

        detail = self._category_detail(name, ratios, data)
        if detail:
            parts.append(detail)

        return " ".join(parts)

    def _category_detail(
        self, name: str, ratios: CalculatedRatios, data: CleanedFinancialData
    ) -> str:
        if name == "Profitability":
            bits = []
            if ratios.roe is not None:
                bits.append(f"ROE {ratios.roe:.1f}%")
            if ratios.operating_margin is not None:
                bits.append(f"operating margin {ratios.operating_margin:.1f}%")
            return f"Latest: {', '.join(bits)}." if bits else ""
        if name == "Liquidity" and ratios.current_ratio is not None:
            return f"Current ratio: {ratios.current_ratio:.2f}x."
        if name == "Leverage" and ratios.debt_to_equity is not None:
            return f"Debt-to-equity: {ratios.debt_to_equity:.2f}x."
        if name == "Cash Generation":
            bits = []
            if ratios.fcf_margin is not None:
                bits.append(f"FCF margin {ratios.fcf_margin:.1f}%")
            if ratios.cash_flow_growth is not None:
                bits.append(f"cash flow CAGR {ratios.cash_flow_growth:.1f}%")
            return f"Latest: {', '.join(bits)}." if bits else ""
        if name == "Growth" and ratios.revenue_growth is not None:
            return f"Revenue CAGR: {ratios.revenue_growth:.1f}%."
        return ""

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _metric_strong(score_map: Dict[str, MetricScore], key: str) -> bool:
        ms = score_map.get(key)
        return ms is not None and ms.data_available and ms.score >= 7

    @staticmethod
    def _metric_weak(score_map: Dict[str, MetricScore], key: str) -> bool:
        ms = score_map.get(key)
        return ms is not None and ms.data_available and ms.score <= 3

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 9.0:
            return "A+"
        if score >= 8.0:
            return "A"
        if score >= 7.0:
            return "B"
        if score >= 5.5:
            return "C"
        if score >= 4.0:
            return "D"
        return "F"

    @staticmethod
    def _margin_expanding(numerator: List[float], denominator: List[float]) -> bool:
        length = min(len(numerator), len(denominator))
        if length < 2:
            return False
        current = safe_margin(numerator[0], denominator[0])
        prior = safe_margin(numerator[1], denominator[1])
        return current is not None and prior is not None and current > prior + 0.5

    @staticmethod
    def _margin_contracting(numerator: List[float], denominator: List[float]) -> bool:
        length = min(len(numerator), len(denominator))
        if length < 2:
            return False
        current = safe_margin(numerator[0], denominator[0])
        prior = safe_margin(numerator[1], denominator[1])
        return current is not None and prior is not None and current < prior - 0.5

    def _series_trend_sentence(
        self,
        label: str,
        series: List[float],
        cagr_val: Optional[float],
        unit: str,
    ) -> str:
        if not series:
            return ""
        latest = series[0]
        if unit == "₹ Cr":
            latest_str = f"₹{latest:,.0f} Cr"
        else:
            latest_str = f"{latest:.2f}"

        if cagr_val is not None:
            direction = "grown" if cagr_val > 2 else "declined" if cagr_val < -2 else "remained stable"
            return f"{label} has {direction} at a {abs(cagr_val):.1f}% CAGR; latest annual figure is {latest_str}."

        if len(series) >= 2:
            yoy = yoy_growth(series)
            if yoy is not None:
                direction = "increased" if yoy > 0 else "decreased"
                return f"{label} {direction} {abs(yoy):.1f}% year-over-year to {latest_str}."

        return f"Latest {label.lower()} is {latest_str}."

    def _collect_profitability_signals(
        self, score_map: Dict[str, MetricScore], ratios: CalculatedRatios
    ) -> str:
        bits = []
        if ratios.roe is not None and ratios.roe >= 15:
            bits.append(f"high ROE of {ratios.roe:.1f}%")
        if ratios.operating_margin is not None and ratios.operating_margin >= 15:
            bits.append(f"operating margins of {ratios.operating_margin:.1f}%")
        if not bits and self._metric_strong(score_map, "roe"):
            bits.append("strong profitability metrics")
        if bits:
            return f"demonstrates {' and '.join(bits)}"
        return ""

    def _collect_growth_signals(
        self,
        data: CleanedFinancialData,
        score_map: Dict[str, MetricScore],
        ratios: CalculatedRatios,
    ) -> str:
        bits = []
        if ratios.revenue_growth is not None and ratios.revenue_growth >= 5:
            bits.append("revenue has grown steadily")
        elif data.revenue_series and len(data.revenue_series) >= 2 and yoy_growth(data.revenue_series) and yoy_growth(data.revenue_series) > 0:
            bits.append("revenue is trending upward")
        if ratios.profit_growth is not None and ratios.profit_growth >= 5:
            bits.append("earnings have expanded")
        if bits:
            return f"{' and '.join(bits).capitalize()}"
        return ""

    def _collect_leverage_signal(
        self, score_map: Dict[str, MetricScore], ratios: CalculatedRatios
    ) -> str:
        if ratios.debt_to_equity is not None:
            if ratios.debt_to_equity < 0.3:
                return "while maintaining minimal debt"
            if ratios.debt_to_equity > 1.5:
                return "though leverage is elevated and warrants monitoring"
        if self._metric_strong(score_map, "debt_to_equity"):
            return "with a conservative balance sheet"
        return ""

    def _collect_cash_signal(
        self,
        score_map: Dict[str, MetricScore],
        ratios: CalculatedRatios,
        data: CleanedFinancialData,
    ) -> str:
        if ratios.fcf_margin is not None and ratios.fcf_margin >= 10:
            return "Cash generation remains strong"
        if self._metric_strong(score_map, "fcf_margin") or self._metric_strong(score_map, "cash_flow_growth"):
            return "Cash generation remains healthy"
        if ratios.free_cash_flow is not None and ratios.free_cash_flow > 0:
            return f"Generated ₹{ratios.free_cash_flow:,.0f} Cr in free cash flow recently"
        return ""


def safe_margin(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return (numerator / denominator) * 100
