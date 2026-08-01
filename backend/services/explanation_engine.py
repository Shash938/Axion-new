"""
services/explanation_engine.py — Human-Readable Explanation Generator
=======================================================================
Why this file exists:
    Core to the product's value proposition: retail investors should understand
    WHY a stock receives its score, not just WHAT the score is.
    This service generates beginner-friendly, contextual explanations for each
    metric score and an overall investment summary paragraph.

How it connects:
    - Receives the `List[MetricScore]` (without explanations) from
      services/scoring_engine.py.
    - Also receives `CalculatedRatios` for raw values and `CleanedFinancialData`
      for company context.
    - Returns the same list with `explanation` fields populated, plus:
        - A comprehensive `overall_explanation` paragraph.
        - `strengths` — metrics scoring ≥ 7.
        - `weaknesses` — metrics scoring ≤ 3.

Writing style guidelines (enforced via templates):
    - Use simple, concrete language. No jargon without explanation.
    - State the metric value in plain terms (e.g. "₹22 profit for every ₹100 invested").
    - State whether the result is good, adequate, or poor — never ambiguous.
    - For missing data, explain why the metric couldn't be scored.

Possible improvements:
    - Use an LLM (Gemini, GPT-4) to generate more natural, personalised explanations.
    - Add trend commentary ("improving over 3 years", "declining from last year").
    - Add benchmark comparison ("above the sector median of X%").
"""

import logging
from typing import Dict, List, Optional

from config.scoring_rules import RULES_BY_KEY
from models.fundamental import FundamentalScore, MetricScore
from services.data_cleaner import CleanedFinancialData
from services.historical_engine import MetricDetail
from services.metric_utils import cagr_5y, format_inr_crores, yoy_growth
from services.ratio_calculator import CalculatedRatios

logger = logging.getLogger(__name__)

# Score thresholds for classifying metrics
_STRENGTH_THRESHOLD = 7.0
_WEAKNESS_THRESHOLD = 3.0


class ExplanationEngine:
    """
    Generates plain-English explanations for each metric score and an
    overall investment summary.

    Usage:
        engine = ExplanationEngine()
        metric_scores, overall, strengths, weaknesses = engine.explain(
            metric_scores, ratios, cleaned_data, fundamental_score
        )
    """

    def __init__(self) -> None:
        logger.info("ExplanationEngine initialised.")

    def explain(
        self,
        metric_scores: List[MetricScore],
        ratios: CalculatedRatios,
        cleaned_data: CleanedFinancialData,
        fundamental_score: FundamentalScore,
        metric_details: Optional[Dict[str, MetricDetail]] = None,
    ) -> tuple:
        """
        Populates explanations for each MetricScore and generates an overall summary.
        Uses metric_details for historical context when available.
        """
        self._cleaned_data = cleaned_data
        self._metric_details = metric_details or {}
        annotated: List[MetricScore] = []
        strengths: List[str] = []
        weaknesses: List[str] = []

        for ms in metric_scores:
            explanation = self._explain_metric(ms, ratios)
            annotated.append(ms.model_copy(update={"explanation": explanation}))

            if ms.data_available and not ms.informational:
                if ms.score >= _STRENGTH_THRESHOLD:
                    strengths.append(self._strength_label(ms))
                elif ms.score <= _WEAKNESS_THRESHOLD:
                    weaknesses.append(self._weakness_label(ms))

        if not strengths:
            # Find the highest-scoring metric and label it as a strength
            best_scored = [
                ms for ms in metric_scores
                if ms.data_available and not ms.informational
            ]
            if best_scored:
                best = max(best_scored, key=lambda m: m.score)
                strengths.append(self._strength_label(best))

        weak_candidates = [
            ms for ms in metric_scores
            if ms.data_available and not ms.informational
        ]
        if weak_candidates:
            weakest = min(weak_candidates, key=lambda m: m.score)
            relative_weakness = self._relative_weakness_label(weakest)
            if relative_weakness not in weaknesses:
                weaknesses = [relative_weakness] + [
                    item for item in weaknesses if item != self._weakness_label(weakest)
                ]

        overall = self._build_overall_explanation(
            cleaned_data, fundamental_score, strengths, weaknesses
        )

        logger.info(
            "ExplanationEngine: %s — %d strengths, %d weaknesses.",
            cleaned_data.ticker,
            len(strengths),
            len(weaknesses),
        )
        return annotated, overall, strengths, weaknesses

    # ------------------------------------------------------------------
    # Per-Metric Explanations
    # ------------------------------------------------------------------

    def _explain_metric(self, ms: MetricScore, ratios: CalculatedRatios) -> str:
        """Dispatches to the correct explanation method by metric key."""
        detail = self._metric_details.get(ms.metric_key) if hasattr(self, "_metric_details") else None

        if not ms.data_available or ms.raw_value is None:
            rule = RULES_BY_KEY.get(ms.metric_key)
            name = rule.display_name if rule else ms.metric_name
            if ms.metric_key == "dividend":
                return (
                    f"**{name}**: No Dividend History. "
                    "This company may not pay dividends, or Yahoo Finance has no payment record."
                )
            reason = ""
            if detail and not detail.data_available:
                reason = f" Reason: {self._unavailability_reason(ms.metric_key, detail)}"
            suffix = reason or " Required financial statement rows were not found."
            return (
                f"**{name}**: Data was not available to calculate this metric.{suffix} "
                "It has been scored 0 and excluded from the weighted total."
            )

        explainer = self._METRIC_EXPLAINERS.get(ms.metric_key)
        if explainer:
            base_explanation = explainer(self, ms, ratios)
        else:
            base_explanation = self._generic_explanation(ms)
            
        if ms.peer_metrics:
            pm = ms.peer_metrics
            base_explanation += f" Relative to its peers, it ranks in the {pm.percentile}th percentile (Industry Rank: {pm.industry_rank}/{pm.total_peers})."
        elif getattr(RULES_BY_KEY.get(ms.metric_key), 'is_relative_eligible', False):
            base_explanation += " (Industry peer comparison currently unavailable)."
            
        return base_explanation

    def _history_context(self, series: list, unit: str = "") -> str:
        """Builds a historical narrative from a newest-first series."""
        if not series or len(series) < 2:
            return ""
        oldest = series[-1]
        latest = series[0]
        cagr5 = cagr_5y(series)
        yoy = yoy_growth(series)
        parts = []
        if len(series) >= 2:
            if unit == "₹ Cr":
                parts.append(
                    f"Moved from {format_inr_crores(oldest)} to {format_inr_crores(latest)} over "
                    f"{len(series) - 1} years"
                )
            else:
                parts.append(f"Moved from {oldest:.2f} to {latest:.2f} over {len(series) - 1} years")
        if cagr5 is not None:
            parts.append(f"5-year CAGR of {cagr5:.1f}%")
        if yoy is not None:
            parts.append(f"latest YoY {yoy:+.1f}%")
        return ". ".join(parts) + "." if parts else ""

    def _detail_history_text(self, metric_key: str) -> str:
        """Builds history narrative from precomputed MetricDetail."""
        detail = getattr(self, "_metric_details", {}).get(metric_key)
        if not detail or not detail.history:
            return ""
        points = [f"{pt.value}{detail.unit} ({pt.year})" for pt in detail.history[-5:]]
        trend = detail.trend or "Unavailable"
        parts = [f"Historical values: {', '.join(points)}"]
        if detail.yoy is not None:
            parts.append(f"latest YoY {detail.yoy:+.1f}%")
        if detail.cagr3 is not None:
            parts.append(f"3-year CAGR {detail.cagr3:.1f}%")
        if detail.cagr5 is not None:
            parts.append(f"5-year CAGR {detail.cagr5:.1f}%")
        parts.append(f"trend direction is {trend}")
        if trend in {"Growing", "Accelerating", "Expanding", "Improving"}:
            parts.append("the multi-year pattern supports consistency")
        elif trend in {"Declining", "Decelerating", "Contracting", "Weakening"}:
            parts.append("the multi-year pattern weakens consistency")
        return ". ".join(parts) + "."

    def _benchmark_context(self, metric_key: str) -> str:
        detail = getattr(self, "_metric_details", {}).get(metric_key)
        if not detail or not detail.benchmark_label:
            return ""
        summary = detail.benchmark_summary or ""
        return f" Relative to the industry peer group, this is {detail.benchmark_label.lower()} ({summary})." if summary else f" Relative to the industry peer group, this is {detail.benchmark_label.lower()}."

    @staticmethod
    def _unavailability_reason(metric_key: str, detail: MetricDetail) -> str:
        if metric_key in {"roce"} and not detail.data_available:
            return "ROCE requires EBIT and capital employed — one or both line items were missing."
        if metric_key in {"operating_margin", "net_margin"}:
            return "Margin requires both numerator and revenue — insufficient matched rows."
        if metric_key.endswith("_growth"):
            return "Growth requires at least two years of consistent history."
        return "Required financial statement rows were not found in Yahoo Finance data."

    def _explain_roe(self, ms: MetricScore, ratios: CalculatedRatios) -> str:
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        hist = self._detail_history_text("roe")
        bench = self._benchmark_context("roe")
        implication = (
            "This indicates excellent capital efficiency and durable competitive advantage."
            if v >= 20
            else "This indicates healthy return levels suitable for long-term compounding."
            if v >= 15
            else "This indicates average returns — monitor whether management can improve capital allocation."
            if v >= 10
            else "A ROE below 10% suggests inefficient use of shareholder capital."
        )
        return (
            f"**Return on Equity (ROE): {v:.1f}%** — {quality}. "
            f"For every ₹100 of equity, the company generates ₹{v:.1f} in profit. "
            f"{hist} "
            f"The professional benchmark for excellence is 20%.{bench} "
            f"{implication}"
        )

    def _explain_roce(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        hist = self._detail_history_text("roce")
        bench = self._benchmark_context("roce")
        return (
            f"**Return on Capital Employed (ROCE): {v:.1f}%** — {quality}. "
            f"The company earns ₹{v:.1f} for every ₹100 of capital employed (equity + debt). "
            f"{hist} "
            f"ROCE above 20% is considered excellent for industrial companies.{bench} "
            f"Sustained high ROCE indicates pricing power and efficient deployment of total capital."
        )

    def _explain_revenue_growth(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        return (
            f"**Revenue Growth (CAGR): {v:.1f}%** — {quality}. "
            f"This is the compound annual growth rate of revenue over the available history. "
            f"A CAGR above 10% usually signals healthy expansion and improving competitive position."
        )

    def _explain_profit_growth(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        return (
            f"**Profit Growth (CAGR): {v:.1f}%** — {quality}. "
            f"This is the compound annual growth rate of net profit over the available history. "
            f"Growth above 10% is solid and usually suggests operating leverage and improving profitability."
        )

    def _explain_eps_growth(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        return (
            f"**EPS Growth (CAGR): {v:.1f}%** — {quality}. "
            f"This is the compound annual growth rate of earnings per share over the available history. "
            f"High EPS growth indicates compounding of shareholder value without excessive dilution."
        )

    def _explain_debt_to_equity(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        hist = ""
        if ms.history:
            hist = f"Historical D/E levels: " + ", ".join([f"{pt.value}x ({pt.year})" for pt in ms.history]) + "."
        return (
            f"**Debt to Equity: {v:.2f}x** — {quality}. "
            f"This ratio measures financial leverage by dividing total debt by shareholder equity. "
            f"{hist} "
            f"A D/E under 0.3x is considered Excellent (nearly debt-free), while above 1.5x-2.0x poses "
            f"significant leverage risk. {'Very conservative debt profile.' if v < 0.3 else 'Healthy leverage.' if v < 1.0 else 'High leverage, indicating elevated insolvency risk.'}"
        )

    def _explain_current_ratio(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        hist = ""
        if ms.history:
            hist = f"Historical short-term liquidity: " + ", ".join([f"{pt.value}x ({pt.year})" for pt in ms.history]) + "."
        return (
            f"**Current Ratio: {v:.2f}x** — {quality}. "
            f"This measures a company's ability to cover short-term liabilities with current assets. "
            f"The company holds ₹{v:.2f} of liquid assets for every ₹1 of current debt. "
            f"{hist} "
            f"Ratios between 1.5x to 2.5x are considered healthy. A ratio below 1.0x indicates liquidity risk."
        )

    def _explain_operating_margin(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        hist = ""
        if ms.history:
            hist = f"Operating margin trend: " + ", ".join([f"{pt.value}% ({pt.year})" for pt in ms.history]) + "."
        return (
            f"**Operating Margin: {v:.1f}%** — {quality}. "
            f"Operating profit kept from each ₹100 of sales, measuring core pricing power. "
            f"{hist} "
            f"Margins above 15% are generally good, and >25% are exceptional. "
            f"Expanding operating margins validate cost optimization and strong pricing power."
        )

    def _explain_net_margin(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        hist = ""
        if ms.history:
            hist = f"Net margin trend: " + ", ".join([f"{pt.value}% ({pt.year})" for pt in ms.history]) + "."
        return (
            f"**Net Profit Margin: {v:.1f}%** — {quality}. "
            f"Bottom-line margin representing profit left after all expenses, taxes, and interest. "
            f"{hist} "
            f"Margins above 15% indicate strong cash conversion and bottom-line health, "
            f"leaving more cash for dividend payouts and reinvestment."
        )

    def _explain_fcf_margin(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        fcf_abs = ratios.free_cash_flow if ratios else None
        fcf_text = ""
        if fcf_abs is not None:
            fcf_text = f" This corresponds to an absolute Free Cash Flow of {format_inr_crores(fcf_abs)} in the latest year."
        return (
            f"**FCF Margin: {v:.1f}%** — {quality}. "
            f"FCF margin measures cash conversion quality relative to revenue — a size-neutral metric. "
            f"{fcf_text} "
            f"An FCF margin >15% is considered Excellent. Positive FCF margin proves that "
            f"profits are fully backed by hard cash rather than non-cash accounting entries."
        )

    def _explain_cash_flow_growth(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        return (
            f"**Cash Flow Growth (CAGR): {v:.1f}%** — {quality}. "
            f"This is the compound annual growth rate of operating cash flow over the available history. "
            f"Healthy cash flow growth indicates the business can fund expansion organically."
        )

    def _explain_interest_coverage(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        if v >= 99:
            return (
                "**Interest Coverage: N/A (Debt-Free)** — Excellent. "
                "The company has no significant debt obligations, eliminating interest payment risk entirely."
            )
        hist = ""
        if ms.history:
            hist = f"Historical coverage: " + ", ".join([f"{pt.value}x ({pt.year})" for pt in ms.history]) + "."
        return (
            f"**Interest Coverage: {v:.1f}x** — {quality}. "
            f"EBIT divided by interest expense, showing how comfortably earnings cover finance costs. "
            f"Earnings cover interest {v:.1f} times. {hist} "
            f"Coverage > 10x is Excellent, while <1.5x indicates high risk of default during operational slowdowns."
        )

    def _explain_book_value_growth(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        return (
            f"**Book Value Growth (CAGR): {v:.1f}%** — {quality}. "
            f"This is the compound annual growth rate of book value per share over the available history. "
            f"A strong result points to retained earnings building intrinsic value per share."
        )

    def _explain_dividend_growth(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        return (
            f"**Dividend Growth (CAGR): {v:.1f}%** — {quality}. "
            f"This is the compound annual growth rate of dividend per share over the available history. "
            f"Consistent dividend growth signals management confidence in future cash flow stability."
        )

    # ------------------------------------------------------------------
    # New Informational Explainers
    # ------------------------------------------------------------------

    def _explain_revenue(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.revenue_series, "₹ Cr")
        return (
            f"**Revenue: {format_inr_crores(v)}** — Informational. "
            f"Revenue represents the total top-line sales generated by the company. "
            f"Historically, the company's revenue has {hist} "
            f"Revenue expansion indicates business scale growth, while contracting revenue "
            f"could signal market share loss or industry decline. This metric provides the foundation for profit margins."
        )

    def _explain_net_profit(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.net_profit_series, "₹ Cr")
        return (
            f"**Net Profit: {format_inr_crores(v)}** — Informational. "
            f"Net Profit is the bottom-line earnings after all expenses, interest, and taxes. "
            f"Historically, net profit has {hist} "
            f"Increasing net profit demonstrates strong cost control and overall profitability, "
            f"directly contributing to shareholder value accumulation and capacity for reinvestment."
        )

    def _explain_eps(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.eps_series, "")
        return (
            f"**Earnings Per Share (EPS): ₹{v:.2f}** — Informational. "
            f"EPS measures the portion of profit allocated to each outstanding share of common stock. "
            f"Historically, EPS has {hist} "
            f"An expanding EPS shows that profitability growth is outpacing share dilution, "
            f"which typically supports share price appreciation over the long term."
        )

    def _explain_operating_income(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.operating_income_series, "₹ Cr")
        return (
            f"**Operating Income: {format_inr_crores(v)}** — Informational. "
            f"Operating Income represents the earnings generated from core operations before interest and taxes. "
            f"Historically, operating income has {hist} "
            f"Growth in operating income validates the core business's viability and operational scaling "
            f"independent of capital structure decisions."
        )

    def _explain_ebit(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.ebit_series, "₹ Cr")
        return (
            f"**EBIT: {format_inr_crores(v)}** — Informational. "
            f"EBIT (Earnings Before Interest and Taxes) is a key measure of operating efficiency. "
            f"Historically, EBIT has {hist} "
            f"EBIT is used to calculate returns on capital employed (ROCE) and interest coverage, "
            f"providing a pure view of profit generation capacity."
        )

    def _explain_debt(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.total_debt_series, "₹ Cr")
        return (
            f"**Total Debt: {format_inr_crores(v)}** — Informational. "
            f"Total Debt is the sum of short-term and long-term borrowings. "
            f"Historically, borrowings have {hist} "
            f"Decreasing debt reduces financial risk and interest burden, while increasing debt "
            f"warrants monitoring to ensure it is being used productively to generate higher returns."
        )

    def _explain_equity(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.total_equity_series, "₹ Cr")
        return (
            f"**Shareholder Equity: {format_inr_crores(v)}** — Informational. "
            f"Shareholder Equity represents the net worth of the company belonging to stockholders. "
            f"Historically, equity has {hist} "
            f"Consistent equity growth indicates that the company is retaining earnings and building a "
            f"solid financial buffer against future downturns."
        )

    def _explain_current_assets(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.current_assets_series, "₹ Cr")
        return (
            f"**Current Assets: {format_inr_crores(v)}** — Informational. "
            f"Current Assets are short-term resources that can be converted into cash within one year. "
            f"Historically, current assets have {hist} "
            f"Adequate current assets are vital for maintaining smooth day-to-day operations and meeting working capital requirements."
        )

    def _explain_current_liabilities(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.current_liabilities_series, "₹ Cr")
        return (
            f"**Current Liabilities: {format_inr_crores(v)}** — Informational. "
            f"Current Liabilities are obligations that must be settled within one year. "
            f"Historically, current liabilities have {hist} "
            f"Control over current liabilities prevents working capital deficits and liquidity stress."
        )

    def _explain_operating_cash_flow(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.operating_cash_flow_series, "₹ Cr")
        return (
            f"**Operating Cash Flow: {format_inr_crores(v)}** — Informational. "
            f"Operating Cash Flow measures actual cash generated from core activities, excluding investing/financing. "
            f"Historically, operating cash flow has {hist} "
            f"Positive operating cash flow shows the company can sustain its operations, pay dividends, "
            f"and make capital investments without relying solely on debt."
        )

    def _explain_capex(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.capex_series, "₹ Cr")
        return (
            f"**Capital Expenditure (CapEx): {format_inr_crores(v)}** — Informational. "
            f"CapEx represents the cash spent on acquiring or upgrading long-term physical assets. "
            f"Historically, CapEx has {hist} "
            f"Strategic CapEx is critical for future revenue scaling, but high capital intensity "
            f"can restrict free cash flow if returns on investment are deferred or low."
        )

    def _explain_free_cash_flow(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        return (
            f"**Free Cash Flow: {format_inr_crores(v)}** — Informational. "
            f"Free Cash Flow (FCF) is the actual surplus cash generated after paying for capital expenditures. "
            f"FCF represents the pool available to pay dividends, repurchase shares, or pay down debt. "
            f"Consistent positive FCF is the ultimate hallmark of a high-quality self-funding business model."
        )

    def _explain_interest_expense(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.interest_expense_series, "₹ Cr")
        return (
            f"**Interest Expense: {format_inr_crores(v)}** — Informational. "
            f"Interest Expense is the finance cost incurred on borrowings. "
            f"Historically, interest expense has {hist} "
            f"Lower interest expenses improve profit margins and reduce insolvency risk during difficult operating cycles."
        )

    def _explain_book_value(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.book_value_per_share_series, "")
        return (
            f"**Book Value Per Share: ₹{v:.2f}** — Informational. "
            f"Book Value represents the net asset value of the firm on a per-share basis. "
            f"Historically, book value has {hist} "
            f"Consistent growth in book value per share signals long-term equity compounding and corporate wealth accumulation."
        )

    def _explain_dividend(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.dividend_per_share_series, "")
        return (
            f"**Dividend Per Share (DPS): ₹{v:.2f}** — Informational. "
            f"DPS represents the absolute dividend paid out to shareholders per share. "
            f"Historically, DPS has {hist} "
            f"Consistent dividend payouts indicate mature cash flows and corporate commitment to direct shareholder returns."
        )

    def _explain_capital_employed(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        hist = self._history_context(self._cleaned_data.capital_employed_series, "₹ Cr")
        return (
            f"**Capital Employed: {format_inr_crores(v)}** — Informational. "
            f"Capital Employed represents the total capital utilized to generate profits (Total Assets - Current Liabilities). "
            f"Historically, capital employed has {hist} "
            f"Tracking capital employed is essential to evaluate efficiency metrics like ROCE (Return on Capital Employed)."
        )

    def _explain_market_cap(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        return (
            f"**Market Cap: {format_inr_crores(v)}** — Informational. "
            f"Market Capitalization is the total market value of the company's outstanding shares. "
            f"It classifies the stock into cap size: Large Cap (>₹20,000 Cr), Mid Cap (₹5,000-20,000 Cr), "
            f"or Small Cap (<₹5,000 Cr), which dictates risk and volatility profiles."
        )

    def _explain_current_price(self, ms: MetricScore, ratios: CalculatedRatios = None) -> str:
        v = ms.raw_value
        return (
            f"**Current Price: ₹{v:,.2f}** — Informational. "
            f"This is the latest traded price of the stock on the exchange. "
            f"It represents the current market valuation per share and is the base for calculating ratios like P/E and dividend yields."
        )

    def _generic_explanation(self, ms: MetricScore) -> str:
        """Fallback explanation for metrics without a dedicated explainer."""
        v = ms.raw_value
        quality = self._quality_label(ms.score)
        
        rel_text = ""
        if ms.relative_score is not None:
            rel_text = f" (Absolute: {ms.absolute_score:.1f}, Relative: {ms.relative_score:.1f})"
            
        return (
            f"**{ms.metric_name}: {v:.2f}{ms.raw_value_unit}** — {quality}. "
            f"This metric scored {ms.score:.1f}/10{rel_text}."
        )

    # ------------------------------------------------------------------
    # Overall Summary
    # ------------------------------------------------------------------

    def _build_overall_explanation(
        self,
        cleaned_data: CleanedFinancialData,
        fundamental_score: FundamentalScore,
        strengths: List[str],
        weaknesses: List[str],
    ) -> str:
        """Assembles a multi-dimensional research summary."""
        name = cleaned_data.company_name
        bq   = fundamental_score.business_quality_score
        val  = fundamental_score.valuation_score
        risk = fundamental_score.risk_score
        grade = fundamental_score.grade.value
        rec   = fundamental_score.recommendation.value
        conf  = fundamental_score.confidence_score
        coverage = fundamental_score.coverage_pct

        # Valuation commentary
        if val >= 7.5:
            val_label = "attractively valued"
        elif val >= 5.5:
            val_label = "fairly valued"
        elif val >= 3.5:
            val_label = "slightly stretched on valuation"
        else:
            val_label = "richly priced relative to fundamentals"

        # Risk commentary
        if risk >= 8.0:
            risk_label = "very low financial risk"
        elif risk >= 6.5:
            risk_label = "manageable financial risk"
        elif risk >= 4.5:
            risk_label = "moderate financial risk"
        else:
            risk_label = "elevated financial risk requiring monitoring"

        intro = (
            f"{name} has received a **Business Quality Score of {bq:.1f}/10 (Grade {grade})**, "
            f"resulting in a **{rec}** recommendation. "
            f"The company carries **{risk_label}** (Risk Score: {risk:.1f}/10) and is currently "
            f"{val_label} (Valuation Score: {val:.1f}/10). "
        )

        dimension_text = (
            f"Financial quality is {fundamental_score.financial_quality_score:.1f}/10, "
            f"Historical consistency is {fundamental_score.consistency_score:.1f}/10, "
            f"earnings quality is {fundamental_score.earnings_quality_score:.1f}/10, "
            f"capital allocation is {fundamental_score.capital_allocation_score:.1f}/10, "
            f"moat is {fundamental_score.moat_score:.1f}/10, risk is {risk:.1f}/10, "
            f"valuation is {val:.1f}/10, and confidence is {conf:.0f}%. "
        )

        recommendation_rationale = self._build_recommendation_rationale(rec, bq, val, risk)

        strength_text = ""
        if strengths:
            listed = ", ".join(strengths[:3])
            if len(strengths) > 3:
                listed += f", and {len(strengths) - 3} more"
            strength_text = f"Primary strengths are {listed}. "

        weakness_text = ""
        if weaknesses:
            listed = ", ".join(weaknesses[:3])
            if len(weaknesses) > 3:
                listed += f", and {len(weaknesses) - 3} more"
            weakness_text = f"Primary risks are {listed}. "

        coverage_text = ""
        if coverage < 70:
            coverage_text = (
                f"Note: Only {coverage:.0f}% of metrics could be calculated due to data availability. "
                "Treat this analysis with caution. "
            )

        data_quality_text = ""
        if fundamental_score.data_quality_notes:
            quality_notes = "; ".join(fundamental_score.data_quality_notes[:2])
            data_quality_text = f"Data quality note: {quality_notes} "

        conf_text = f"Data Confidence: {conf:.0f}%. "

        disclaimer = (
            "This analysis is based on publicly available financial statements and is intended "
            "for informational purposes only. Always conduct your own research before making "
            "investment decisions."
        )

        return (
            intro + dimension_text + recommendation_rationale + strength_text + weakness_text +
            coverage_text + data_quality_text + conf_text + "\n\n*" + disclaimer + "*"
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _build_recommendation_rationale(rec: str, bq: float, val: float, risk: float) -> str:
        if rec == "BUY":
            return (
                "Recommendation rationale: business quality is strong enough to justify ownership, "
                "and the valuation and risk balance supports a constructive stance. "
            )
        if rec == "SELL":
            return (
                "Recommendation rationale: business quality is insufficiently strong and the valuation-risk "
                "combination does not support ownership at this time. "
            )
        return (
            "Recommendation rationale: the business is acceptable on quality, but the valuation and risk "
            "balance should be monitored carefully before increasing exposure. "
        )

    @staticmethod
    def _quality_label(score: float) -> str:
        """Maps a 0–10 score to a natural-language quality descriptor."""
        if score >= 8.5:
            return "🟢 Excellent"
        elif score >= 7.0:
            return "🟢 Good"
        elif score >= 5.5:
            return "🟡 Adequate"
        elif score >= 3.5:
            return "🟠 Below Average"
        elif score > 0:
            return "🔴 Poor"
        else:
            return "⚫ No Data"

    @staticmethod
    def _strength_label(metric: MetricScore) -> str:
        label = metric.metric_name
        if metric.metric_key == "roe":
            return f"ROE is strong relative to its peer group"
        if metric.metric_key == "revenue_growth":
            return f"Revenue growth is above the industry average"
        if metric.metric_key == "fcf_margin":
            return f"Free cash flow conversion is healthy"
        return f"{label} is performing well"

    @staticmethod
    def _weakness_label(metric: MetricScore) -> str:
        label = metric.metric_name
        if metric.metric_key == "revenue_growth":
            return "Revenue growth is the weakest measured growth signal"
        if metric.metric_key == "roe":
            return "ROE is the weakest measured return signal"
        if metric.metric_key == "debt_to_equity":
            return "Leverage is the weakest measured risk signal"
        if metric.metric_key == "fcf_margin":
            return "Cash conversion is the weakest measured cash-flow signal"
        return f"{label} is the weakest measured area"

    @staticmethod
    def _relative_weakness_label(metric: MetricScore) -> str:
        if metric.score >= 6.5:
            return (
                f"{metric.metric_name} is the comparatively weakest area, "
                f"although it still scores {metric.score:.1f}/10"
            )
        return f"{metric.metric_name} is the comparatively weakest area at {metric.score:.1f}/10"

    # Dispatch table: metric_key → bound method
    # Defined here so `_METRIC_EXPLAINERS` is a class attribute
    _METRIC_EXPLAINERS = {
        "roe": _explain_roe,
        "roce": _explain_roce,
        "revenue_growth": _explain_revenue_growth,
        "profit_growth": _explain_profit_growth,
        "eps_growth": _explain_eps_growth,
        "debt_to_equity": _explain_debt_to_equity,
        "current_ratio": _explain_current_ratio,
        "operating_margin": _explain_operating_margin,
        "net_margin": _explain_net_margin,
        "fcf_margin": _explain_fcf_margin,
        "cash_flow_growth": _explain_cash_flow_growth,
        "interest_coverage": _explain_interest_coverage,
        "book_value_growth": _explain_book_value_growth,
        "dividend_growth": _explain_dividend_growth,
        
        # Absolute Metrics
        "revenue": _explain_revenue,
        "net_profit": _explain_net_profit,
        "eps": _explain_eps,
        "operating_income": _explain_operating_income,
        "ebit": _explain_ebit,
        "debt": _explain_debt,
        "equity": _explain_equity,
        "current_assets": _explain_current_assets,
        "current_liabilities": _explain_current_liabilities,
        "operating_cash_flow": _explain_operating_cash_flow,
        "capex": _explain_capex,
        "free_cash_flow": _explain_free_cash_flow,
        "interest_expense": _explain_interest_expense,
        "book_value": _explain_book_value,
        "dividend": _explain_dividend,
        "capital_employed": _explain_capital_employed,
        "market_cap": _explain_market_cap,
        "current_price": _explain_current_price,
    }
