"""
services/ratio_calculator.py — Financial Ratio Calculation Service
===================================================================
Why this file exists:
    Pure calculation logic, completely isolated from data fetching, scoring,
    or API concerns. Every method is a deterministic function — same input
    always produces the same output, with no side effects. This makes the
    service trivially unit-testable without any mocking.

How it connects:
    - Receives `CleanedFinancialData` from services/data_cleaner.py.
    - Outputs a `CalculatedRatios` dataclass consumed by
      services/scoring_engine.py and services/explanation_engine.py.

Calculation approach:
    Growth metrics use CAGR (Compound Annual Growth Rate) over the available
    data period rather than simple year-on-year change. This is more robust
    for comparing companies with different data availability (3 vs 5 years).

    Formula: CAGR = (latest / oldest) ^ (1 / n_years) − 1

    Point-in-time metrics (ROE, D/E, margins) use the most recent year's values.

Possible improvements:
    - Add TTM (Trailing Twelve Month) calculations using quarterly data for
      more current margin and earnings metrics.
    - Add peer comparison context (vs sector median).
    - Add trend direction (improving / stable / deteriorating).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.data_cleaner import CleanedFinancialData
from services.metric_utils import cagr, safe_divide

logger = logging.getLogger(__name__)


# ==============================================================================
# Output Container
# ==============================================================================


@dataclass
class CalculatedRatios:
    """
    The result of running all 14 financial ratio calculations.

    All percentage values are in actual percentage terms (e.g. 22.5, not 0.225).
    None means insufficient data was available to compute the metric.
    """
    # --- Growth Metrics (%) ---
    revenue_growth: Optional[float] = None        # CAGR %
    profit_growth: Optional[float] = None         # CAGR %
    eps_growth: Optional[float] = None            # CAGR %
    cash_flow_growth: Optional[float] = None      # CAGR %
    book_value_growth: Optional[float] = None     # CAGR %
    dividend_growth: Optional[float] = None       # CAGR %

    # --- Profitability Metrics (%) ---
    roe: Optional[float] = None                   # %
    roce: Optional[float] = None                  # %
    operating_margin: Optional[float] = None      # %
    net_margin: Optional[float] = None            # %

    # --- Leverage Metrics (ratios, 'x') ---
    debt_to_equity: Optional[float] = None        # x
    current_ratio: Optional[float] = None         # x
    interest_coverage: Optional[float] = None     # x

    # --- Absolute Cash Metric (INR Crores) ---
    free_cash_flow: Optional[float] = None        # INR Crores
    fcf_margin: Optional[float] = None            # FCF / Revenue × 100
    
    # --- Valuation Metrics ---
    enterprise_value: Optional[float] = None      # INR Crores
    pe_ratio: Optional[float] = None              # x (from yfinance info when available)
    pb_ratio: Optional[float] = None              # x (from yfinance info when available)
    ev_ebitda: Optional[float] = None             # x
    peg_ratio: Optional[float] = None             # x
    dividend_yield: Optional[float] = None        # %
    price_to_sales: Optional[float] = None        # x
    # --- Traceability / Metadata ---
    metric_metadata: Optional[Dict[str, Dict[str, Any]]] = None
    def __post_init__(self) -> None:
        for field_name in (
            "enterprise_value",
            "pe_ratio",
            "pb_ratio",
            "ev_ebitda",
            "peg_ratio",
            "dividend_yield",
            "price_to_sales",
        ):
            if getattr(self, field_name, None) is None:
                delattr(self, field_name)


# ==============================================================================
# Service
# ==============================================================================


class RatioCalculatorService:
    """
    Calculates all 14 fundamental financial ratios from cleaned financial data.

    Pure service — no constructor dependencies, no state, no I/O.

    Usage:
        calc = RatioCalculatorService()
        ratios = calc.calculate(cleaned_data)
    """

    def __init__(self) -> None:
        logger.info("RatioCalculatorService initialised.")

    def calculate(self, data: CleanedFinancialData) -> CalculatedRatios:
        """
        Runs all ratio calculations on the cleaned data.

        Each calculation is isolated in its own method. A failed calculation
        logs a warning and sets the corresponding ratio to None — it does NOT
        abort the entire analysis.
        """
        ratios = CalculatedRatios()
        ratios.metric_metadata = None

        ratios.revenue_growth = self._safe_calculate("revenue_growth", self._revenue_growth, data)
        ratios.profit_growth = self._safe_calculate("profit_growth", self._profit_growth, data)
        ratios.eps_growth = self._safe_calculate("eps_growth", self._eps_growth, data)
        ratios.cash_flow_growth = self._safe_calculate("cash_flow_growth", self._cash_flow_growth, data)
        ratios.book_value_growth = self._safe_calculate("book_value_growth", self._book_value_growth, data)
        ratios.dividend_growth = self._safe_calculate("dividend_growth", self._dividend_growth, data)
        ratios.roe = self._safe_calculate("roe", self._roe, data)
        ratios.roce = self._safe_calculate("roce", self._roce, data)
        ratios.operating_margin = self._safe_calculate("operating_margin", self._operating_margin, data)
        ratios.net_margin = self._safe_calculate("net_margin", self._net_margin, data)
        ratios.debt_to_equity = self._safe_calculate("debt_to_equity", self._debt_to_equity, data)
        ratios.current_ratio = self._safe_calculate("current_ratio", self._current_ratio, data)
        ratios.interest_coverage = self._safe_calculate("interest_coverage", self._interest_coverage, data)
        ratios.free_cash_flow = self._safe_calculate("free_cash_flow", self._free_cash_flow, data)
        ratios.fcf_margin = self._safe_calculate("fcf_margin", self._fcf_margin, data)
        self._assign_if_available(ratios, "enterprise_value", self._safe_calculate("enterprise_value", self._enterprise_value, data))
        self._assign_if_available(ratios, "pe_ratio", self._safe_calculate("pe_ratio", self._pe_ratio, data))
        self._assign_if_available(ratios, "pb_ratio", self._safe_calculate("pb_ratio", self._pb_ratio, data))
        self._assign_if_available(ratios, "ev_ebitda", self._safe_calculate("ev_ebitda", self._ev_ebitda, data))
        self._assign_if_available(ratios, "peg_ratio", self._safe_calculate("peg_ratio", self._peg_ratio, data))
        self._assign_if_available(ratios, "dividend_yield", self._safe_calculate("dividend_yield", self._dividend_yield, data))
        self._assign_if_available(ratios, "price_to_sales", self._safe_calculate("price_to_sales", self._price_to_sales, data))

        self._attach_metric_metadata(ratios, data)

        available = sum(1 for v in vars(ratios).values() if v is not None)
        logger.info(
            "RatioCalculatorService: Calculated %d/%d ratios for %s.",
            available,
            len(vars(ratios)),
            data.ticker,
        )
        return ratios

    # ------------------------------------------------------------------
    # Growth Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _is_banking_sector(data: CleanedFinancialData) -> bool:
        sector = (data.sector or "").lower()
        industry = (data.industry or "").lower()
        return any(token in sector or token in industry for token in ("bank", "financial", "insurance", "nbfc"))

    @staticmethod
    def _revenue_growth(data: CleanedFinancialData) -> Optional[float]:
        return cagr(data.revenue_series)

    @staticmethod
    def _profit_growth(data: CleanedFinancialData) -> Optional[float]:
        return cagr(data.net_profit_series)

    @staticmethod
    def _eps_growth(data: CleanedFinancialData) -> Optional[float]:
        return cagr(data.eps_series)

    @staticmethod
    def _cash_flow_growth(data: CleanedFinancialData) -> Optional[float]:
        return cagr(data.operating_cash_flow_series)

    @staticmethod
    def _book_value_growth(data: CleanedFinancialData) -> Optional[float]:
        return cagr(data.book_value_per_share_series)

    @staticmethod
    def _dividend_growth(data: CleanedFinancialData) -> Optional[float]:
        s = data.dividend_per_share_series
        if len(s) < 2:
            return None
        return cagr(s)

    # ------------------------------------------------------------------
    # Profitability Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _roe(data: CleanedFinancialData) -> Optional[float]:
        """Return on Equity = Net Income / Average Shareholders' Equity × 100."""
        np_ = data.net_profit_series
        eq = data.total_equity_series
        if not np_ or not eq or len(eq) < 2:
            return None

        latest_equity = eq[0]
        prior_equity = eq[1]
        if latest_equity == 0 or prior_equity == 0:
            return None

        avg_equity = (latest_equity + prior_equity) / 2.0
        if avg_equity == 0:
            return None
        return round((np_[0] / avg_equity) * 100, 2)

    @staticmethod
    def _roce(data: CleanedFinancialData) -> Optional[float]:
        """ROCE = EBIT / Average Capital Employed × 100. Not used for banks/financials."""
        if RatioCalculatorService._is_banking_sector(data):
            return None
        ebit = data.ebit_series
        ce = data.capital_employed_series
        if not ebit or not ce or len(ce) < 2:
            return None

        latest_ce = ce[0]
        prior_ce = ce[1]
        if latest_ce == 0 or prior_ce == 0:
            return None

        avg_ce = (latest_ce + prior_ce) / 2.0
        if avg_ce == 0:
            return None
        return round((ebit[0] / avg_ce) * 100, 2)

    @staticmethod
    def _operating_margin(data: CleanedFinancialData) -> Optional[float]:
        """Operating Margin = Operating Income / Revenue × 100. Not used for banks/financials."""
        if RatioCalculatorService._is_banking_sector(data):
            return None
        op = data.operating_income_series
        rev = data.revenue_series
        if not op or not rev or rev[0] == 0:
            return None
        return round((op[0] / rev[0]) * 100, 2)

    @staticmethod
    def _net_margin(data: CleanedFinancialData) -> Optional[float]:
        """Net Margin = Net Profit (latest) / Revenue (latest) × 100"""
        np_ = data.net_profit_series
        rev = data.revenue_series
        if not np_ or not rev or rev[0] == 0:
            return None
        return round((np_[0] / rev[0]) * 100, 2)

    # ------------------------------------------------------------------
    # Leverage & Liquidity Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _debt_to_equity(data: CleanedFinancialData) -> Optional[float]:
        """D/E = Total Debt (latest) / Total Equity (latest)"""
        debt = data.total_debt_series
        eq = data.total_equity_series
        if not debt or not eq or eq[0] == 0:
            return None
        # D/E can be negative if equity is negative; cap at large value
        raw_de = debt[0] / eq[0]
        return round(max(raw_de, 0.0), 2)

    @staticmethod
    def _current_ratio(data: CleanedFinancialData) -> Optional[float]:
        """Current Ratio = Current Assets / Current Liabilities. Not used for banks/financials."""
        if RatioCalculatorService._is_banking_sector(data):
            return None
        ca = data.current_assets_series
        cl = data.current_liabilities_series
        if not ca or not cl or cl[0] == 0:
            return None
        return round(ca[0] / cl[0], 2)

    @staticmethod
    def _interest_coverage(data: CleanedFinancialData) -> Optional[float]:
        """Interest Coverage = EBIT / Interest Expense. Returns None when expense is zero or unavailable."""
        if RatioCalculatorService._is_banking_sector(data):
            return None
        ebit = data.ebit_series
        interest = data.interest_expense_series
        if not ebit or not interest:
            return None

        interest_val = interest[0]
        if interest_val is None:
            return None

        interest_abs = abs(float(interest_val))
        if interest_abs == 0:
            return None
        if ebit[0] is None:
            return None
        return round(float(ebit[0]) / interest_abs, 2)

    # ------------------------------------------------------------------
    # Cash Flow Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _free_cash_flow(data: CleanedFinancialData) -> Optional[float]:
        """FCF = Operating Cash Flow − CapEx, using capex's sign convention as reported."""
        ocf = data.operating_cash_flow_series
        capex = data.capex_series
        if not ocf:
            return None
        ocf_val = ocf[0]
        capex_val = capex[0] if capex else 0.0
        if capex_val < 0:
            return round(ocf_val + capex_val, 2)
        if capex_val > 0:
            return round(ocf_val - capex_val, 2)
        return round(ocf_val, 2)

    @staticmethod
    def _fcf_margin(data: CleanedFinancialData) -> Optional[float]:
        """FCF Margin = Free Cash Flow / Revenue × 100 (size-neutral measure)."""
        ocf = data.operating_cash_flow_series
        rev = data.revenue_series
        capex = data.capex_series
        if not ocf or not rev or rev[0] == 0:
            return None
        capex_val = capex[0] if capex else 0.0
        if capex_val < 0:
            fcf = ocf[0] + capex_val
        elif capex_val > 0:
            fcf = ocf[0] - capex_val
        else:
            fcf = ocf[0]
        return safe_divide(fcf, rev[0], as_pct=True)

    # ------------------------------------------------------------------
    # Valuation Metrics
    # ------------------------------------------------------------------
    
    @staticmethod
    def _enterprise_value(data: CleanedFinancialData) -> Optional[float]:
        """EV = Market Cap + Total Debt - Cash and Equivalents"""
        if data.market_cap is None:
            return None
        debt = data.total_debt_series[0] if data.total_debt_series else 0.0
        cash = data.cash_and_equivalents_series[0] if data.cash_and_equivalents_series else 0.0
        return round(data.market_cap + debt - cash, 2)
        
    @staticmethod
    def _pe_ratio(data: CleanedFinancialData) -> Optional[float]:
        """P/E ratio — prefer yfinance trailingPE (already currency-normalized)."""
        # Priority 1: yfinance info (accurate, handles currency)
        info = getattr(data, '_raw_info', {})
        for key in ("trailingPE", "forwardPE"):
            v = info.get(key)
            if v is not None:
                try:
                    f = float(v)
                    if 0 < f < 2000:  # Sanity bound
                        return round(f, 2)
                except (TypeError, ValueError):
                    pass
        # Priority 2: compute from cleaned EPS series
        eps = data.eps_series
        if not data.current_price or not eps or eps[0] <= 0:
            return None
        return round(data.current_price / eps[0], 2)
        
    @staticmethod
    def _pb_ratio(data: CleanedFinancialData) -> Optional[float]:
        """P/B ratio — prefer yfinance priceToBook (already currency-normalized)."""
        # Priority 1: yfinance info
        info = getattr(data, '_raw_info', {})
        v = info.get("priceToBook")
        if v is not None:
            try:
                f = float(v)
                if 0 < f < 500:
                    return round(f, 2)
            except (TypeError, ValueError):
                pass
        # Priority 2: compute from book value per share
        bvps = data.book_value_per_share_series
        if not data.current_price or not bvps or bvps[0] <= 0:
            return None
        return round(data.current_price / bvps[0], 2)
        
    @staticmethod
    def _ev_ebitda(data: CleanedFinancialData) -> Optional[float]:
        """EV / EBITDA. Returns None unless actual EBITDA-like data is present."""
        ev = RatioCalculatorService._enterprise_value(data)
        if ev is None:
            return None

        info = getattr(data, "_raw_info", {}) or {}
        ebitda = None
        for key in ("ebitda", "EBITDA"):
            value = info.get(key)
            if value is None:
                continue
            try:
                ebitda = float(value)
            except (TypeError, ValueError):
                continue
            break

        if ebitda is None or ebitda <= 0:
            return None
        return round(ev / ebitda, 2)
        
    @staticmethod
    def _peg_ratio(data: CleanedFinancialData) -> Optional[float]:
        """PEG = P/E / EPS Growth CAGR. Prefer yfinance pegRatio."""
        info = getattr(data, '_raw_info', {})
        v = info.get("pegRatio")
        if v is not None:
            try:
                f = float(v)
                if 0 < f < 100:
                    return round(f, 2)
            except (TypeError, ValueError):
                pass
        # Fallback: compute
        pe = RatioCalculatorService._pe_ratio(data)
        eps_g = RatioCalculatorService._eps_growth(data)
        if pe is None or not eps_g or eps_g <= 0:
            return None
        return round(pe / eps_g, 2)
        
    @staticmethod
    def _dividend_yield(data: CleanedFinancialData) -> Optional[float]:
        """Dividend Yield = Latest Dividend Per Share / Current Price * 100"""
        div = data.dividend_per_share_series
        if not div or not data.current_price or data.current_price <= 0:
            return None
        return round((div[0] / data.current_price) * 100, 2)
        
    @staticmethod
    def _price_to_sales(data: CleanedFinancialData) -> Optional[float]:
        """P/S = Market Cap / Revenue"""
        rev = data.revenue_series
        if not data.market_cap or not rev or rev[0] <= 0:
            return None
        return round(data.market_cap / rev[0], 2)

    # ------------------------------------------------------------------
    # Shared Calculation Utilities (delegates to metric_utils)
    # ------------------------------------------------------------------

    @staticmethod
    def _cagr(series: List[float], label: str = "") -> Optional[float]:
        """Backward-compatible wrapper around metric_utils.cagr."""
        return cagr(series)

    @staticmethod
    def _safe_calculate(name: str, func, data: CleanedFinancialData) -> Optional[float]:
        """
        Wraps a calculation function to catch unexpected exceptions.
        Returns None and logs a warning so one broken metric never kills
        the full analysis pipeline.
        """
        try:
            return func(data)
        except Exception as exc:
            logger.warning("Ratio '%s' calculation failed unexpectedly: %s", name, exc)
            return None

    @staticmethod
    def _assign_if_available(ratios: CalculatedRatios, name: str, value: Optional[float]) -> None:
        if value is not None:
            setattr(ratios, name, value)

    @staticmethod
    def _attach_metric_metadata(ratios: CalculatedRatios, data: CleanedFinancialData) -> None:
        if not any(
            value is not None
            for value in (
                ratios.roe,
                ratios.roce,
                ratios.interest_coverage,
                ratios.free_cash_flow,
                ratios.ev_ebitda,
            )
        ):
            ratios.metric_metadata = None
            return

        period = data.fiscal_years[0] if data.fiscal_years else "latest"
        ratios.metric_metadata = {
            "roe": {
                "formula": "Net Income / Average Shareholders' Equity",
                "numerator": "net_profit_series[0]",
                "denominator": "(total_equity_series[0] + total_equity_series[1]) / 2",
                "fiscal_period": period,
                "source_metrics": ["net_profit_series", "total_equity_series"],
            },
            "roce": {
                "formula": "EBIT / Average Capital Employed",
                "numerator": "ebit_series[0]",
                "denominator": "(capital_employed_series[0] + capital_employed_series[1]) / 2",
                "fiscal_period": period,
                "source_metrics": ["ebit_series", "capital_employed_series"],
            },
            "interest_coverage": {
                "formula": "EBIT / Interest Expense",
                "numerator": "ebit_series[0]",
                "denominator": "abs(interest_expense_series[0])",
                "fiscal_period": period,
                "source_metrics": ["ebit_series", "interest_expense_series"],
            },
            "free_cash_flow": {
                "formula": "Operating Cash Flow - CapEx",
                "numerator": "operating_cash_flow_series[0]",
                "denominator": "capex_series[0] (sign preserved)",
                "fiscal_period": period,
                "source_metrics": ["operating_cash_flow_series", "capex_series"],
            },
            "ev_ebitda": {
                "formula": "Enterprise Value / EBITDA",
                "numerator": "enterprise_value",
                "denominator": "raw_info['ebitda']",
                "fiscal_period": period,
                "source_metrics": ["market_cap", "total_debt_series", "cash_and_equivalents_series", "_raw_info"],
            },
        }
