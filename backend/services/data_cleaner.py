"""
services/data_cleaner.py — Data Validation and Cleaning Service
================================================================
Why this file exists:
    Raw yfinance data is notoriously messy: rows may have NaN, columns may be
    missing or reordered, currency may differ, and some statements may be empty.
    This service is the "data quality gate" — nothing passes to the ratio
    calculator unless it's been validated and normalised here.

How it connects:
    - Receives `RawFinancialData` from services/data_fetcher.py.
    - Outputs a `CleanedFinancialData` dict consumed by
      services/ratio_calculator.py.

Cleaning rules:
    1. Extract scalar values from DataFrames by row label.
    2. Convert INR paise to standard units where necessary.
    3. Replace Inf/-Inf with NaN.
    4. Tag each metric as available/unavailable (for coverage reporting).
    5. Surface non-fatal warnings without aborting the analysis.

Possible improvements:
    - Add sector-specific normalisation (banks use different line items).
    - Validate expected currency is INR; raise if USD-denominated data is found.
    - Generate a DataQualityReport with per-field confidence scores.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from services.data_fetcher import RawFinancialData
from services.financial_parser import (
    BALANCE_ALIASES,
    CASHFLOW_ALIASES,
    INCOME_ALIASES,
    FinancialParser,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Output Container
# ==============================================================================


@dataclass
class CleanedFinancialData:
    """
    Validated, normalised financial data ready for ratio calculation.

    All monetary values are in INR Crores (1 Crore = 10,000,000 INR).
    All growth rates and margins are expressed as percentages (not decimals).
    Optional fields are None when data was not available.
    """

    # --- Company metadata ---
    ticker: str
    exchange: str
    company_name: str = "Unknown"
    sector: str = "Unknown"
    industry: str = "Unknown"
    market_cap: Optional[float] = None           # INR Crores
    current_price: Optional[float] = None        # INR
    shares_outstanding: Optional[float] = None   # Raw count (not Crores)
    currency: str = "INR"

    # --- Income Statement (annual, INR Crores) ---
    revenue_series: List[float] = field(default_factory=list)          # newest → oldest
    net_profit_series: List[float] = field(default_factory=list)
    ebit_series: List[float] = field(default_factory=list)
    interest_expense_series: List[float] = field(default_factory=list)
    operating_income_series: List[float] = field(default_factory=list)
    eps_series: List[float] = field(default_factory=list)

    # --- Balance Sheet (annual, INR Crores) ---
    total_equity_series: List[float] = field(default_factory=list)
    total_debt_series: List[float] = field(default_factory=list)
    current_assets_series: List[float] = field(default_factory=list)
    current_liabilities_series: List[float] = field(default_factory=list)
    book_value_per_share_series: List[float] = field(default_factory=list)
    capital_employed_series: List[float] = field(default_factory=list)   # Total Assets − Current Liabilities
    cash_and_equivalents_series: List[float] = field(default_factory=list)

    # --- Cash Flow (annual, INR Crores) ---
    operating_cash_flow_series: List[float] = field(default_factory=list)
    capex_series: List[float] = field(default_factory=list)              # Capital expenditure (negative)

    # --- Growth histories (percentage, newest → oldest) ---
    revenue_growth_history: List[float] = field(default_factory=list)
    profit_growth_history: List[float] = field(default_factory=list)
    eps_growth_history: List[float] = field(default_factory=list)
    cash_flow_growth_history: List[float] = field(default_factory=list)
    book_value_growth_history: List[float] = field(default_factory=list)
    dividend_growth_history: List[float] = field(default_factory=list)

    # --- Dividend (per share, INR) ---
    dividend_per_share_series: List[float] = field(default_factory=list)
    dividend_years: List[str] = field(default_factory=list)

    # --- Fiscal year labels (from statement columns, newest first) ---
    fiscal_years: List[str] = field(default_factory=list)
    reporting_periods: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unit_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # --- Data Quality ---
    warnings: List[str] = field(default_factory=list)

    # --- Raw yfinance info dict (for valuation multiples) ---
    _raw_info: Dict[str, Any] = field(default_factory=dict, repr=False)


# ==============================================================================
# Service
# ==============================================================================


class DataCleanerService:
    """
    Validates and normalises raw yfinance data into CleanedFinancialData.

    Usage:
        cleaner = DataCleanerService()
        cleaned = cleaner.clean(raw_data)
    """

    def __init__(self, parser: Optional[FinancialParser] = None) -> None:
        self._parser = parser or FinancialParser()
        logger.info("DataCleanerService initialised.")

    def clean(self, raw: RawFinancialData) -> CleanedFinancialData:
        """
        Main entry point. Processes all financial statements.

        This method does NOT raise for missing optional data — instead it
        appends warnings and leaves the corresponding series empty. Fatal
        issues (e.g. empty income statement) are propagated as ValueError.
        """
        cleaned = CleanedFinancialData(
            ticker=raw.ticker,
            exchange=raw.exchange,
        )

        self._extract_company_info(raw, cleaned)
        self._extract_income_statement(raw, cleaned)
        self._extract_balance_sheet(raw, cleaned)
        self._extract_cash_flow(raw, cleaned)
        self._validate_reporting_periods(cleaned, raw)
        self._validate_series_consistency(cleaned)
        self._extract_dividends(raw, cleaned)
        self._build_growth_histories(cleaned)

        logger.info(
            "DataCleanerService: Cleaned %s — %d income years, %d BS years. "
            "%d warning(s).",
            raw.ticker,
            len(cleaned.revenue_series),
            len(cleaned.total_equity_series),
            len(cleaned.warnings),
        )
        return cleaned

    # ------------------------------------------------------------------
    # Section Extractors
    # ------------------------------------------------------------------

    def _extract_company_info(self, raw: RawFinancialData, cleaned: CleanedFinancialData) -> None:
        """Pulls metadata from the .info dict. Never raises — uses defaults."""
        info = raw.info
        fast_info = getattr(raw, "fast_info", {}) or {}
        cleaned._raw_info = dict(info)  # Store for downstream valuation use
        cleaned.company_name = info.get("longName") or info.get("shortName") or "Unknown"
        cleaned.sector = info.get("sector") or "Unknown"
        cleaned.industry = info.get("industry") or "Unknown"
        cleaned.currency = info.get("currency") or fast_info.get("currency") or "INR"

        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or fast_info.get("lastPrice")
            or info.get("previousClose")
        )
        cleaned.current_price = float(price) if price else None
        
        shares = (
            fast_info.get("sharesOutstanding")
            or info.get("sharesOutstanding")
            or fast_info.get("shares")
            or info.get("impliedSharesOutstanding")
        )
        cleaned.shares_outstanding = float(shares) if shares else None

        mc = self._resolve_market_cap_inr(raw, cleaned.current_price)
        cleaned.market_cap = self._to_crores(mc) if mc else None

    def _extract_income_statement(self, raw: RawFinancialData, cleaned: CleanedFinancialData) -> None:
        """Extracts annual income statement series."""
        df = raw.income_stmt
        if df.empty:
            cleaned.warnings.append("Annual income statement unavailable.")
            return

        df = self._parser.sort_columns_newest_first(df)
        cleaned.fiscal_years = self._parser.extract_fiscal_years(df)

        mappings = (
            ("revenue", INCOME_ALIASES["revenue"], "revenue_series", True),
            ("net_profit", INCOME_ALIASES["net_profit"], "net_profit_series", True),
            ("ebit", INCOME_ALIASES["ebit"], "ebit_series", True),
            ("interest_expense", INCOME_ALIASES["interest_expense"], "interest_expense_series", False),
            ("operating_income", INCOME_ALIASES["operating_income"], "operating_income_series", True),
        )
        for key, aliases, attr, required in mappings:
            result = self._parser.extract_series(df, aliases, key, required=required, statement="income_statement")
            setattr(cleaned, attr, result.values)
            self._record_unit_metadata(cleaned, key, result, raw_unit="INR", internal_unit="INR Crores", display_unit="₹ Cr")
            if not result.available and result.reason:
                cleaned.warnings.append(result.reason)

        eps_result = self._parser.extract_eps(df)
        cleaned.eps_series = eps_result.values
        self._record_unit_metadata(cleaned, "eps", eps_result, raw_unit="INR/share", internal_unit="INR/share", display_unit="₹")
        if not eps_result.available and eps_result.reason:
            cleaned.warnings.append(eps_result.reason)

    def _extract_balance_sheet(self, raw: RawFinancialData, cleaned: CleanedFinancialData) -> None:
        """Extracts annual balance sheet series."""
        df = raw.balance_sheet
        if df.empty:
            cleaned.warnings.append("Annual balance sheet unavailable.")
            return

        df = self._parser.sort_columns_newest_first(df)

        balance_mappings = (
            ("total_equity", BALANCE_ALIASES["total_equity"], "total_equity_series", True),
            ("total_debt", BALANCE_ALIASES["total_debt"], "total_debt_series", False),
            ("current_assets", BALANCE_ALIASES["current_assets"], "current_assets_series", False),
            ("current_liabilities", BALANCE_ALIASES["current_liabilities"], "current_liabilities_series", False),
        )
        for key, aliases, attr, required in balance_mappings:
            result = self._parser.extract_series(df, aliases, key, required=required, statement="balance_sheet")
            setattr(cleaned, attr, result.values)
            self._record_unit_metadata(cleaned, key, result, raw_unit="INR", internal_unit="INR Crores", display_unit="₹ Cr")
            if not result.available and result.reason:
                cleaned.warnings.append(result.reason)

        cash_result = self._parser.extract_series(
            df, BALANCE_ALIASES["cash_and_equivalents"], "cash_and_equivalents", required=False, statement="balance_sheet"
        )
        cleaned.cash_and_equivalents_series = cash_result.values
        self._record_unit_metadata(cleaned, "cash_and_equivalents", cash_result, raw_unit="INR", internal_unit="INR Crores", display_unit="₹ Cr")
        if not cash_result.available and cash_result.reason:
            cleaned.warnings.append(cash_result.reason)

        bv_result = self._parser.extract_series(
            df, BALANCE_ALIASES["book_value_per_share"], "book_value_per_share",
            convert_to_crores=False, required=False, statement="balance_sheet",
        )
        cleaned.book_value_per_share_series = bv_result.values
        self._record_unit_metadata(cleaned, "book_value_per_share", bv_result, raw_unit="INR/share", internal_unit="INR/share", display_unit="₹")
        if not bv_result.available and bv_result.reason:
            cleaned.warnings.append(bv_result.reason)

        total_assets_result = self._parser.extract_series(
            df, BALANCE_ALIASES["total_assets"], "total_assets", required=False, statement="balance_sheet",
        )
        total_assets_series = total_assets_result.values
        if not total_assets_result.available and total_assets_result.reason:
            cleaned.warnings.append(total_assets_result.reason)
        if total_assets_series and cleaned.current_liabilities_series:
            length = min(len(total_assets_series), len(cleaned.current_liabilities_series))
            cleaned.capital_employed_series = [
                total_assets_series[i] - cleaned.current_liabilities_series[i]
                for i in range(length)
            ]

    def _extract_cash_flow(self, raw: RawFinancialData, cleaned: CleanedFinancialData) -> None:
        """Extracts annual cash flow series."""
        df = raw.cash_flow
        if df.empty:
            cleaned.warnings.append("Annual cash flow statement unavailable.")
            return

        df = self._parser.sort_columns_newest_first(df)

        for key, aliases, attr, required in (
            ("operating_cash_flow", CASHFLOW_ALIASES["operating_cash_flow"], "operating_cash_flow_series", False),
            ("capex", CASHFLOW_ALIASES["capex"], "capex_series", False),
        ):
            result = self._parser.extract_series(df, aliases, key, required=required, statement="cash_flow")
            setattr(cleaned, attr, result.values)
            self._record_unit_metadata(cleaned, key, result, raw_unit="INR", internal_unit="INR Crores", display_unit="₹ Cr")
            if not result.available and result.reason:
                cleaned.warnings.append(result.reason)

    def _extract_dividends(self, raw: RawFinancialData, cleaned: CleanedFinancialData) -> None:
        """
        Aggregates ticker.dividends into yearly totals (newest first).
        Does not use info dividendRate — only actual payment history.
        """
        dividends = raw.dividends
        if dividends is not None and not dividends.empty:
            yearly = dividends.groupby(dividends.index.year).sum()
            yearly = yearly.sort_index(ascending=False)

            # Drop the current calendar year if it likely has incomplete payouts
            current_year = pd.Timestamp.now().year
            if len(yearly) > 1 and int(yearly.index[0]) == current_year:
                prior_year_total = float(yearly.iloc[1])
                current_year_total = float(yearly.iloc[0])
                if current_year_total < prior_year_total * 0.85:
                    yearly = yearly.iloc[1:]

            cleaned.dividend_per_share_series = [
                round(float(v), 4)
                for v in yearly.values
                if FinancialParser._clean_value(v) is not None and float(v) > 0
            ]
            cleaned.dividend_years = [
                str(int(y)) for y in yearly.index[: len(cleaned.dividend_per_share_series)]
            ]
            if cleaned.dividend_per_share_series:
                return

        cleaned.warnings.append(
            "No Dividend History — this company may not pay dividends, or Yahoo Finance "
            "has no dividend payment record for this ticker."
        )

    # ------------------------------------------------------------------
    # Low-Level Helpers
    # ------------------------------------------------------------------

    def _build_growth_histories(self, cleaned: CleanedFinancialData) -> None:
        """Build explicit growth-percentage histories for the growth metric cards."""
        cleaned.revenue_growth_history = self._build_growth_history(cleaned.revenue_series)
        cleaned.profit_growth_history = self._build_growth_history(cleaned.net_profit_series)
        cleaned.eps_growth_history = self._build_growth_history(cleaned.eps_series)
        cleaned.cash_flow_growth_history = self._build_growth_history(cleaned.operating_cash_flow_series)
        cleaned.book_value_growth_history = self._build_growth_history(cleaned.book_value_per_share_series)
        cleaned.dividend_growth_history = self._build_growth_history(cleaned.dividend_per_share_series)

    def _validate_reporting_periods(self, cleaned: CleanedFinancialData, raw: RawFinancialData) -> None:
        """Warn when income, balance-sheet and cash-flow statements do not share a consistent fiscal period."""
        statements = {
            "income_statement": raw.income_stmt,
            "balance_sheet": raw.balance_sheet,
            "cash_flow": raw.cash_flow,
        }
        normalized = {}
        for name, df in statements.items():
            if df is None or df.empty:
                continue
            cols = list(df.columns)
            normalized[name] = {
                "columns": [str(col) for col in cols],
                "frequency": self._parser._infer_reporting_frequency(cols),
                "latest": self._parser._latest_period_label(cols),
            }

        if not normalized:
            return

        frequencies = {entry["frequency"] for entry in normalized.values()}
        if len(frequencies) > 1:
            cleaned.warnings.append(
                "Reporting periods do not align across statements: mixed annual/quarterly/TTM values detected."
            )

        latest_labels = {name: entry["latest"] for name, entry in normalized.items() if entry.get("latest")}
        if len(set(latest_labels.values())) > 1:
            cleaned.warnings.append(
                "Reporting periods do not align across statements: latest fiscal labels differ."
            )

        cleaned.reporting_periods = normalized

    def _validate_series_consistency(self, cleaned: CleanedFinancialData) -> None:
        """Validate series ordering and uniqueness for downstream ratios."""
        series_map = {
            "revenue": cleaned.revenue_series,
            "net_profit": cleaned.net_profit_series,
            "total_equity": cleaned.total_equity_series,
            "current_assets": cleaned.current_assets_series,
            "operating_cash_flow": cleaned.operating_cash_flow_series,
        }
        for name, values in series_map.items():
            if not values:
                continue
            if len(set(values)) != len(values):
                cleaned.warnings.append(f"{name} series contains duplicate values; historical validation may be affected.")

    @staticmethod
    def _record_unit_metadata(
        cleaned: CleanedFinancialData,
        key: str,
        result,
        *,
        raw_unit: str,
        internal_unit: str,
        display_unit: str,
    ) -> None:
        cleaned.unit_metadata[key] = {
            "raw_unit": raw_unit,
            "conversion_applied": "crores" if internal_unit == "INR Crores" else "none",
            "internal_unit": internal_unit,
            "display_unit": display_unit,
            "statement": getattr(result, "statement", "unknown"),
            "matched_row": getattr(result, "matched_row", None),
            "confidence": getattr(result, "confidence", 0.0),
            "reporting_frequency": getattr(result, "reporting_frequency", "unknown"),
            "reporting_date": getattr(result, "reporting_date", None),
        }

    @staticmethod
    def _build_growth_history(series: List[float]) -> List[float]:
        """Convert an absolute series into a newest-first growth-percentage series."""
        if len(series) < 2:
            return []

        values: List[float] = []
        for idx in range(len(series) - 1):
            prior = series[idx + 1]
            current = series[idx]
            if prior == 0:
                continue
            values.append(round(((current - prior) / abs(prior)) * 100, 2))
        return values

    @staticmethod
    def _resolve_market_cap_inr(
        raw: RawFinancialData,
        current_price: Optional[float],
    ) -> Optional[float]:
        """
        Resolves market capitalisation in absolute INR (not Crores).

        Priority:
            1. fast_info.market_cap (most reliable for live quotes)
            2. info.marketCap
            3. sharesOutstanding × current price
            4. impliedSharesOutstanding × current price
        """
        info = raw.info or {}
        fast_info = getattr(raw, "fast_info", {}) or {}
        candidates: List[float] = []

        # Priority 1: fast_info market cap (stored as marketCap by fetcher)
        for key in ("marketCap", "market_cap"):
            mc = fast_info.get(key)
            if mc is not None:
                mc_val = float(mc)
                if mc_val > 0:
                    candidates.append(mc_val)

        # Priority 2: info marketCap
        info_mc = info.get("marketCap")
        if info_mc is not None:
            mc_val = float(info_mc)
            if mc_val > 0:
                candidates.append(mc_val)

        # Priority 3 & 4: compute from shares × price
        shares = (
            fast_info.get("sharesOutstanding")
            or info.get("sharesOutstanding")
            or fast_info.get("shares")
        )
        if shares and current_price:
            computed = float(shares) * float(current_price)
            if computed > 0:
                candidates.append(computed)

        imputed = info.get("impliedSharesOutstanding")
        if imputed and current_price:
            computed = float(imputed) * float(current_price)
            if computed > 0:
                candidates.append(computed)

        if not candidates:
            return None

        return max(candidates)

    @staticmethod
    def _to_crores(value: float) -> float:
        """Converts absolute INR to INR Crores."""
        return round(value / 1_00_00_000, 4)
