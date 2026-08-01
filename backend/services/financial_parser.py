"""
services/financial_parser.py — Robust Financial Statement Parser
==================================================================
Extracts line items from yfinance DataFrames using alias lists and
normalized fuzzy matching. Never relies on a single row name.

Returns explicit unavailability reasons instead of silent failures.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================================
# Row alias registry — deterministic, ranked mappings
# ==============================================================================

@dataclass
class MappingRule:
    """A single mapping candidate for a financial statement row."""
    alias: str
    confidence: float
    statement: str = "unknown"


INCOME_ALIASES: Dict[str, List[str]] = {
    "revenue": [
        "Total Revenue",
        "Revenue",
        "Net Revenue",
        "Revenues",
        "Sales",
        "Net Sales",
        "Operating Revenue",
        "Total Operating Revenue",
    ],
    "net_profit": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Applicable To Common Shares",
        "Normalized Income",
        "Net Profit",
        "Profit After Tax",
        "Net Income From Continuing Operations",
    ],
    "ebit": [
        "EBIT",
        "Operating Income",
        "Operating Profit",
        "Operating Profit Loss",
        "Operating Earnings",
    ],
    "interest_expense": [
        "Interest Expense",
        "Interest Expense Non Operating",
        "Interest Expense Non Operating Income",
        "Finance Cost",
        "Finance Expense",
        "Interest And Debt Expense",
        "Interest Paid",
        "Interest Paid Cfo",
    ],
    "operating_income": [
        "Operating Income",
        "Operating Profit",
        "Operating Profit Loss",
        "EBIT",
        "Total Operating Income As Reported",
    ],
    "eps": [
        "Diluted EPS",
        "Basic EPS",
        "Diluted Average Shares",
    ],
}

BALANCE_ALIASES: Dict[str, List[str]] = {
    "total_equity": [
        "Stockholders Equity",
        "Common Stock Equity",
        "Total Equity Gross Minority Interest",
        "Shareholder Equity",
        "Total Equity",
        "Total Stockholders Equity",
        "Common Equity",
    ],
    "total_debt": [
        "Total Debt",
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation",
        "Long Term Debt And Capital Lease Obligation Total",
    ],
    "current_assets": [
        "Current Assets",
        "Total Current Assets",
    ],
    "current_liabilities": [
        "Current Liabilities",
        "Total Current Liabilities Net Minority Interest",
        "Current Liabilities Net Minority Interest",
        "Total Current Liabilities",
    ],
    "book_value_per_share": [
        "Book Value Per Share",
    ],
    "total_assets": [
        "Total Assets",
    ],
    "cash_and_equivalents": [
        "Cash And Cash Equivalents",
        "Cash",
        "Cash Equivalents",
        "Cash And Equivalents",
    ],
}

CASHFLOW_ALIASES: Dict[str, List[str]] = {
    "operating_cash_flow": [
        "Operating Cash Flow",
        "Cash Flow From Operations",
        "Net Cash Provided By Operating Activities",
        "Total Cash From Operating Activities",
    ],
    "capex": [
        "Capital Expenditure",
        "Capital Expenditures",
        "Purchase Of Property Plant And Equipment",
        "Purchase Of PPE",
        "Capital Expenditure Reported",
    ],
}


@dataclass
class ParseResult:
    """Outcome of parsing a single line item from a statement."""

    key: str
    values: List[float] = field(default_factory=list)
    matched_row: Optional[str] = None
    available: bool = False
    reason: str = ""
    statement: str = "unknown"
    confidence: float = 0.0
    reporting_frequency: str = "unknown"
    reporting_date: Optional[str] = None


class FinancialParser:
    """
    Parses yfinance financial statement DataFrames into numeric series.

    Usage:
        parser = FinancialParser()
        result = parser.extract_series(df, INCOME_ALIASES["revenue"], "revenue")
    """

    PER_SHARE_KEYS = {"eps", "book_value_per_share"}

    def extract_series(
        self,
        df: pd.DataFrame,
        aliases: List[str],
        key: str,
        *,
        convert_to_crores: bool = True,
        required: bool = True,
        statement: Optional[str] = None,
    ) -> ParseResult:
        """
        Tries each alias (exact then fuzzy) and returns cleaned values newest-first.
        """
        if df is None or df.empty:
            return ParseResult(
                key=key,
                available=False,
                reason=f"{key.replace('_', ' ').title()} unavailable — statement is empty.",
            )

        candidates = self._build_mapping_candidates(key, aliases, statement)
        matched_label, confidence = self._find_row_label(df, candidates)
        if matched_label is None:
            tried = ", ".join(aliases[:4])
            suffix = "..." if len(aliases) > 4 else ""
            return ParseResult(
                key=key,
                available=False,
                reason=(
                    f"{key.replace('_', ' ').title()} unavailable — none of the expected "
                    f"row labels matched (tried: {tried}{suffix})."
                ),
                statement=statement or "unknown",
                confidence=0.0,
                reporting_frequency=self._infer_reporting_frequency(df.columns),
                reporting_date=self._latest_period_label(df.columns),
            )

        # Only a label explicitly containing "per share" is valid here.
        # This blocks both tangible book value and generic book-value rows,
        # which are absolute balance-sheet amounts.  The guard is needed
        # because generic fuzzy matching permits substring matches.
        normalized_matched_label = self._normalize_label(matched_label)
        if key == "book_value_per_share" and (
            "tangible" in normalized_matched_label or "pershare" not in normalized_matched_label
        ):
            return ParseResult(
                key=key,
                matched_row=matched_label,
                available=False,
                reason="Book Value Per Share unavailable — tangible book value is not a per-share metric.",
                statement=statement or "unknown",
                confidence=confidence,
                reporting_frequency=self._infer_reporting_frequency(df.columns),
                reporting_date=self._latest_period_label(df.columns),
            )

        raw_series = df.loc[matched_label]
        per_share = key in self.PER_SHARE_KEYS or not convert_to_crores
        values: List[float] = []

        for v in raw_series:
            cleaned = self._clean_value(v)
            if cleaned is None:
                continue
            if per_share:
                values.append(round(float(cleaned), 4))
            else:
                values.append(self._to_crores(float(cleaned)))

        if not values:
            return ParseResult(
                key=key,
                matched_row=matched_label,
                available=False,
                reason=(
                    f"{key.replace('_', ' ').title()} row '{matched_label}' found but all "
                    "period values were missing or invalid."
                ),
            )

        # Reject EPS if we accidentally matched share count
        if key == "eps" and matched_label == "Diluted Average Shares":
            return ParseResult(
                key=key,
                matched_row=matched_label,
                available=False,
                reason="EPS unavailable — only share count row found, not earnings per share.",
            )

        return ParseResult(
            key=key,
            values=values,
            matched_row=matched_label,
            available=True,
            reason="",
            statement=statement or "unknown",
            confidence=confidence,
            reporting_frequency=self._infer_reporting_frequency(df.columns),
            reporting_date=self._latest_period_label(df.columns),
        )

    def extract_eps(self, df: pd.DataFrame) -> ParseResult:
        """EPS extraction with share-count guard."""
        eps_aliases = [a for a in INCOME_ALIASES["eps"] if a != "Diluted Average Shares"]
        result = self.extract_series(
            df, eps_aliases + ["Diluted EPS", "Basic EPS"], "eps",
            convert_to_crores=False, required=True, statement="income_statement",
        )
        if not result.available:
            result.reason = result.reason or "EPS unavailable — no Diluted/Basic EPS row in income statement."
        return result

    def sort_columns_newest_first(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures fiscal columns are ordered newest → oldest."""
        try:
            return df[sorted(df.columns, reverse=True)]
        except TypeError:
            return df

    def extract_fiscal_years(self, df: pd.DataFrame) -> List[str]:
        years: List[str] = []
        for col in df.columns:
            if hasattr(col, "year"):
                years.append(str(col.year))
            else:
                years.append(str(col))
        return years

    @staticmethod
    def _normalize_label(label: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(label).lower())

    def _build_mapping_candidates(
        self,
        key: str,
        aliases: List[str],
        statement: Optional[str],
    ) -> List[MappingRule]:
        """Build ranked, deterministic mapping candidates for a metric."""
        statement_name = statement or "unknown"
        if key == "ebit":
            return [
                MappingRule("EBIT", 0.95, statement_name),
                MappingRule("Operating Income", 0.8, statement_name),
                MappingRule("Operating Profit", 0.75, statement_name),
            ]
        if key == "total_debt":
            return [
                MappingRule("Total Debt", 0.95, statement_name),
                MappingRule("Long Term Debt", 0.75, statement_name),
                MappingRule("Long Term Debt And Capital Lease Obligation", 0.7, statement_name),
            ]
        if key == "cash_and_equivalents":
            return [
                MappingRule("Cash And Cash Equivalents", 0.95, statement_name),
                MappingRule("Cash", 0.8, statement_name),
                MappingRule("Cash Equivalents", 0.75, statement_name),
            ]

        return [
            MappingRule(alias, max(0.55, 0.95 - (idx * 0.08)), statement_name)
            for idx, alias in enumerate(aliases)
        ]

    def _find_row_label(self, df: pd.DataFrame, aliases: List[MappingRule]) -> Tuple[Optional[str], float]:
        """Exact match first, then normalized match, then substring fuzzy match."""
        index_labels = list(df.index)

        # Pass 1: exact string match
        for rule in aliases:
            if rule.alias in index_labels:
                return rule.alias, rule.confidence

        # Pass 2: normalized exact match
        norm_map: Dict[str, str] = {
            self._normalize_label(idx): str(idx) for idx in index_labels
        }
        for rule in aliases:
            norm_alias = self._normalize_label(rule.alias)
            if norm_alias in norm_map:
                return norm_map[norm_alias], max(rule.confidence - 0.05, 0.55)

        # Pass 3: substring fuzzy match (alias contained in row or vice versa)
        for rule in aliases:
            norm_alias = self._normalize_label(rule.alias)
            if len(norm_alias) < 4:
                continue
            for norm_row, original in norm_map.items():
                if norm_alias in norm_row or norm_row in norm_alias:
                    logger.debug("Fuzzy matched '%s' → '%s'", rule.alias, original)
                    return original, max(rule.confidence - 0.1, 0.5)

        return None, 0.0

    @staticmethod
    def _infer_reporting_frequency(columns: List[Any]) -> str:
        if columns is None:
            return "unknown"
        if hasattr(columns, "empty") and columns.empty:
            return "unknown"
        if not list(columns):
            return "unknown"
        labels = [str(col).lower() for col in columns]
        if any("ttm" in label for label in labels):
            return "ttm"
        if any("q" in label for label in labels):
            return "quarterly"
        return "annual"

    @staticmethod
    def _latest_period_label(columns: List[Any]) -> Optional[str]:
        if columns is None:
            return None
        if hasattr(columns, "empty") and columns.empty:
            return None
        items = list(columns)
        if not items:
            return None
        return str(items[0])

    @staticmethod
    def _to_crores(value: float) -> float:
        return round(value / 1_00_00_000, 4)

    @staticmethod
    def _clean_value(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            f = float(value)
            if np.isnan(f) or np.isinf(f):
                return None
            return f
        except (TypeError, ValueError):
            return None
