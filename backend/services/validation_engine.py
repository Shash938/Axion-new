"""
services/validation_engine.py — Pre-Calculation Data Validation
=================================================================
Validates cleaned financial data before ratio calculation.
Surfaces warnings for impossible or suspicious values — never fabricates data.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from services.data_cleaner import CleanedFinancialData

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Result of validating cleaned financial data."""

    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    blocked_metrics: List[str] = field(default_factory=list)


class ValidationEngine:
    """
    Validates financial inputs before ratio calculation.

    Usage:
        engine = ValidationEngine()
        report = engine.validate(cleaned_data)
    """

    def validate(self, data: CleanedFinancialData) -> ValidationReport:
        report = ValidationReport()
        existing = set(data.warnings)

        self._check_series("Revenue", data.revenue_series, allow_negative=False, report=report)
        self._check_series("Net profit", data.net_profit_series, allow_negative=True, report=report)
        self._check_series("Total debt", data.total_debt_series, allow_negative=False, report=report)
        self._check_series("Shareholder equity", data.total_equity_series, allow_negative=True, report=report)
        self._check_series("Operating cash flow", data.operating_cash_flow_series, allow_negative=True, report=report)
        self._check_series("EPS", data.eps_series, allow_negative=True, report=report, per_share=True)
        self._check_interest(data, report)
        self._check_market_cap(data, report)
        self._check_dividends(data, report)

        # Deduplicate while preserving order
        merged = list(data.warnings) + [w for w in report.warnings if w not in existing]
        report.warnings = merged
        data.warnings = merged

        if report.blocked_metrics:
            report.is_valid = False

        logger.info(
            "ValidationEngine: %s — %d warning(s), %d blocked metric(s).",
            data.ticker,
            len(report.warnings),
            len(report.blocked_metrics),
        )
        return report

    @staticmethod
    def _check_series(
        label: str,
        series: List[float],
        *,
        allow_negative: bool,
        report: ValidationReport,
        per_share: bool = False,
    ) -> None:
        if not series:
            return

        latest = series[0]
        unit = "₹" if per_share else "₹ Cr"

        if latest == 0:
            report.warnings.append(
                f"{label} is zero in the latest period — related ratios may be unavailable."
            )
            return

        if not allow_negative and latest < 0:
            report.warnings.append(
                f"{label} is negative ({latest:,.2f} {unit}) — this is unusual and ratios may be unreliable."
            )

        if not per_share and abs(latest) > 50_00_000:
            report.warnings.append(
                f"{label} value ({latest:,.0f} Cr) appears abnormally large — verify data source units."
            )

        if per_share and abs(latest) > 100_000:
            report.warnings.append(
                f"{label} per share ({latest:,.2f}) appears abnormally high — possible unit mismatch."
            )

    @staticmethod
    def _check_interest(data: CleanedFinancialData, report: ValidationReport) -> None:
        if not data.interest_expense_series:
            return
        latest = abs(data.interest_expense_series[0])
        if data.ebit_series and data.ebit_series[0] > 0 and latest > data.ebit_series[0] * 5:
            report.warnings.append(
                "Interest expense exceeds EBIT by a wide margin — interest coverage will be very low."
            )

    @staticmethod
    def _check_market_cap(data: CleanedFinancialData, report: ValidationReport) -> None:
        if data.market_cap is None:
            report.warnings.append(
                "Market cap could not be resolved from Yahoo Finance (fast_info or info fields missing)."
            )
        elif data.market_cap <= 0:
            report.warnings.append("Market cap resolved to zero — price or shares data may be missing.")

    @staticmethod
    def _check_dividends(data: CleanedFinancialData, report: ValidationReport) -> None:
        if data.dividend_per_share_series:
            return
        # Only add if not already present from data cleaner
        msg = "No dividend payment history found for this ticker on Yahoo Finance."
        if msg not in report.warnings:
            pass  # data_cleaner already adds this
