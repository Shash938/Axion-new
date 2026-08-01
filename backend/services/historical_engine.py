"""
services/historical_engine.py — Historical Data Engine
========================================================
Builds unified metric detail objects with history, YoY, CAGR, trend,
and benchmark context from cleaned financial data.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from services.benchmark_engine import BenchmarkEngine, BenchmarkResult
from services.data_cleaner import CleanedFinancialData
from services.metric_utils import cagr, cagr_3y, cagr_5y, safe_divide, yoy_growth
from services.trend_engine import TrendDirection, TrendEngine

logger = logging.getLogger(__name__)


@dataclass
class HistoryPoint:
    """Single year in a metric's history."""
    year: str
    value: float


@dataclass
class MetricDetail:
    """
    Unified structure for every metric exposed via the API.
    Matches the institutional research report format.
    """
    metric: str
    metric_key: str
    display_name: str
    unit: str
    history: List[HistoryPoint] = field(default_factory=list)
    latest_value: Optional[float] = None
    yoy: Optional[float] = None
    cagr3: Optional[float] = None
    cagr5: Optional[float] = None
    trend: str = TrendDirection.UNAVAILABLE.value
    benchmark_label: Optional[str] = None
    benchmark_summary: Optional[str] = None
    data_available: bool = False


class HistoricalEngine:
    """
    Transforms CleanedFinancialData into a dict of MetricDetail objects.
    Reuses parsed statements — no duplicate fetching.
    """

    def __init__(
        self,
        trend_engine: Optional[TrendEngine] = None,
        benchmark_engine: Optional[BenchmarkEngine] = None,
    ) -> None:
        self._trend = trend_engine or TrendEngine()
        self._benchmark = benchmark_engine or BenchmarkEngine()

    def build_all(self, data: CleanedFinancialData) -> Dict[str, MetricDetail]:
        """Builds metric details for all supported metrics."""
        years = data.fiscal_years or self._default_years(data)
        sector = data.sector

        details: Dict[str, MetricDetail] = {}

        # Raw absolute metrics (INR Crores unless per-share)
        self._add_absolute(details, "revenue", "Revenue", "₹ Cr", data.revenue_series, years, sector)
        self._add_absolute(details, "net_profit", "Net Profit", "₹ Cr", data.net_profit_series, years, sector)
        self._add_absolute(details, "operating_income", "Operating Income", "₹ Cr", data.operating_income_series, years, sector)
        self._add_absolute(details, "ebit", "EBIT", "₹ Cr", data.ebit_series, years, sector)
        self._add_absolute(details, "debt", "Total Debt", "₹ Cr", data.total_debt_series, years, sector)
        self._add_absolute(details, "equity", "Shareholder Equity", "₹ Cr", data.total_equity_series, years, sector)
        self._add_absolute(details, "current_assets", "Current Assets", "₹ Cr", data.current_assets_series, years, sector)
        self._add_absolute(details, "current_liabilities", "Current Liabilities", "₹ Cr", data.current_liabilities_series, years, sector)
        self._add_absolute(details, "operating_cash_flow", "Operating Cash Flow", "₹ Cr", data.operating_cash_flow_series, years, sector)
        self._add_absolute(details, "capex", "Capital Expenditure", "₹ Cr", data.capex_series, years, sector, abs_values=True)
        self._add_absolute(details, "interest_expense", "Interest Expense", "₹ Cr", data.interest_expense_series, years, sector, abs_values=True)
        self._add_absolute(details, "capital_employed", "Capital Employed", "₹ Cr", data.capital_employed_series, years, sector)

        # Per-share metrics (no crore conversion)
        self._add_absolute(details, "eps", "Earnings Per Share", "₹", data.eps_series, years, sector, per_share=True)
        self._add_absolute(details, "book_value", "Book Value Per Share", "₹", data.book_value_per_share_series, years, sector, per_share=True)

        # Dividend history (yearly aggregated)
        div_years = data.dividend_years or years[: len(data.dividend_per_share_series)]
        self._add_absolute(
            details, "dividend", "Dividend Per Share", "₹",
            data.dividend_per_share_series, div_years, sector, per_share=True,
        )
        if details.get("dividend") and details["dividend"].data_available:
            div_bench = self._benchmark.evaluate("dividend", details["dividend"].latest_value, sector)
            details["dividend"].benchmark_label = div_bench.label if div_bench else None
            details["dividend"].benchmark_summary = div_bench.band_summary if div_bench else None

        # Derived ratio series (computed per year)
        self._add_derived_ratio(details, "operating_margin", "Operating Margin", "%",
            data.operating_income_series, data.revenue_series, years, sector, as_pct=True)
        self._add_derived_ratio(details, "net_margin", "Net Margin", "%",
            data.net_profit_series, data.revenue_series, years, sector, as_pct=True)
        self._add_derived_ratio(details, "roe", "Return on Equity", "%",
            data.net_profit_series, data.total_equity_series, years, sector, as_pct=True)
        self._add_derived_ratio(details, "roce", "Return on Capital Employed", "%",
            data.ebit_series, data.capital_employed_series, years, sector, as_pct=True)
        self._add_derived_ratio(details, "debt_to_equity", "Debt to Equity", "x",
            data.total_debt_series, data.total_equity_series, years, sector)
        self._add_derived_ratio(details, "current_ratio", "Current Ratio", "x",
            data.current_assets_series, data.current_liabilities_series, years, sector)
        self._add_interest_coverage(details, data, years, sector)
        self._add_fcf(details, data, years, sector)

        # Growth metrics (prefer explicit growth histories, fall back to derived values)
        growth_map = {
            "revenue_growth": ("Revenue Growth", data.revenue_growth_history, data.revenue_series),
            "profit_growth": ("Profit Growth", data.profit_growth_history, data.net_profit_series),
            "eps_growth": ("EPS Growth", data.eps_growth_history, data.eps_series),
            "cash_flow_growth": ("Cash Flow Growth", data.cash_flow_growth_history, data.operating_cash_flow_series),
            "book_value_growth": ("Book Value Growth", data.book_value_growth_history, data.book_value_per_share_series),
            "dividend_growth": ("Dividend Growth", data.dividend_growth_history, data.dividend_per_share_series),
        }
        for key, (name, history_series, fallback_series) in growth_map.items():
            self._add_growth_metric(details, key, name, history_series, fallback_series, years, sector)

        # Point-in-time company metrics
        mc_bench = self._benchmark.evaluate("market_cap", data.market_cap, sector) if data.market_cap else None
        details["market_cap"] = MetricDetail(
            metric="market_cap",
            metric_key="market_cap",
            display_name="Market Cap",
            unit="₹ Cr",
            latest_value=data.market_cap,
            data_available=data.market_cap is not None and data.market_cap > 0,
            benchmark_label=mc_bench.label if mc_bench else None,
            benchmark_summary=mc_bench.band_summary if mc_bench else None,
        )
        details["current_price"] = MetricDetail(
            metric="current_price",
            metric_key="current_price",
            display_name="Current Price",
            unit="₹",
            latest_value=data.current_price,
            data_available=data.current_price is not None,
        )

        return details

    def _default_years(self, data: CleanedFinancialData) -> List[str]:
        n = max(
            len(data.revenue_series),
            len(data.net_profit_series),
            1,
        )
        return [f"Y-{i}" for i in range(n)]

    def _add_absolute(
        self,
        details: Dict[str, MetricDetail],
        key: str,
        name: str,
        unit: str,
        series: List[float],
        years: List[str],
        sector: str,
        per_share: bool = False,
        abs_values: bool = False,
        no_dividend_message: bool = False,
    ) -> None:
        if not series:
            details[key] = MetricDetail(
                metric=key, metric_key=key, display_name=name, unit=unit,
                trend=TrendDirection.UNAVAILABLE.value,
                data_available=False,
            )
            return

        values = [abs(v) if abs_values else v for v in series]
        history = self._build_history(values, years)
        trend = self._trend.classify_for_metric(key, values)

        details[key] = MetricDetail(
            metric=key,
            metric_key=key,
            display_name=name,
            unit=unit,
            history=history,
            latest_value=values[0] if values else None,
            yoy=yoy_growth(values),
            cagr3=cagr_3y(values),
            cagr5=cagr_5y(values),
            trend=trend.value,
            data_available=True,
        )

    def _add_derived_ratio(
        self,
        details: Dict[str, MetricDetail],
        key: str,
        name: str,
        unit: str,
        numerators: List[float],
        denominators: List[float],
        years: List[str],
        sector: str,
        as_pct: bool = False,
    ) -> None:
        length = min(len(numerators), len(denominators))
        if length == 0:
            details[key] = MetricDetail(
                metric=key, metric_key=key, display_name=name, unit=unit,
                data_available=False,
            )
            return

        values = []
        for i in range(length):
            val = safe_divide(numerators[i], denominators[i], as_pct=as_pct)
            if val is not None:
                values.append(val)

        if not values:
            details[key] = MetricDetail(
                metric=key, metric_key=key, display_name=name, unit=unit,
                data_available=False,
            )
            return

        bench = self._benchmark.evaluate(key, values[0], sector)
        history = self._build_history(values, years[: len(values)])
        trend = self._trend.classify_for_metric(key, values)

        details[key] = MetricDetail(
            metric=key,
            metric_key=key,
            display_name=name,
            unit=unit,
            history=history,
            latest_value=values[0],
            yoy=yoy_growth(values),
            cagr3=cagr_3y(values),
            cagr5=cagr_5y(values),
            trend=trend.value,
            benchmark_label=bench.label if bench else None,
            benchmark_summary=bench.band_summary if bench else None,
            data_available=True,
        )

    def _add_interest_coverage(
        self, details: Dict[str, MetricDetail], data: CleanedFinancialData,
        years: List[str], sector: str,
    ) -> None:
        ebit = data.ebit_series
        interest = data.interest_expense_series
        length = min(len(ebit), len(interest))
        values = []
        for i in range(length):
            int_abs = abs(interest[i])
            if int_abs == 0:
                values.append(99.0)
            else:
                val = safe_divide(ebit[i], int_abs)
                if val is not None:
                    values.append(val)

        if not values:
            details["interest_coverage"] = MetricDetail(
                metric="interest_coverage", metric_key="interest_coverage",
                display_name="Interest Coverage", unit="x", data_available=False,
            )
            return

        bench = self._benchmark.evaluate("interest_coverage", values[0], sector)
        details["interest_coverage"] = MetricDetail(
            metric="interest_coverage",
            metric_key="interest_coverage",
            display_name="Interest Coverage",
            unit="x",
            history=self._build_history(values, years[: len(values)]),
            latest_value=values[0],
            yoy=yoy_growth(values),
            cagr3=cagr_3y(values),
            cagr5=cagr_5y(values),
            trend=self._trend.classify_for_metric("interest_coverage", values).value,
            benchmark_label=bench.label if bench else None,
            benchmark_summary=bench.band_summary if bench else None,
            data_available=True,
        )

    def _add_fcf(
        self, details: Dict[str, MetricDetail], data: CleanedFinancialData,
        years: List[str], sector: str,
    ) -> None:
        ocf = data.operating_cash_flow_series
        capex = data.capex_series
        length = len(ocf)
        fcf_values = []
        for i in range(length):
            capex_val = abs(capex[i]) if i < len(capex) else 0.0
            fcf_values.append(round(ocf[i] - capex_val, 2))

        if not fcf_values:
            details["free_cash_flow"] = MetricDetail(
                metric="free_cash_flow", metric_key="free_cash_flow",
                display_name="Free Cash Flow", unit="₹ Cr", data_available=False,
            )
            details["fcf_margin"] = MetricDetail(
                metric="fcf_margin", metric_key="fcf_margin",
                display_name="FCF Margin", unit="%", data_available=False,
            )
            return

        # FCF margin series
        rev = data.revenue_series
        margin_values = []
        for i in range(min(len(fcf_values), len(rev))):
            val = safe_divide(fcf_values[i], rev[i], as_pct=True)
            if val is not None:
                margin_values.append(val)

        bench = self._benchmark.evaluate("fcf_margin", margin_values[0] if margin_values else None, sector)

        details["free_cash_flow"] = MetricDetail(
            metric="free_cash_flow",
            metric_key="free_cash_flow",
            display_name="Free Cash Flow",
            unit="₹ Cr",
            history=self._build_history(fcf_values, years[: len(fcf_values)]),
            latest_value=fcf_values[0],
            yoy=yoy_growth(fcf_values),
            cagr3=cagr_3y(fcf_values),
            cagr5=cagr_5y(fcf_values),
            trend=self._trend.classify_cash_flow(fcf_values).value,
            data_available=True,
        )

        details["fcf_margin"] = MetricDetail(
            metric="fcf_margin",
            metric_key="fcf_margin",
            display_name="FCF Margin",
            unit="%",
            history=self._build_history(margin_values, years[: len(margin_values)]),
            latest_value=margin_values[0] if margin_values else None,
            yoy=yoy_growth(margin_values) if margin_values else None,
            cagr3=cagr_3y(margin_values) if margin_values else None,
            cagr5=cagr_5y(margin_values) if margin_values else None,
            trend=self._trend.classify_margin(margin_values).value if margin_values else TrendDirection.UNAVAILABLE.value,
            benchmark_label=bench.label if bench else None,
            benchmark_summary=bench.band_summary if bench else None,
            data_available=bool(margin_values),
        )

    def _add_growth_metric(
        self,
        details: Dict[str, MetricDetail],
        key: str,
        name: str,
        growth_history: Optional[List[float]],
        fallback_series: Optional[List[float]],
        years: List[str],
        sector: str,
    ) -> None:
        if growth_history and len(growth_history) >= 2:
            growth_series = list(growth_history)
        elif fallback_series and len(fallback_series) >= 2:
            growth_series = self._build_growth_series(fallback_series)
        else:
            details[key] = MetricDetail(
                metric=key, metric_key=key, display_name=name, unit="%",
                data_available=False,
                trend=TrendDirection.UNAVAILABLE.value,
            )
            return

        growth_value = growth_series[0] if growth_series else None
        cagr_value = cagr(fallback_series) if fallback_series and len(fallback_series) >= 2 else None
        if cagr_value is None and growth_series:
            cagr_value = cagr(growth_series)
        bench = self._benchmark.evaluate(key, cagr_value if cagr_value is not None else growth_value, sector)
        if cagr_value is not None:
            if cagr_value > 3.0:
                trend = TrendDirection.GROWING
            elif cagr_value < -3.0:
                trend = TrendDirection.DECLINING
            else:
                trend = TrendDirection.STABLE
        else:
            trend = self._trend.classify_growth(growth_series) if growth_series else TrendDirection.UNAVAILABLE

        details[key] = MetricDetail(
            metric=key,
            metric_key=key,
            display_name=name,
            unit="%",
            history=self._build_history(growth_series, years[: len(growth_series)]),
            latest_value=cagr_value,
            yoy=growth_value,
            cagr3=cagr(fallback_series, periods=3) if fallback_series and len(fallback_series) >= 4 else None,
            cagr5=cagr(fallback_series, periods=5) if fallback_series and len(fallback_series) >= 6 else None,
            trend=trend.value,
            benchmark_label=bench.label if bench else None,
            benchmark_summary=bench.band_summary if bench else None,
            data_available=bool(growth_series),
        )

    @staticmethod
    def _build_growth_series(series: List[float]) -> List[float]:
        """Convert an absolute series into year-on-year growth percentages."""
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
    def _build_history(values: List[float], years: List[str]) -> List[HistoryPoint]:
        """Builds history points, limiting to 5 most recent years for display."""
        points = []
        for i, val in enumerate(values[:5]):
            year_label = years[i] if i < len(years) else f"Y-{i}"
            if hasattr(year_label, "year"):
                year_label = str(year_label.year)
            else:
                year_label = str(year_label)
            points.append(HistoryPoint(year=year_label, value=round(val, 2)))
        return list(reversed(points))  # oldest → newest for table display
