"""
config/benchmark_rules.py — Professional Benchmark Definitions
==============================================================
Sector-aware benchmark tables. Default benchmarks apply when no sector
override exists. Architecture supports future sector-specific scoring.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BenchmarkBand:
    """A single benchmark tier with label and optional numeric bounds."""
    label: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""


@dataclass(frozen=True)
class MetricBenchmark:
    """Benchmark definition for one metric."""
    key: str
    display_name: str
    unit: str
    bands: Tuple[BenchmarkBand, ...]
    lower_is_better: bool = False


def _default_benchmarks() -> Dict[str, MetricBenchmark]:
    """Default (cross-sector) benchmark tables."""
    return {
        "roe": MetricBenchmark(
            key="roe",
            display_name="Return on Equity",
            unit="%",
            bands=(
                BenchmarkBand("Excellent", min_value=20, description=">20%"),
                BenchmarkBand("Good", min_value=15, max_value=20, description="15–20%"),
                BenchmarkBand("Average", min_value=10, max_value=15, description="10–15%"),
                BenchmarkBand("Weak", max_value=10, description="<10%"),
            ),
        ),
        "roce": MetricBenchmark(
            key="roce",
            display_name="Return on Capital Employed",
            unit="%",
            bands=(
                BenchmarkBand("Excellent", min_value=20, description=">20%"),
                BenchmarkBand("Good", min_value=15, max_value=20, description="15–20%"),
                BenchmarkBand("Average", min_value=10, max_value=15, description="10–15%"),
                BenchmarkBand("Weak", max_value=10, description="<10%"),
            ),
        ),
        "operating_margin": MetricBenchmark(
            key="operating_margin",
            display_name="Operating Margin",
            unit="%",
            bands=(
                BenchmarkBand("Excellent", min_value=25, description=">25%"),
                BenchmarkBand("Good", min_value=20, max_value=25, description="20–25%"),
                BenchmarkBand("Average", min_value=10, max_value=20, description="10–20%"),
                BenchmarkBand("Weak", max_value=10, description="<10%"),
            ),
        ),
        "net_margin": MetricBenchmark(
            key="net_margin",
            display_name="Net Profit Margin",
            unit="%",
            bands=(
                BenchmarkBand("Excellent", min_value=20, description=">20%"),
                BenchmarkBand("Good", min_value=15, max_value=20, description="15–20%"),
                BenchmarkBand("Average", min_value=8, max_value=15, description="8–15%"),
                BenchmarkBand("Weak", max_value=8, description="<8%"),
            ),
        ),
        "debt_to_equity": MetricBenchmark(
            key="debt_to_equity",
            display_name="Debt to Equity",
            unit="x",
            lower_is_better=True,
            bands=(
                BenchmarkBand("Excellent", max_value=0.3, description="<0.3"),
                BenchmarkBand("Good", max_value=0.6, description="<0.6"),
                BenchmarkBand("Average", max_value=1.0, description="<1.0"),
                BenchmarkBand("Poor", max_value=2.0, description="1–2"),
                BenchmarkBand("Very Poor", min_value=2.0, description=">2"),
            ),
        ),
        "current_ratio": MetricBenchmark(
            key="current_ratio",
            display_name="Current Ratio",
            unit="x",
            bands=(
                BenchmarkBand("Excellent", min_value=2.5, description=">2.5"),
                BenchmarkBand("Good", min_value=2.0, max_value=2.5, description="2.0–2.5"),
                BenchmarkBand("Average", min_value=1.5, max_value=2.0, description="1.5–2.0"),
                BenchmarkBand("Weak", max_value=1.5, description="<1.5"),
            ),
        ),
        "interest_coverage": MetricBenchmark(
            key="interest_coverage",
            display_name="Interest Coverage",
            unit="x",
            bands=(
                BenchmarkBand("Excellent", min_value=10, description=">10x"),
                BenchmarkBand("Good", min_value=5, max_value=10, description="5–10x"),
                BenchmarkBand("Average", min_value=3, max_value=5, description="3–5x"),
                BenchmarkBand("Weak", max_value=3, description="<3x"),
            ),
        ),
        "revenue_growth": MetricBenchmark(
            key="revenue_growth",
            display_name="Revenue Growth",
            unit="%",
            bands=(
                BenchmarkBand("Excellent", min_value=15, description=">15% CAGR"),
                BenchmarkBand("Very Good", min_value=10, max_value=15, description="10–15%"),
                BenchmarkBand("Good", min_value=7, max_value=10, description="7–10%"),
                BenchmarkBand("Average", min_value=5, max_value=7, description="5–7%"),
                BenchmarkBand("Weak", min_value=3, max_value=5, description="3–5%"),
                BenchmarkBand("Poor", max_value=3, description="<3%"),
            ),
        ),
        "profit_growth": MetricBenchmark(
            key="profit_growth",
            display_name="Profit Growth",
            unit="%",
            bands=(
                BenchmarkBand("Excellent", min_value=15, description=">15% CAGR"),
                BenchmarkBand("Very Good", min_value=10, max_value=15, description="10–15%"),
                BenchmarkBand("Good", min_value=7, max_value=10, description="7–10%"),
                BenchmarkBand("Average", min_value=5, max_value=7, description="5–7%"),
                BenchmarkBand("Weak", min_value=3, max_value=5, description="3–5%"),
                BenchmarkBand("Poor", max_value=3, description="<3%"),
            ),
        ),
        "eps_growth": MetricBenchmark(
            key="eps_growth",
            display_name="EPS Growth",
            unit="%",
            bands=(
                BenchmarkBand("Excellent", min_value=12, description=">12% CAGR"),
                BenchmarkBand("Very Good", min_value=8, max_value=12, description="8–12%"),
                BenchmarkBand("Good", min_value=5, max_value=8, description="5–8%"),
                BenchmarkBand("Average", min_value=3, max_value=5, description="3–5%"),
                BenchmarkBand("Weak", min_value=1, max_value=3, description="1–3%"),
                BenchmarkBand("Poor", max_value=1, description="<1%"),
            ),
        ),
        "fcf_margin": MetricBenchmark(
            key="fcf_margin",
            display_name="FCF Margin",
            unit="%",
            bands=(
                BenchmarkBand("Excellent", min_value=15, description=">15%"),
                BenchmarkBand("Good", min_value=10, max_value=15, description="10–15%"),
                BenchmarkBand("Average", min_value=5, max_value=10, description="5–10%"),
                BenchmarkBand("Weak", max_value=5, description="<5%"),
            ),
        ),
        "cash_flow_growth": MetricBenchmark(
            key="cash_flow_growth",
            display_name="Cash Flow Growth",
            unit="%",
            bands=(
                BenchmarkBand("Excellent", min_value=12, description=">12% CAGR"),
                BenchmarkBand("Very Good", min_value=8, max_value=12, description="8–12%"),
                BenchmarkBand("Good", min_value=5, max_value=8, description="5–8%"),
                BenchmarkBand("Average", min_value=3, max_value=5, description="3–5%"),
                BenchmarkBand("Weak", min_value=1, max_value=3, description="1–3%"),
                BenchmarkBand("Poor", max_value=1, description="<1%"),
            ),
        ),
        "dividend_growth": MetricBenchmark(
            key="dividend_growth",
            display_name="Dividend Growth",
            unit="%",
            bands=(
                BenchmarkBand("Excellent", min_value=10, description=">10% CAGR"),
                BenchmarkBand("Very Good", min_value=6, max_value=10, description="6–10%"),
                BenchmarkBand("Good", min_value=3, max_value=6, description="3–6%"),
                BenchmarkBand("Average", min_value=0, max_value=3, description="0–3%"),
                BenchmarkBand("Weak", max_value=0, description="Declining"),
            ),
        ),
        "book_value_growth": MetricBenchmark(
            key="book_value_growth",
            display_name="Book Value Growth",
            unit="%",
            bands=(
                BenchmarkBand("Excellent", min_value=12, description=">12% CAGR"),
                BenchmarkBand("Very Good", min_value=8, max_value=12, description="8–12%"),
                BenchmarkBand("Good", min_value=5, max_value=8, description="5–8%"),
                BenchmarkBand("Average", min_value=3, max_value=5, description="3–5%"),
                BenchmarkBand("Weak", min_value=1, max_value=3, description="1–3%"),
                BenchmarkBand("Poor", max_value=1, description="<1%"),
            ),
        ),
        "revenue": MetricBenchmark(
            key="revenue",
            display_name="Revenue",
            unit="₹ Cr",
            bands=(
                BenchmarkBand("Large Scale", min_value=10000, description=">10K Cr"),
                BenchmarkBand("Medium Scale", min_value=1000, max_value=10000, description="1K–10K Cr"),
                BenchmarkBand("Small Scale", max_value=1000, description="<1K Cr"),
            ),
        ),
        "net_profit": MetricBenchmark(
            key="net_profit",
            display_name="Net Profit",
            unit="₹ Cr",
            bands=(
                BenchmarkBand("Large Profit", min_value=1000, description=">1K Cr"),
                BenchmarkBand("Medium Profit", min_value=100, max_value=1000, description="100–1K Cr"),
                BenchmarkBand("Small Profit", max_value=100, description="<100 Cr"),
            ),
        ),
        "eps": MetricBenchmark(
            key="eps",
            display_name="Earnings Per Share",
            unit="₹",
            bands=(
                BenchmarkBand("High EPS", min_value=50, description=">50"),
                BenchmarkBand("Moderate EPS", min_value=10, max_value=50, description="10–50"),
                BenchmarkBand("Low EPS", max_value=10, description="<10"),
            ),
        ),
        "operating_income": MetricBenchmark(
            key="operating_income",
            display_name="Operating Income",
            unit="₹ Cr",
            bands=(
                BenchmarkBand("Large Operating Income", min_value=2000, description=">2K Cr"),
                BenchmarkBand("Medium Operating Income", min_value=200, max_value=2000, description="200–2K Cr"),
                BenchmarkBand("Small Operating Income", max_value=200, description="<200 Cr"),
            ),
        ),
        "ebit": MetricBenchmark(
            key="ebit",
            display_name="EBIT",
            unit="₹ Cr",
            bands=(
                BenchmarkBand("Large EBIT", min_value=2000, description=">2K Cr"),
                BenchmarkBand("Medium EBIT", min_value=200, max_value=2000, description="200–2K Cr"),
                BenchmarkBand("Small EBIT", max_value=200, description="<200 Cr"),
            ),
        ),
        "debt": MetricBenchmark(
            key="debt",
            display_name="Total Debt",
            unit="₹ Cr",
            lower_is_better=True,
            bands=(
                BenchmarkBand("Low Debt", max_value=500, description="<500 Cr"),
                BenchmarkBand("Moderate Debt", max_value=5000, description="500–5K Cr"),
                BenchmarkBand("High Debt", min_value=5000, description=">5K Cr"),
            ),
        ),
        "equity": MetricBenchmark(
            key="equity",
            display_name="Shareholder Equity",
            unit="₹ Cr",
            bands=(
                BenchmarkBand("Large Equity Base", min_value=5000, description=">5K Cr"),
                BenchmarkBand("Medium Equity Base", min_value=500, max_value=5000, description="500–5K Cr"),
                BenchmarkBand("Small Equity Base", max_value=500, description="<500 Cr"),
            ),
        ),
        "current_assets": MetricBenchmark(
            key="current_assets",
            display_name="Current Assets",
            unit="₹ Cr",
            bands=(
                BenchmarkBand("Large", min_value=1000, description=">1K Cr"),
                BenchmarkBand("Medium", min_value=100, max_value=1000, description="100–1K Cr"),
                BenchmarkBand("Small", max_value=100, description="<100 Cr"),
            ),
        ),
        "current_liabilities": MetricBenchmark(
            key="current_liabilities",
            display_name="Current Liabilities",
            unit="₹ Cr",
            lower_is_better=True,
            bands=(
                BenchmarkBand("Low", max_value=100, description="<100 Cr"),
                BenchmarkBand("Moderate", max_value=1000, description="100–1K Cr"),
                BenchmarkBand("High", min_value=1000, description=">1K Cr"),
            ),
        ),
        "operating_cash_flow": MetricBenchmark(
            key="operating_cash_flow",
            display_name="Operating Cash Flow",
            unit="₹ Cr",
            bands=(
                BenchmarkBand("Strong OCF", min_value=1000, description=">1K Cr"),
                BenchmarkBand("Healthy OCF", min_value=100, max_value=1000, description="100–1K Cr"),
                BenchmarkBand("Weak OCF", max_value=100, description="<100 Cr"),
            ),
        ),
        "capex": MetricBenchmark(
            key="capex",
            display_name="Capital Expenditure",
            unit="₹ Cr",
            bands=(
                BenchmarkBand("High Reinvestment", min_value=1000, description=">1K Cr"),
                BenchmarkBand("Moderate Reinvestment", min_value=100, max_value=1000, description="100–1K Cr"),
                BenchmarkBand("Low Reinvestment", max_value=100, description="<100 Cr"),
            ),
        ),
        "free_cash_flow": MetricBenchmark(
            key="free_cash_flow",
            display_name="Free Cash Flow",
            unit="₹ Cr",
            bands=(
                BenchmarkBand("Strong FCF", min_value=500, description=">500 Cr"),
                BenchmarkBand("Moderate FCF", min_value=0, max_value=500, description="0–500 Cr"),
                BenchmarkBand("Negative FCF", max_value=0, description="<0 Cr"),
            ),
        ),
        "interest_expense": MetricBenchmark(
            key="interest_expense",
            display_name="Interest Expense",
            unit="₹ Cr",
            lower_is_better=True,
            bands=(
                BenchmarkBand("Low Interest Cost", max_value=50, description="<50 Cr"),
                BenchmarkBand("Moderate Interest Cost", max_value=500, description="50–500 Cr"),
                BenchmarkBand("High Interest Cost", min_value=500, description=">500 Cr"),
            ),
        ),
        "book_value": MetricBenchmark(
            key="book_value",
            display_name="Book Value Per Share",
            unit="₹",
            bands=(
                BenchmarkBand("High BV", min_value=500, description=">500"),
                BenchmarkBand("Moderate BV", min_value=100, max_value=500, description="100–500"),
                BenchmarkBand("Low BV", max_value=100, description="<100"),
            ),
        ),
        "dividend": MetricBenchmark(
            key="dividend",
            display_name="Dividend Per Share",
            unit="₹",
            bands=(
                BenchmarkBand("High Dividend", min_value=15, description=">15"),
                BenchmarkBand("Moderate Dividend", min_value=2, max_value=15, description="2–15"),
                BenchmarkBand("Low Dividend", max_value=2, description="<2"),
            ),
        ),
        "capital_employed": MetricBenchmark(
            key="capital_employed",
            display_name="Capital Employed",
            unit="₹ Cr",
            bands=(
                BenchmarkBand("Large Scale", min_value=10000, description=">10K Cr"),
                BenchmarkBand("Medium Scale", min_value=1000, max_value=10000, description="1K–10K Cr"),
                BenchmarkBand("Small Scale", max_value=1000, description="<1K Cr"),
            ),
        ),
        "market_cap": MetricBenchmark(
            key="market_cap",
            display_name="Market Cap",
            unit="₹ Cr",
            bands=(
                BenchmarkBand("Large Cap", min_value=20000, description=">20K Cr"),
                BenchmarkBand("Mid Cap", min_value=5000, max_value=20000, description="5K–20K Cr"),
                BenchmarkBand("Small Cap", max_value=5000, description="<5K Cr"),
            ),
        ),
        "current_price": MetricBenchmark(
            key="current_price",
            display_name="Current Price",
            unit="₹",
            bands=(
                BenchmarkBand("High Price", min_value=1000, description=">1000"),
                BenchmarkBand("Mid Price", min_value=100, max_value=1000, description="100–1000"),
                BenchmarkBand("Low Price", max_value=100, description="<100"),
            ),
        ),
    }


# Sector-specific overrides (future-ready; empty bands inherit defaults)
SECTOR_BENCHMARK_OVERRIDES: Dict[str, Dict[str, MetricBenchmark]] = {
    "Technology": {},
    "Financial Services": {
        "debt_to_equity": MetricBenchmark(
            key="debt_to_equity",
            display_name="Debt to Equity",
            unit="x",
            lower_is_better=True,
            bands=(
                BenchmarkBand("Excellent", max_value=1.0, description="<1.0 (banks)"),
                BenchmarkBand("Good", max_value=2.0, description="<2.0"),
                BenchmarkBand("Average", max_value=4.0, description="<4.0"),
                BenchmarkBand("Weak", min_value=4.0, description=">4.0"),
            ),
        ),
    },
    "Consumer Defensive": {},
    "Energy": {},
    "Healthcare": {},
    "Industrials": {},
    "Utilities": {},
    "Communication Services": {},
    "Basic Materials": {},
    "Real Estate": {},
}

DEFAULT_BENCHMARKS: Dict[str, MetricBenchmark] = _default_benchmarks()

SUPPORTED_SECTORS: List[str] = list(SECTOR_BENCHMARK_OVERRIDES.keys())
