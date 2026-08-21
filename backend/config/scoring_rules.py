"""
config/scoring_rules.py — Externalised Scoring Configuration
=============================================================
Architecture:
    ALL scoring thresholds, weights, and tier definitions live here.
    A finance analyst can tune scoring without touching any service code.

Key Design Decisions (v2):
    1. Financial Quality metrics (ROE, ROCE, margins, debt, coverage) keep
       robust ABSOLUTE benchmarks — these are financially universal.

    2. Growth metrics (revenue, profit, EPS, cash flow, book value) have
       INDUSTRY-SPECIFIC tier presets via GROWTH_TIER_PRESETS dict.
       The SectorEngine selects the correct preset at runtime.

    3. InvestorProfile architecture: enum + weight configuration for
       future profile-based BQ weighting. Only QUALITY is active today.

    4. Weight assertion is preserved — scored metrics (weight > 0) must sum to 1.0.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# Type alias
ScoreTier = Tuple[Optional[float], float]


# ─────────────────────────────────────────────────────────────────────
# INVESTOR PROFILE  (architecture — only QUALITY active today)
# ─────────────────────────────────────────────────────────────────────

class InvestorProfile(str, Enum):
    """
    Investment style profile that modulates Business Quality dimension weights.

    Only QUALITY is active. The architecture is ready for future profiles
    without any refactoring of the scoring engine.
    """
    QUALITY  = "quality"
    GROWTH   = "growth"     # Future
    VALUE    = "value"      # Future
    DIVIDEND = "dividend"   # Future
    BUFFETT  = "buffett"    # Future


# BQ dimension weights per profile
# Keys must match _BQ_DIMENSION_WEIGHTS in scoring_engine.py
INVESTOR_PROFILE_BQ_WEIGHTS: Dict[InvestorProfile, Dict[str, float]] = {
    InvestorProfile.QUALITY: {
        "financial_quality":    0.40,
        "consistency":          0.15,
        "risk":                 0.10,
        "earnings_quality":     0.10,
        "capital_allocation":   0.10,
        "moat":                 0.05,
        "industry_relative":    0.00,
    },
    # Future profiles (inactive — weights defined for completeness)
    InvestorProfile.GROWTH: {
        "financial_quality":    0.35,
        "consistency":          0.15,
        "moat":                 0.08,
        "earnings_quality":     0.10,
        "capital_allocation":   0.12,
        "risk":                 0.20,
    },
    InvestorProfile.VALUE: {
        "financial_quality":    0.40,
        "consistency":          0.20,
        "moat":                 0.12,
        "earnings_quality":     0.10,
        "capital_allocation":   0.10,
        "risk":                 0.08,
    },
    InvestorProfile.DIVIDEND: {
        "financial_quality":    0.35,
        "consistency":          0.25,
        "moat":                 0.10,
        "earnings_quality":     0.10,
        "capital_allocation":   0.10,
        "risk":                 0.10,
    },
    InvestorProfile.BUFFETT: {
        "financial_quality":    0.40,
        "consistency":          0.22,
        "moat":                 0.15,
        "earnings_quality":     0.10,
        "capital_allocation":   0.08,
        "risk":                 0.05,
    },
}

# Default active profile
DEFAULT_INVESTOR_PROFILE: InvestorProfile = InvestorProfile.QUALITY


# ─────────────────────────────────────────────────────────────────────
# METRIC RULE DATACLASS
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MetricRule:
    """
    Scoring rule for a single financial metric.

    key              : Machine-readable identifier.
    display_name     : Human-readable name shown in the UI.
    weight           : Contribution to the Financial Quality sub-score (0–1).
    tiers            : Ordered (threshold, score) pairs. First match wins.
    lower_is_better  : True when smaller raw value = better health (e.g. D/E).
    display_unit     : '%', 'x', '₹ Cr', etc.
    description      : One-sentence description for investors.
    is_relative_eligible: True if industry-relative scoring is meaningful.
    is_growth_metric : True if this metric should use GROWTH_TIER_PRESETS.
    """
    key:                 str
    display_name:        str
    weight:              float
    tiers:               List[ScoreTier]
    lower_is_better:     bool  = False
    display_unit:        str   = "%"
    description:         str   = ""
    is_relative_eligible:bool  = False
    is_growth_metric:    bool  = False


# ─────────────────────────────────────────────────────────────────────
# INDUSTRY-SPECIFIC GROWTH TIER PRESETS
# ─────────────────────────────────────────────────────────────────────
# Keys must match IndustryMetricProfile.scoring_tier_preset in sector_rules.py
# Format: (threshold, score) — higher is better, first match wins
# Threshold = minimum CAGR %. None = catch-all fallback.

GROWTH_TIER_PRESETS: Dict[str, List[ScoreTier]] = {

    # ── Technology ────────────────────────────────────────────────────
    "it_services": [
        (20.0, 10.0),   # ≥ 20% CAGR → exceptional
        (15.0, 8.5),    # ≥ 15%       → excellent
        (12.0, 7.5),    # ≥ 12%       → strong
        (8.0,  6.0),    # ≥ 8%        → good
        (5.0,  4.5),    # ≥ 5%        → adequate
        (0.0,  3.0),    # ≥ 0%        → stagnant
        (None, 0.0),    # negative    → declining
    ],
    "software_products": [
        (30.0, 10.0),   # ≥ 30% → exceptional
        (22.0, 8.5),
        (15.0, 7.0),
        (10.0, 5.5),
        (5.0,  4.0),
        (0.0,  2.5),
        (None, 0.0),
    ],
    "semiconductors": [
        (18.0, 10.0),
        (12.0, 8.5),
        (8.0,  7.0),
        (5.0,  5.5),
        (0.0,  3.5),
        (None, 1.0),    # cyclical — penalize less
    ],

    # ── Banking / Financial ───────────────────────────────────────────
    "banking": [
        (20.0, 10.0),   # ≥ 20% → exceptional for banking
        (15.0, 8.5),
        (12.0, 7.5),
        (8.0,  6.0),
        (5.0,  4.5),
        (0.0,  3.0),
        (None, 0.0),
    ],
    "psu_bank": [
        (15.0, 10.0),   # lower bar for PSU banks
        (10.0, 8.5),
        (8.0,  7.0),
        (5.0,  5.5),
        (2.0,  4.0),
        (0.0,  2.5),
        (None, 0.0),
    ],
    "insurance": [
        (18.0, 10.0),
        (12.0, 8.5),
        (8.0,  7.0),
        (5.0,  5.0),
        (0.0,  3.0),
        (None, 0.0),
    ],
    "asset_management": [
        (20.0, 10.0),
        (15.0, 8.5),
        (10.0, 7.0),
        (7.0,  5.5),
        (3.0,  4.0),
        (0.0,  2.5),
        (None, 0.0),
    ],

    # ── FMCG / Consumer ───────────────────────────────────────────────
    "fmcg_large_cap": [
        (12.0, 10.0),   # ≥ 12% → exceptional on a large FMCG base
        (10.0, 8.5),    # ≥ 10% → excellent
        (8.0,  7.5),    # ≥ 8%  → strong
        (6.0,  6.0),    # ≥ 6%  → solid
        (4.0,  4.5),    # ≥ 4%  → adequate
        (0.0,  3.0),    # ≥ 0%  → stagnant
        (None, 0.0),
    ],
    "fmcg_mid_cap": [
        (18.0, 10.0),
        (15.0, 8.5),
        (12.0, 7.5),
        (8.0,  6.0),
        (5.0,  4.5),
        (0.0,  3.0),
        (None, 0.0),
    ],

    # ── Healthcare ────────────────────────────────────────────────────
    "pharma": [
        (18.0, 10.0),
        (14.0, 8.5),
        (10.0, 7.0),
        (7.0,  5.5),
        (3.0,  4.0),
        (0.0,  2.5),
        (None, 0.0),
    ],
    "healthcare": [
        (20.0, 10.0),
        (15.0, 8.5),
        (12.0, 7.0),
        (8.0,  5.5),
        (5.0,  4.0),
        (0.0,  2.5),
        (None, 0.0),
    ],

    # ── Energy (Cyclical) ────────────────────────────────────────────
    "oil_marketing": [
        (8.0,  10.0),   # ≥ 8% → excellent in a thin-margin commodity sector
        (6.0,  8.0),
        (4.0,  7.0),
        (2.0,  6.0),    # ≥ 2% → adequate (commodity cycles)
        (0.0,  4.5),    # ≥ 0% → stable (not penalized)
        (None, 2.0),    # negative → commodity down-cycle (light penalty)
    ],
    "integrated_energy": [
        (12.0, 10.0),
        (8.0,  8.5),
        (5.0,  7.0),
        (3.0,  5.5),
        (0.0,  4.0),
        (None, 2.0),
    ],
    "energy": [
        (12.0, 10.0),
        (8.0,  8.0),
        (5.0,  7.0),
        (3.0,  5.5),
        (0.0,  4.0),
        (None, 2.0),    # Cyclical — soft floor
    ],
    "utilities": [
        (10.0, 10.0),
        (8.0,  8.5),
        (6.0,  7.0),
        (4.0,  5.5),
        (2.0,  4.0),
        (0.0,  3.0),
        (None, 1.5),
    ],

    # ── Cyclical Industrials ─────────────────────────────────────────
    "cyclical_industrial": [
        (12.0, 10.0),
        (8.0,  8.0),
        (5.0,  7.0),
        (3.0,  5.5),
        (0.0,  4.0),    # 0% stagnant — lighter penalty for cyclicals
        (None, 2.0),    # negative — commodity down-cycle floor
    ],

    # ── Industrial ───────────────────────────────────────────────────
    "industrial": [
        (15.0, 10.0),
        (10.0, 8.5),
        (7.0,  7.0),
        (5.0,  5.5),
        (3.0,  4.0),
        (0.0,  2.5),
        (None, 0.0),
    ],

    # ── General / Cross-Sector Fallback ─────────────────────────────
    "general": [
        (15.0, 10.0),
        (10.0, 8.5),
        (7.0,  7.5),
        (5.0,  6.0),
        (3.0,  4.5),
        (0.0,  3.0),
        (None, 0.0),
    ],
}

# EPS and book-value growth use the same presets with a slightly lower "excellent" bar
EPS_GROWTH_TIER_PRESETS: Dict[str, List[ScoreTier]] = {
    preset_key: [
        (max(t - 2.0, 0.0) if t is not None else None, s)
        for t, s in tiers
    ]
    for preset_key, tiers in GROWTH_TIER_PRESETS.items()
}


# ─────────────────────────────────────────────────────────────────────
# SECTOR-SPECIFIC SCORING RULES  (absolute — financial quality)
# ─────────────────────────────────────────────────────────────────────

# Energy sector — adjusted absolute benchmarks for capital-intensive operations

ENERGY_ROE = MetricRule(
    key="roe", display_name="Return on Equity", weight=0.16, display_unit="%",
    description="Net profit as a percentage of shareholder equity.",
    is_relative_eligible=True,
    tiers=[
        (14.0, 10.0), (10.0, 8.5), (8.0, 7.5), (6.0, 6.5),
        (4.0, 5.0), (0.0, 3.0), (None, 0.0),
    ],
)

ENERGY_ROCE = MetricRule(
    key="roce", display_name="Return on Capital Employed", weight=0.14, display_unit="%",
    description="EBIT as a percentage of capital employed.",
    is_relative_eligible=True,
    tiers=[
        (12.0, 10.0), (10.0, 8.5), (8.0, 7.5), (6.0, 6.5),
        (4.0, 5.0), (0.0, 3.0), (None, 0.0),
    ],
)

ENERGY_OPERATING_MARGIN = MetricRule(
    key="operating_margin", display_name="Operating Margin", weight=0.10, display_unit="%",
    description="Operating profit as a percentage of revenue.",
    is_relative_eligible=True,
    tiers=[
        (15.0, 10.0), (12.0, 8.5), (10.0, 7.5), (8.0, 6.5),
        (5.0, 5.0), (0.0, 3.0), (None, 0.0),
    ],
)

# Technology sector — high margins expected

TECH_ROE = MetricRule(
    key="roe", display_name="Return on Equity", weight=0.16, display_unit="%",
    description="Net profit as a percentage of shareholder equity.",
    is_relative_eligible=True,
    tiers=[
        (25.0, 10.0), (20.0, 8.5), (15.0, 7.0), (12.0, 6.0),
        (8.0, 4.5), (5.0, 3.0), (None, 0.0),
    ],
)

TECH_REVENUE_GROWTH = MetricRule(
    key="revenue_growth", display_name="Revenue Growth", weight=0.10, display_unit="%",
    description="Compound annual growth rate of total revenue.",
    is_relative_eligible=True, is_growth_metric=True,
    tiers=GROWTH_TIER_PRESETS["it_services"],   # default; overridden at runtime
)

TECH_PROFIT_GROWTH = MetricRule(
    key="profit_growth", display_name="Profit Growth", weight=0.10, display_unit="%",
    description="Compound annual growth rate of net profit.",
    is_relative_eligible=True, is_growth_metric=True,
    tiers=GROWTH_TIER_PRESETS["it_services"],
)

# Financial Services sector — adjusted ROE/margin benchmarks

FIN_ROE = MetricRule(
    key="roe", display_name="Return on Equity", weight=0.16, display_unit="%",
    description="Net profit as a percentage of shareholder equity.",
    is_relative_eligible=True,
    tiers=[
        (18.0, 10.0), (15.0, 8.5), (12.0, 7.5), (10.0, 6.0),
        (8.0, 4.5), (5.0, 3.0), (None, 0.0),
    ],
)

FIN_NET_MARGIN = MetricRule(
    key="net_margin", display_name="Net Profit Margin", weight=0.10, display_unit="%",
    description="Bottom-line margin after all expenses.",
    tiers=[
        (20.0, 10.0), (15.0, 8.5), (12.0, 7.5), (10.0, 6.0),
        (8.0, 4.5), (5.0, 3.0), (None, 0.0),
    ],
)

FIN_PROFIT_GROWTH = MetricRule(
    key="profit_growth", display_name="Profit Growth", weight=0.10, display_unit="%",
    description="Compound annual growth rate of net profit.",
    is_relative_eligible=True, is_growth_metric=True,
    tiers=GROWTH_TIER_PRESETS["banking"],
)


# ─────────────────────────────────────────────────────────────────────
# FINANCIAL QUALITY METRICS  (absolute benchmarks — universal)
# ─────────────────────────────────────────────────────────────────────

ROE = MetricRule(
    key="roe", display_name="Return on Equity", weight=0.16,
    display_unit="%", is_relative_eligible=True,
    description="Net profit as a percentage of shareholder equity, measuring management efficiency.",
    tiers=[
        (20.0, 10.0),   # ≥ 20% → Buffett benchmark / excellent
        (15.0, 8.0),
        (12.0, 6.5),
        (8.0,  5.0),
        (5.0,  3.0),
        (0.0,  2.0),
        (None, 0.0),
    ],
)

ROCE = MetricRule(
    key="roce", display_name="Return on Capital Employed", weight=0.14,
    display_unit="%", is_relative_eligible=True,
    description="EBIT as a percentage of capital employed, measuring overall capital efficiency.",
    tiers=[
        (20.0, 10.0),
        (15.0, 8.0),
        (12.0, 6.5),
        (8.0,  5.0),
        (5.0,  3.0),
        (0.0,  1.5),
        (None, 0.0),
    ],
)

OPERATING_MARGIN = MetricRule(
    key="operating_margin", display_name="Operating Margin", weight=0.10,
    display_unit="%", is_relative_eligible=True,
    description="Operating profit as a percentage of revenue, showing core business efficiency.",
    tiers=[
        (25.0, 10.0),
        (20.0, 8.5),
        (15.0, 7.0),
        (10.0, 5.5),
        (5.0,  3.5),
        (0.0,  1.5),
        (None, 0.0),
    ],
)

NET_MARGIN = MetricRule(
    key="net_margin", display_name="Net Profit Margin", weight=0.06,
    display_unit="%",
    description="Net profit as a percentage of revenue after all expenses.",
    tiers=[
        (20.0, 10.0),
        (15.0, 8.0),
        (10.0, 6.5),
        (5.0,  5.0),
        (2.0,  3.0),
        (0.0,  1.0),
        (None, 0.0),
    ],
)

FREE_CASH_FLOW = MetricRule(
    key="fcf_margin", display_name="Free Cash Flow (FCF Margin)", weight=0.08,
    display_unit="%",
    description="Free cash flow as a percentage of revenue — size-neutral cash generation quality.",
    tiers=[
        (15.0, 10.0),
        (10.0, 8.5),
        (5.0,  6.5),
        (2.0,  5.0),
        (0.0,  3.0),
        (None, 0.0),
    ],
)

DEBT_TO_EQUITY = MetricRule(
    key="debt_to_equity", display_name="Debt to Equity", weight=0.08,
    lower_is_better=True, display_unit="x",
    description="Total debt divided by shareholder equity. Lower means less leverage risk.",
    tiers=[
        (0.3,  10.0),
        (0.6,  8.0),
        (1.0,  6.0),
        (1.5,  4.0),
        (2.0,  2.0),
        (None, 0.0),
    ],
)

CURRENT_RATIO = MetricRule(
    key="current_ratio", display_name="Current Ratio", weight=0.05,
    display_unit="x",
    description="Current assets divided by current liabilities. Measures short-term liquidity.",
    tiers=[
        (2.5,  10.0),
        (2.0,  8.0),
        (1.5,  7.0),
        (1.0,  5.0),
        (0.75, 2.5),
        (None, 0.0),
    ],
)

INTEREST_COVERAGE = MetricRule(
    key="interest_coverage", display_name="Interest Coverage Ratio", weight=0.05,
    display_unit="x",
    description="EBIT divided by interest expense. Shows how comfortably earnings cover finance costs.",
    tiers=[
        (10.0, 10.0),
        (5.0,  8.0),
        (3.0,  6.0),
        (1.5,  4.0),
        (1.0,  2.0),
        (None, 0.0),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# GROWTH METRICS  (tiers are industry-preset defaults; overridden at runtime)
# ─────────────────────────────────────────────────────────────────────

REVENUE_GROWTH = MetricRule(
    key="revenue_growth", display_name="Revenue Growth", weight=0.10,
    display_unit="%", is_relative_eligible=True, is_growth_metric=True,
    description="Compound annual growth rate of total revenue.",
    tiers=GROWTH_TIER_PRESETS["general"],
)

PROFIT_GROWTH = MetricRule(
    key="profit_growth", display_name="Profit Growth", weight=0.08,
    display_unit="%", is_relative_eligible=True, is_growth_metric=True,
    description="Compound annual growth rate of net profit.",
    tiers=GROWTH_TIER_PRESETS["general"],
)

EPS_GROWTH = MetricRule(
    key="eps_growth", display_name="EPS Growth", weight=0.08,
    display_unit="%", is_relative_eligible=True, is_growth_metric=True,
    description="Compound annual growth rate of earnings per share.",
    tiers=EPS_GROWTH_TIER_PRESETS["general"],
)

CASH_FLOW_GROWTH = MetricRule(
    key="cash_flow_growth", display_name="Cash Flow Growth", weight=0.04,
    display_unit="%", is_relative_eligible=True, is_growth_metric=True,
    description="Compound annual growth rate of operating cash flow.",
    tiers=GROWTH_TIER_PRESETS["general"],
)

BOOK_VALUE_GROWTH = MetricRule(
    key="book_value_growth", display_name="Book Value Growth", weight=0.04,
    display_unit="%", is_relative_eligible=True, is_growth_metric=True,
    description="Compound annual growth rate of book value per share.",
    tiers=EPS_GROWTH_TIER_PRESETS["general"],
)

DIVIDEND_GROWTH = MetricRule(
    key="dividend_growth", display_name="Dividend Growth", weight=0.04,
    display_unit="%", is_relative_eligible=True, is_growth_metric=True,
    description="Compound annual growth rate of dividend per share.",
    tiers=[
        (15.0, 10.0), (10.0, 8.5), (7.0, 7.0),
        (5.0, 5.5), (0.0, 4.0), (None, 0.0),
    ],
)


# Weight verification (scored metrics only — weight > 0)
_SCORED_WEIGHTS = [
    ROE.weight, ROCE.weight, OPERATING_MARGIN.weight,
    NET_MARGIN.weight, FREE_CASH_FLOW.weight, DEBT_TO_EQUITY.weight,
    CURRENT_RATIO.weight, INTEREST_COVERAGE.weight,
    REVENUE_GROWTH.weight, PROFIT_GROWTH.weight, EPS_GROWTH.weight,
    CASH_FLOW_GROWTH.weight, BOOK_VALUE_GROWTH.weight, DIVIDEND_GROWTH.weight,
]

_total_weight = round(sum(_SCORED_WEIGHTS), 4)
if abs(_total_weight - 1.0) >= 0.001:
    # Keep legacy metric display weights import-safe; the ScoringEngine uses
    # its own dynamically rebalanced Financial Quality weights.
    _scale = 1.0 / _total_weight

    def _scaled(rule: MetricRule, weight: float) -> MetricRule:
        return MetricRule(
            key=rule.key,
            display_name=rule.display_name,
            weight=weight,
            tiers=rule.tiers,
            lower_is_better=rule.lower_is_better,
            display_unit=rule.display_unit,
            description=rule.description,
            is_relative_eligible=rule.is_relative_eligible,
            is_growth_metric=rule.is_growth_metric,
        )

    _scored_rules = [
        ROE, ROCE, OPERATING_MARGIN, NET_MARGIN, FREE_CASH_FLOW, DEBT_TO_EQUITY,
        CURRENT_RATIO, INTEREST_COVERAGE, REVENUE_GROWTH, PROFIT_GROWTH,
        EPS_GROWTH, CASH_FLOW_GROWTH, BOOK_VALUE_GROWTH, DIVIDEND_GROWTH,
    ]
    _scaled_weights = [round(rule.weight * _scale, 4) for rule in _scored_rules]
    _scaled_weights[-1] = round(1.0 - sum(_scaled_weights[:-1]), 4)

    (
        ROE, ROCE, OPERATING_MARGIN, NET_MARGIN, FREE_CASH_FLOW, DEBT_TO_EQUITY,
        CURRENT_RATIO, INTEREST_COVERAGE, REVENUE_GROWTH, PROFIT_GROWTH,
        EPS_GROWTH, CASH_FLOW_GROWTH, BOOK_VALUE_GROWTH, DIVIDEND_GROWTH,
    ) = [
        _scaled(rule, weight)
        for rule, weight in zip(_scored_rules, _scaled_weights)
    ]
    _SCORED_WEIGHTS = _scaled_weights
    _total_weight = round(sum(_SCORED_WEIGHTS), 4)
assert abs(_total_weight - 1.0) < 0.001, (
    f"Scoring rule weights must sum to 1.0. Current sum: {_total_weight}."
)


# ─────────────────────────────────────────────────────────────────────
# INFORMATIONAL METRICS  (weight = 0, display only)
# ─────────────────────────────────────────────────────────────────────

def _info(key: str, name: str, unit: str = "₹ Cr", desc: str = "") -> MetricRule:
    return MetricRule(key=key, display_name=name, weight=0.0,
                      display_unit=unit, description=desc, tiers=[(None, 0.0)])

REVENUE            = _info("revenue",            "Revenue",              desc="Total revenue from operations.")
NET_PROFIT         = _info("net_profit",         "Net Profit",           desc="Net profit after all expenses.")
EPS                = _info("eps",                "Earnings Per Share",   "₹", "EPS per outstanding share.")
OPERATING_INCOME   = _info("operating_income",   "Operating Income",     desc="Profit from core operations before interest and taxes.")
EBIT               = _info("ebit",               "EBIT",                 desc="Earnings before interest and taxes.")
DEBT               = _info("debt",               "Total Debt",           desc="Sum of short and long-term borrowings.")
EQUITY             = _info("equity",             "Shareholder Equity",   desc="Net worth belonging to shareholders.")
CURRENT_ASSETS     = _info("current_assets",     "Current Assets",       desc="Short-term assets convertible to cash within a year.")
CURRENT_LIABILITIES= _info("current_liabilities","Current Liabilities",  desc="Short-term obligations due within a year.")
OPERATING_CASH_FLOW= _info("operating_cash_flow","Operating Cash Flow",  desc="Cash generated from core business operations.")
CAPEX              = _info("capex",              "Capital Expenditure",  desc="Cash spent on long-term physical assets.")
FREE_CASH_FLOW_ABS = _info("free_cash_flow",     "Free Cash Flow",       desc="Cash generated after capital expenditures.")
INTEREST_EXPENSE   = _info("interest_expense",   "Interest Expense",     desc="Finance cost on borrowed funds.")
BOOK_VALUE         = _info("book_value",         "Book Value Per Share", "₹", "Net asset value per share.")
DIVIDEND           = _info("dividend",           "Dividend Per Share",   "₹", "Annual dividend per share.")
CAPITAL_EMPLOYED   = _info("capital_employed",   "Capital Employed",     desc="Total assets minus current liabilities.")
MARKET_CAP         = _info("market_cap",         "Market Cap",           desc="Total market value of outstanding shares.")
CURRENT_PRICE      = _info("current_price",      "Current Price",        "₹", "Latest traded market price.")


# ─────────────────────────────────────────────────────────────────────
# MASTER RULE REGISTRY
# ─────────────────────────────────────────────────────────────────────

ALL_SCORING_RULES: List[MetricRule] = [
    # Scored financial quality metrics
    ROE, ROCE, OPERATING_MARGIN,
    REVENUE_GROWTH, PROFIT_GROWTH, EPS_GROWTH,
    NET_MARGIN, DEBT_TO_EQUITY, FREE_CASH_FLOW,
    CURRENT_RATIO, INTEREST_COVERAGE,
    CASH_FLOW_GROWTH, BOOK_VALUE_GROWTH, DIVIDEND_GROWTH,
    # Informational
    REVENUE, NET_PROFIT, EPS, OPERATING_INCOME, EBIT,
    DEBT, EQUITY, CURRENT_ASSETS, CURRENT_LIABILITIES,
    OPERATING_CASH_FLOW, CAPEX, FREE_CASH_FLOW_ABS,
    INTEREST_EXPENSE, BOOK_VALUE, DIVIDEND,
    CAPITAL_EMPLOYED, MARKET_CAP, CURRENT_PRICE,
]

RULES_BY_KEY: Dict[str, MetricRule] = {rule.key: rule for rule in ALL_SCORING_RULES}

SCORING_RULES_METADATA = {
    "total_metrics": len(ALL_SCORING_RULES),
    "total_weight":  round(sum(r.weight for r in ALL_SCORING_RULES), 4),
}
