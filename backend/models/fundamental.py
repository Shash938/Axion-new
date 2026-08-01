"""
models/fundamental.py — Pydantic Request / Response Models
============================================================
Why this file exists:
    Defines the strict, validated data contracts between the API layer and
    callers (the Chrome extension, cURL, tests). Using Pydantic v2 ensures
    that malformed inputs are rejected at the boundary before any business
    logic runs.

How it connects:
    - `AnalysisRequest` is received by routers/analysis.py as the request body.
    - `AnalysisResponse` is assembled by services/fundamental_analyzer.py and
      returned by the router.
    - `MetricScore` is constructed by services/explanation_engine.py.
    - `FundamentalScore` is constructed by services/scoring_engine.py.

Possible improvements:
    - Add `TechnicalScore` and `SentimentScore` models for future phases.
    - Add a `PortfolioRequest` model that accepts a list of tickers.
    - Add localization fields when multi-language support is introduced.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ==============================================================================
# Enumerations
# ==============================================================================


class Exchange(str, Enum):
    """
    Supported Indian stock exchanges.
    Values map to yfinance suffix conventions.
    """
    NSE = "NSE"
    BSE = "BSE"


class Recommendation(str, Enum):
    """
    Final investment recommendation produced by the scoring engine.
    Three-state model is deliberately conservative for V1.
    """
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class Grade(str, Enum):
    """
    Letter grade derived from the overall fundamental score (0–10 scale).
        S+: >= 9.0    — outstanding
        S:  >= 8.2    — excellent
        A+: >= 7.6    — very strong
        A:  >= 7.2    — strong
        B+: >= 6.4    — good
        B:  >= 5.5    — average/good
        C:  >= 4.0    — below average
        D:  >= 2.5    — weak
        F:  <2.5      — poor/high risk
    """
    S_PLUS = "S+"
    S = "S"
    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


# ==============================================================================
# Request Models
# ==============================================================================


class AnalysisRequest(BaseModel):
    """
    Payload sent by the client to request a fundamental analysis.

    Example:
        {
            "ticker": "RELIANCE",
            "exchange": "NSE"
        }
    """
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="NSE/BSE stock ticker symbol (e.g. RELIANCE, TCS, INFY).",
        examples=["RELIANCE", "TCS", "INFY"],
    )
    exchange: Exchange = Field(
        default=Exchange.NSE,
        description="Stock exchange the ticker is listed on.",
    )

    @field_validator("ticker")
    @classmethod
    def normalise_ticker(cls, v: str) -> str:
        """
        Strips whitespace, uppercases the ticker, and removes any exchange
        suffixes the user may have accidentally included (e.g. 'TCS.NS' → 'TCS').
        The DataFetcherService will re-append the correct suffix.
        """
        cleaned = v.strip().upper()
        # Remove common exchange suffixes clients may accidentally include
        for suffix in (".NS", ".BO", ".BSE", ".NSE"):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
        if not cleaned:
            raise ValueError("Ticker cannot be empty after normalisation.")
        return cleaned


# ==============================================================================
# Component Models — Building Blocks of the Response
# ==============================================================================


class MetricHistoryPoint(BaseModel):
    """Single year in a metric's historical series."""
    year: str
    value: float


class PeerMetrics(BaseModel):
    """Industry peer comparison metrics for relative scoring."""
    industry_rank: Optional[int] = Field(default=None, description="Rank within the peer group (1 is best).")
    total_peers: Optional[int] = Field(default=None, description="Total number of peers evaluated.")
    percentile: Optional[float] = Field(default=None, description="Percentile rank (e.g., 90th percentile).")
    quartile: Optional[int] = Field(default=None, description="Quartile (1 to 4, where 1 is top 25%).")
    peer_average: Optional[float] = Field(default=None, description="Average metric value among peers.")
    peer_median: Optional[float] = Field(default=None, description="Median metric value among peers.")
    difference_from_average: Optional[float] = Field(default=None, description="Target value minus peer average.")



class MetricScore(BaseModel):
    """
    The score for a single financial metric with full historical context.

    Each MetricScore contains:
    - The raw calculated value (e.g., ROE = 22.5%)
    - Historical series, YoY, 3Y/5Y CAGR, trend, and benchmark
    - A normalised score on a 0–10 scale
    - AI commentary and score justification
    """
    metric_name: str = Field(..., description="Human-readable metric name (e.g. 'Return on Equity').")
    metric_key: str = Field(..., description="Machine-readable key (e.g. 'roe').")
    raw_value: Optional[float] = Field(
        default=None,
        description="The calculated ratio/percentage value. None if data was unavailable.",
    )
    raw_value_unit: str = Field(
        default="",
        description="Unit of the raw value: '%', 'x' (ratio), '₹ Cr', etc.",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Final assigned score. Defaults to absolute_score if peer data is missing, otherwise hybrid_score.",
    )
    absolute_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Score derived purely from fixed financial benchmarks (0–10).",
    )
    relative_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Score derived from peer percentile ranking (0–10). None if peer data is unavailable.",
    )
    hybrid_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Blended score combining absolute and relative components.",
    )
    weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weight of this metric in the overall fundamental score (0–1).",
    )
    informational: bool = Field(
        default=False,
        description="True for display-only metrics that do not affect the overall score.",
    )
    weighted_score: float = Field(
        ...,
        ge=0.0,
        description="Contribution to overall score: score × weight.",
    )
    explanation: str = Field(
        ...,
        description=(
            "Plain-English explanation of what this metric means and why this "
            "stock received this specific score. Written for retail / beginner investors."
        ),
    )
    data_available: bool = Field(
        default=True,
        description="False if this metric could not be calculated due to missing data.",
    )
    # Historical context (institutional research format)
    history: List[MetricHistoryPoint] = Field(
        default_factory=list,
        description="Up to 5 years of historical values, oldest to newest.",
    )
    yoy: Optional[float] = Field(default=None, description="Latest year-over-year change (%).")
    cagr3: Optional[float] = Field(default=None, description="3-year compound annual growth rate (%).")
    cagr5: Optional[float] = Field(default=None, description="5-year compound annual growth rate (%).")
    trend: Optional[str] = Field(default=None, description="Trend classification (Growing, Declining, etc.).")
    benchmark_label: Optional[str] = Field(default=None, description="Benchmark band label (Excellent, Good, etc.).")
    benchmark_summary: Optional[str] = Field(default=None, description="Full benchmark tier descriptions.")
    score_reason: Optional[str] = Field(default=None, description="Why this specific score was assigned.")
    ai_commentary: Optional[str] = Field(default=None, description="Extended AI-style commentary with numbers.")
    peer_metrics: Optional[PeerMetrics] = Field(default=None, description="Peer comparison data, if available.")


class CategoryAssessment(BaseModel):
    """Score, grade, and narrative for a financial category."""
    score: Optional[float] = Field(default=None, description="Average category score (0–10).")
    grade: str = Field(default="N/A", description="Letter grade for this category.")
    explanation: str = Field(default="", description="Short explanation of the category assessment.")


class DashboardSummary(BaseModel):
    """Professional equity research dashboard summary."""
    financial_health: str = Field(default="", description="Overall financial health narrative.")
    financial_summary: str = Field(default="", description="One-paragraph financial overview.")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    growth_drivers: List[str] = Field(default_factory=list)
    profitability: CategoryAssessment = Field(default_factory=CategoryAssessment)
    liquidity: CategoryAssessment = Field(default_factory=CategoryAssessment)
    leverage: CategoryAssessment = Field(default_factory=CategoryAssessment)
    cash_generation: CategoryAssessment = Field(default_factory=CategoryAssessment)
    growth: CategoryAssessment = Field(default_factory=CategoryAssessment)
    financial_quality: CategoryAssessment = Field(default_factory=CategoryAssessment)


class FundamentalScore(BaseModel):
    """
    Aggregated fundamental score calculated from all metric scores.

    Business Quality dimensions are exposed separately so the UI can
    display each analytical layer independently rather than collapsing
    everything into a single opaque number.
    """
    total_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Weighted aggregate fundamental score (0–10). Equals business_quality_score.",
    )
    business_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Composite Business Quality Score across all active dimensions (0–10).",
    )
    valuation_score: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="Independent valuation attractiveness score (0–10). Higher = cheaper.",
    )
    risk_score: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="Financial risk score based on leverage and liquidity (0–10). Higher = lower risk.",
    )
    # ── Named sub-scores (all on 0–10 scale) ───────────────────────────
    financial_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Score from core financial metrics: ROE, ROCE, margins, growth, FCF (0–10).",
    )
    consistency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Historical stability and trend quality of fundamentals over 3–5 years (0–10).",
    )
    moat_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Inferred competitive advantage score from multi-year ROE, margins and FCF (0–10).",
    )
    earnings_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Quality of reported earnings — OCF vs net income, FCF consistency, accrual quality (0–10).",
    )
    capital_allocation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Management capital allocation effectiveness — ROCE trend, FCF generation, reinvestment (0–10).",
    )
    industry_relative_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Peer-relative performance score (0–10). None until real peer benchmark data is available.",
    )
    # ── Legacy / hybrid fields ─────────────────────────────────────────
    absolute_total_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Score derived from purely absolute benchmarks.",
    )
    relative_total_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Score derived purely from peer comparison. None until peer data is available.",
    )
    hybrid_total_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Blended score — equals business_quality_score when peer data is unavailable.",
    )
    grade: Grade = Field(..., description="Letter grade derived from business_quality_score.")
    recommendation: Recommendation = Field(
        ...,
        description="Investment recommendation: BUY, HOLD, or SELL.",
    )
    metrics_evaluated: int = Field(
        ...,
        description="Number of scored metrics for which data was available.",
    )
    metrics_total: int = Field(
        ...,
        description="Total number of scored metrics in the model.",
    )
    coverage_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of scored metrics with available data. Low coverage = less reliable.",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall data confidence (0–100) based on completeness of financial statements.",
    )
    data_quality_notes: List[str] = Field(
        default_factory=list,
        description=(
            "Per-statement data quality notes explaining which line items were available or missing "
            "and why certain metrics could not be calculated."
        ),
    )


class CompanyInfo(BaseModel):
    """
    Basic company metadata fetched alongside the financial data.
    """
    ticker: str
    exchange: Exchange
    company_name: str = Field(default="Unknown")
    sector: str = Field(default="Unknown")
    industry: str = Field(default="Unknown")
    industry_sub_type: str = Field(
        default="General",
        description="Resolved industry sub-classification (e.g. 'IT Services', 'Private Sector Bank').",
    )
    size_category: str = Field(
        default="Unknown",
        description="Market-cap size bucket: 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap'.",
    )
    is_cyclical: bool = Field(
        default=False,
        description="True if the company operates in a cyclical industry (energy, metals, cement, etc.).",
    )
    market_cap: Optional[float] = Field(default=None, description="Market cap in INR Crores.")
    current_price: Optional[float] = Field(default=None, description="Last traded price in INR.")
    currency: str = Field(default="INR")


# ==============================================================================
# Top-Level Response Model
# ==============================================================================


class AnalysisResponse(BaseModel):
    """
    Complete fundamental analysis response returned to the client.

    Design notes:
    - Metric scores are a list (not a dict) so ordering is preserved for the UI.
    - `overall_explanation` is the paragraph the extension displays prominently.
    - `warnings` is used to surface data-quality notes (e.g. partial data).
    - `analysed_at` enables cache-aware clients to know the data freshness.
    """
    company: CompanyInfo
    fundamental_score: FundamentalScore
    metric_scores: List[MetricScore] = Field(
        ...,
        description="Ordered list of individual metric scores, highest-weight first.",
    )
    overall_explanation: str = Field(
        ...,
        description=(
            "AI-generated paragraph summarising the company's fundamental health "
            "in plain English. Designed for beginner investors."
        ),
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="List of metrics where the company scored ≥ 7/10.",
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="List of metrics where the company scored ≤ 3/10.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Data quality warnings (e.g. 'Dividend data unavailable for this ticker').",
    )
    dashboard: Optional[DashboardSummary] = Field(
        default=None,
        description="Structured research dashboard with category assessments.",
    )
    analysed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when this analysis was generated.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "company": {
                    "ticker": "RELIANCE",
                    "exchange": "NSE",
                    "company_name": "Reliance Industries Limited",
                    "sector": "Energy",
                    "industry": "Oil & Gas Refining & Marketing",
                    "market_cap": 1950000.0,
                    "current_price": 2880.5,
                    "currency": "INR",
                },
                "fundamental_score": {
                    "total_score": 7.2,
                    "grade": "B",
                    "recommendation": "BUY",
                    "metrics_evaluated": 12,
                    "metrics_total": 14,
                    "coverage_pct": 85.7,
                },
                "overall_explanation": (
                    "Reliance Industries shows strong fundamentals with a score of 7.2/10. "
                    "The company demonstrates excellent revenue growth and healthy margins, "
                    "though its debt-to-equity ratio warrants monitoring."
                ),
                "strengths": ["Revenue Growth", "Operating Margin", "Interest Coverage"],
                "weaknesses": ["Debt to Equity"],
                "warnings": ["Dividend growth data unavailable."],
            }
        }
    }


# ==============================================================================
# Error Models
# ==============================================================================


class ErrorDetail(BaseModel):
    """RFC 7807 Problem Details — used for all error responses."""
    type: str = Field(default="about:blank", description="URI identifying the problem type.")
    title: str = Field(..., description="Short, human-readable summary of the problem.")
    status: int = Field(..., description="HTTP status code.")
    detail: str = Field(..., description="Detailed explanation of this specific occurrence.")
    ticker: Optional[str] = Field(default=None, description="Ticker that caused the error, if applicable.")
