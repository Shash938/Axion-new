"""
services/response_builder.py — API Response Assembly
====================================================
Enriches MetricScore objects with historical context and builds
the structured dashboard summary for the research report UI.
"""

import logging
from typing import List, Optional

from models.fundamental import DashboardSummary, MetricHistoryPoint, MetricScore
from services.dashboard_engine import DashboardEngine
from services.data_cleaner import CleanedFinancialData
from services.historical_engine import HistoricalEngine, MetricDetail
from services.ratio_calculator import CalculatedRatios

logger = logging.getLogger(__name__)


class ResponseBuilder:
    """Merges scoring output with historical engine data into enriched API responses."""

    def __init__(
        self,
        historical_engine: Optional[HistoricalEngine] = None,
        dashboard_engine: Optional[DashboardEngine] = None,
    ) -> None:
        self._historical = historical_engine or HistoricalEngine()
        self._dashboard = dashboard_engine or DashboardEngine()

    def enrich_metrics(
        self,
        metric_scores: List[MetricScore],
        cleaned_data: CleanedFinancialData,
        ratios: CalculatedRatios,
        metric_details: Optional[dict] = None,
    ) -> List[MetricScore]:
        """Attaches history, trend, benchmark, and score_reason to each MetricScore."""
        details = metric_details if metric_details is not None else self._historical.build_all(cleaned_data)
        enriched: List[MetricScore] = []

        for ms in metric_scores:
            detail = details.get(ms.metric_key)
            if detail is None and ms.metric_key == "fcf_margin":
                detail = details.get("fcf_margin")
            if detail is None:
                enriched.append(ms)
                continue

            score_reason = self._build_score_reason(ms, detail)
            updated = {
                "history": [
                    MetricHistoryPoint(year=h.year, value=h.value)
                    for h in detail.history
                ],
                "yoy": detail.yoy,
                "cagr3": detail.cagr3,
                "cagr5": detail.cagr5,
                "trend": detail.trend,
                "benchmark_label": detail.benchmark_label,
                "benchmark_summary": detail.benchmark_summary,
                "score_reason": score_reason,
                "ai_commentary": ms.explanation,
            }
            enriched.append(ms.model_copy(update=updated))

        return enriched

    def build_dashboard(
        self,
        metric_scores: List[MetricScore],
        cleaned_data: CleanedFinancialData,
        ratios: CalculatedRatios,
        total_score: float,
        grade: str,
        strengths: List[str],
        weaknesses: List[str],
    ) -> DashboardSummary:
        """Builds structured research dashboard sections from calculated metrics."""
        return self._dashboard.build(
            metric_scores=metric_scores,
            cleaned_data=cleaned_data,
            ratios=ratios,
            total_score=total_score,
            grade=grade,
            strengths=strengths,
            weaknesses=weaknesses,
        )

    @staticmethod
    def _build_score_reason(ms: MetricScore, detail: MetricDetail) -> str:
        if ms.informational and ms.data_available:
            return "Informational metric — not included in overall score."
        if not ms.data_available:
            return "Score 0 — data unavailable for calculation."
        parts = [f"Scored {ms.score:.0f}/10"]
        if detail.benchmark_label:
            parts.append(f"benchmark: {detail.benchmark_label}")
        if detail.trend and detail.trend != "Unavailable":
            parts.append(f"trend: {detail.trend}")
        if detail.cagr3 is not None:
            parts.append(f"3Y CAGR: {detail.cagr3:.1f}%")
        return " · ".join(parts)
