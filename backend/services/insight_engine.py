"""
services/insight_engine.py — AI Insight Generation Service
==========================================================
Generates structured research narratives (financial health, growth drivers,
risk factors, category summaries) from scored metrics and financial data.

Single responsibility: narrative synthesis. Delegates category scoring logic
to the same patterns used by the dashboard, keeping insight generation
decoupled from API response assembly.
"""

from typing import List, Optional

from models.fundamental import DashboardSummary, MetricScore
from services.dashboard_engine import DashboardEngine
from services.data_cleaner import CleanedFinancialData
from services.ratio_calculator import CalculatedRatios


class InsightEngine:
    """
    Produces AI-style insight sections for the research report.

    Usage:
        engine = InsightEngine()
        dashboard = engine.generate(
            metric_scores, cleaned_data, ratios, total_score, grade, strengths, weaknesses
        )
    """

    def __init__(self, dashboard_engine: Optional[DashboardEngine] = None) -> None:
        self._dashboard = dashboard_engine or DashboardEngine()

    def generate(
        self,
        metric_scores: List[MetricScore],
        cleaned_data: CleanedFinancialData,
        ratios: CalculatedRatios,
        total_score: float,
        grade: str,
        strengths: List[str],
        weaknesses: List[str],
    ) -> DashboardSummary:
        """Builds all insight sections from actual metric values — no placeholders."""
        return self._dashboard.build(
            metric_scores=metric_scores,
            cleaned_data=cleaned_data,
            ratios=ratios,
            total_score=total_score,
            grade=grade,
            strengths=strengths,
            weaknesses=weaknesses,
        )
