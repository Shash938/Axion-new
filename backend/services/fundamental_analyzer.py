"""
services/fundamental_analyzer.py — Pipeline Orchestrator
=========================================================
Coordinates the complete fundamental analysis pipeline:

    Data Fetcher → Data Cleaner → Validation Engine → Ratio Calculator
    → Historical Engine → Scoring Engine → Explanation Engine
    → Response Builder → Insight Engine → API Response
"""

import logging
from datetime import datetime
from typing import Dict, Optional

from models.fundamental import AnalysisRequest, AnalysisResponse, CompanyInfo
from services.data_cleaner import DataCleanerService
from services.data_fetcher import DataFetcherService
from services.explanation_engine import ExplanationEngine
from services.historical_engine import HistoricalEngine, MetricDetail
from services.insight_engine import InsightEngine
from services.ratio_calculator import RatioCalculatorService
from services.response_builder import ResponseBuilder
from services.scoring_engine import ScoringEngine
from services.validation_engine import ValidationEngine

logger = logging.getLogger(__name__)


class FundamentalAnalyzerService:
    """Orchestrates the complete fundamental analysis pipeline."""

    def __init__(
        self,
        data_fetcher: Optional[DataFetcherService] = None,
        data_cleaner: Optional[DataCleanerService] = None,
        validation_engine: Optional[ValidationEngine] = None,
        ratio_calculator: Optional[RatioCalculatorService] = None,
        scoring_engine: Optional[ScoringEngine] = None,
        explanation_engine: Optional[ExplanationEngine] = None,
        historical_engine: Optional[HistoricalEngine] = None,
        response_builder: Optional[ResponseBuilder] = None,
        insight_engine: Optional[InsightEngine] = None,
    ) -> None:
        self._fetcher = data_fetcher or DataFetcherService()
        self._cleaner = data_cleaner or DataCleanerService()
        self._validator = validation_engine or ValidationEngine()
        self._calculator = ratio_calculator or RatioCalculatorService()
        self._scorer = scoring_engine or ScoringEngine()
        self._explainer = explanation_engine or ExplanationEngine()
        self._historical = historical_engine or HistoricalEngine()
        self._builder = response_builder or ResponseBuilder(self._historical)
        self._insights = insight_engine or InsightEngine()
        logger.info("FundamentalAnalyzerService initialised.")

    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Executes the full fundamental analysis pipeline for the given request.

        Pipeline stages:
            1. Fetch     — DataFetcherService
            2. Clean     — DataCleanerService
            3. Validate  — ValidationEngine
            4. Calculate — RatioCalculatorService
            5. History   — HistoricalEngine (metric details with trends)
            6. Score     — ScoringEngine
            7. Explain   — ExplanationEngine (with historical context)
            8. Enrich    — ResponseBuilder (attach history to API models)
            9. Insights  — InsightEngine (dashboard narratives)
           10. Assemble  — Build AnalysisResponse
        """
        ticker = request.ticker
        exchange = request.exchange.value
        started_at = datetime.utcnow()
        logger.info("Starting fundamental analysis for %s (%s).", ticker, exchange)

        # Stage 1: Fetch
        logger.debug("Stage 1/10: Fetching raw data...")
        raw_data = self._fetcher.fetch(ticker=ticker, exchange=exchange)

        # Stage 2: Clean
        logger.debug("Stage 2/10: Cleaning and validating data...")
        cleaned_data = self._cleaner.clean(raw_data)

        # Stage 3: Validate inputs before ratio calculation
        logger.debug("Stage 3/10: Running validation engine...")
        self._validator.validate(cleaned_data)

        # Stage 4: Calculate Ratios
        logger.debug("Stage 4/10: Calculating financial ratios...")
        ratios = self._calculator.calculate(cleaned_data)

        # Stage 5: Build historical metric details (before explanations)
        logger.debug("Stage 5/10: Building historical context...")
        metric_details: Dict[str, MetricDetail] = self._historical.build_all(cleaned_data)

        # Stage 6: Score
        logger.debug("Stage 6/10: Scoring metrics...")
        metric_scores, fundamental_score = self._scorer.score(ratios, cleaned_data)

        # Stage 7: Explain (with historical + benchmark context)
        logger.debug("Stage 7/10: Generating explanations...")
        annotated_scores, overall_explanation, strengths, weaknesses = self._explainer.explain(
            metric_scores=metric_scores,
            ratios=ratios,
            cleaned_data=cleaned_data,
            fundamental_score=fundamental_score,
            metric_details=metric_details,
        )

        # Stage 8: Enrich API models with history fields
        logger.debug("Stage 8/10: Enriching metric response models...")
        enriched_scores = self._builder.enrich_metrics(
            annotated_scores, cleaned_data, ratios, metric_details=metric_details
        )

        # Stage 9: Generate insight dashboard
        logger.debug("Stage 9/10: Generating research insights...")
        dashboard = self._insights.generate(
            enriched_scores,
            cleaned_data,
            ratios,
            fundamental_score.total_score,
            fundamental_score.grade.value,
            strengths,
            weaknesses,
        )

        # Stage 10: Assemble response
        sector_profile = self._scorer._sector.profile(cleaned_data)
        company_info = CompanyInfo(
            ticker=ticker,
            exchange=request.exchange,
            company_name=cleaned_data.company_name,
            sector=cleaned_data.sector,
            industry=cleaned_data.industry,
            industry_sub_type=sector_profile.metric_profile.display_name,
            size_category=self._size_category(cleaned_data.market_cap),
            is_cyclical=bool(sector_profile.metric_profile.cyclical),
            market_cap=cleaned_data.market_cap,
            current_price=cleaned_data.current_price,
            currency=cleaned_data.currency,
        )

        response = AnalysisResponse(
            company=company_info,
            fundamental_score=fundamental_score,
            metric_scores=enriched_scores,
            overall_explanation=overall_explanation,
            strengths=strengths,
            weaknesses=weaknesses,
            warnings=list(cleaned_data.warnings),
            dashboard=dashboard,
            analysed_at=started_at,
        )

        elapsed_ms = (datetime.utcnow() - started_at).total_seconds() * 1000
        logger.info(
            "Analysis complete for %s: score=%.2f, grade=%s, recommendation=%s, elapsed=%.0fms",
            ticker,
            fundamental_score.total_score,
            fundamental_score.grade.value,
            fundamental_score.recommendation.value,
            elapsed_ms,
        )

        return response

    @staticmethod
    def _size_category(market_cap: Optional[float]) -> str:
        if market_cap is None:
            return "Unknown"
        if market_cap >= 20_000:
            return "Large Cap"
        if market_cap >= 5_000:
            return "Mid Cap"
        if market_cap >= 500:
            return "Small Cap"
        return "Micro Cap"
