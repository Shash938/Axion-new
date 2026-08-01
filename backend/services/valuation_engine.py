"""
services/valuation_engine.py - Valuation Score Engine
=====================================================
Evaluates valuation attractiveness independently from Business Quality.
The engine uses sector-aware metric mixes without fabricating peer multiples.
"""

from typing import Dict, Optional, Tuple

from config.sector_rules import SectorProfile, SectorType
from services.data_cleaner import CleanedFinancialData
from services.ratio_calculator import CalculatedRatios


class ValuationEngine:
    """Scores valuation multiples on a 0-10 scale. Higher means cheaper."""

    def evaluate(
        self,
        ratios: CalculatedRatios,
        sector_profile: Optional[SectorProfile] = None,
        cleaned_data: Optional[CleanedFinancialData] = None,
    ) -> float:
        score_accumulator = 0.0
        weight_accumulator = 0.0

        metric_weights = self._metric_weights(sector_profile)
        for metric_key, weight in metric_weights.items():
            raw_value = getattr(ratios, metric_key, None)
            if raw_value is None or raw_value <= 0:
                continue
            score_accumulator += self._score_metric(metric_key, raw_value, sector_profile, cleaned_data) * weight
            weight_accumulator += weight

        if weight_accumulator == 0:
            return 5.0
        return round(score_accumulator / weight_accumulator, 2)

    @staticmethod
    def _metric_weights(sector_profile: Optional[SectorProfile]) -> Dict[str, float]:
        sector_type = sector_profile.sector_type if sector_profile else SectorType.UNKNOWN
        if sector_type in (SectorType.BANK, SectorType.NBFC, SectorType.INSURANCE):
            return {"pb_ratio": 0.50, "pe_ratio": 0.30, "peg_ratio": 0.20}
        if sector_type == SectorType.TECHNOLOGY:
            return {"pe_ratio": 0.35, "peg_ratio": 0.35, "price_to_sales": 0.15, "ev_ebitda": 0.15}
        if sector_type == SectorType.ENERGY:
            return {"ev_ebitda": 0.45, "pe_ratio": 0.25, "pb_ratio": 0.20, "dividend_yield": 0.10}
        if sector_type == SectorType.HEALTHCARE:
            return {"peg_ratio": 0.35, "pe_ratio": 0.30, "ev_ebitda": 0.20, "price_to_sales": 0.15}
        if sector_type == SectorType.FMCG:
            return {"pe_ratio": 0.40, "peg_ratio": 0.30, "ev_ebitda": 0.20, "dividend_yield": 0.10}
        return {"pe_ratio": 0.30, "pb_ratio": 0.20, "ev_ebitda": 0.25, "peg_ratio": 0.25}

    def _score_metric(
        self,
        metric_key: str,
        value: float,
        sector_profile: Optional[SectorProfile],
        cleaned_data: Optional[CleanedFinancialData],
    ) -> float:
        sector_type = sector_profile.sector_type if sector_profile else SectorType.UNKNOWN
        if metric_key == "pe_ratio":
            return self._score_lower_better(value, self._pe_tiers(sector_type, cleaned_data))
        if metric_key == "pb_ratio":
            return self._score_lower_better(value, self._pb_tiers(sector_type))
        if metric_key == "ev_ebitda":
            return self._score_lower_better(value, self._ev_ebitda_tiers(sector_type))
        if metric_key == "peg_ratio":
            return self._score_lower_better(value, ((0.7, 10.0), (1.0, 8.0), (1.5, 6.0), (2.0, 4.0), (3.0, 2.0)))
        if metric_key == "price_to_sales":
            return self._score_lower_better(value, ((2.0, 10.0), (4.0, 8.0), (7.0, 6.0), (10.0, 4.0), (15.0, 2.0)))
        if metric_key == "dividend_yield":
            if value >= 5.0:
                return 10.0
            if value >= 3.0:
                return 8.0
            if value >= 1.5:
                return 6.0
            if value >= 0.5:
                return 4.0
            return 2.0
        return 5.0

    @staticmethod
    def _score_lower_better(value: float, tiers: Tuple[Tuple[float, float], ...]) -> float:
        for threshold, score in tiers:
            if value < threshold:
                return score
        return 0.0

    @staticmethod
    def _pe_tiers(
        sector_type: SectorType,
        cleaned_data: Optional[CleanedFinancialData],
    ) -> Tuple[Tuple[float, float], ...]:
        if sector_type == SectorType.TECHNOLOGY:
            return ((18.0, 10.0), (28.0, 8.0), (40.0, 6.0), (55.0, 4.0), (75.0, 2.0))
        if sector_type == SectorType.HEALTHCARE:
            return ((20.0, 10.0), (30.0, 8.0), (45.0, 6.0), (60.0, 4.0), (80.0, 2.0))
        if sector_type == SectorType.FMCG:
            return ((25.0, 10.0), (35.0, 8.0), (50.0, 6.0), (65.0, 4.0), (85.0, 2.0))
        if sector_type == SectorType.ENERGY:
            return ((8.0, 10.0), (12.0, 8.0), (18.0, 6.0), (25.0, 4.0), (35.0, 2.0))
        if sector_type in (SectorType.BANK, SectorType.NBFC, SectorType.INSURANCE):
            return ((10.0, 10.0), (15.0, 8.0), (22.0, 6.0), (30.0, 4.0), (45.0, 2.0))
        if cleaned_data and cleaned_data.market_cap and cleaned_data.market_cap >= 100_000:
            return ((12.0, 10.0), (18.0, 8.0), (25.0, 6.0), (35.0, 4.0), (55.0, 2.0))
        return ((10.0, 10.0), (15.0, 8.0), (20.0, 6.0), (30.0, 4.0), (50.0, 2.0))

    @staticmethod
    def _pb_tiers(sector_type: SectorType) -> Tuple[Tuple[float, float], ...]:
        if sector_type in (SectorType.BANK, SectorType.NBFC):
            return ((1.0, 10.0), (1.8, 8.0), (2.8, 6.0), (4.0, 4.0), (6.0, 2.0))
        if sector_type == SectorType.TECHNOLOGY:
            return ((3.0, 10.0), (6.0, 8.0), (10.0, 6.0), (15.0, 4.0), (25.0, 2.0))
        return ((1.0, 10.0), (2.0, 8.0), (3.5, 6.0), (5.0, 4.0), (8.0, 2.0))

    @staticmethod
    def _ev_ebitda_tiers(sector_type: SectorType) -> Tuple[Tuple[float, float], ...]:
        if sector_type == SectorType.ENERGY:
            return ((6.0, 10.0), (9.0, 8.0), (13.0, 6.0), (18.0, 4.0), (25.0, 2.0))
        if sector_type in (SectorType.TECHNOLOGY, SectorType.HEALTHCARE):
            return ((12.0, 10.0), (18.0, 8.0), (25.0, 6.0), (35.0, 4.0), (50.0, 2.0))
        return ((8.0, 10.0), (12.0, 8.0), (16.0, 6.0), (22.0, 4.0), (30.0, 2.0))
