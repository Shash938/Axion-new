"""
services/sector_engine.py — Sector Detection and Scoring Adjustments
====================================================================
Two-level classification:
  1. SectorType   (coarse) — controls metric exclusions
  2. IndustrySubType (fine) — selects the correct growth-tier preset

Industry-Relative Scoring Architecture:
  - When peer data is available, inject into PeerAnalysisEngine
  - When peer data is absent, industry_relative contributes 0% (not estimated)
  - This module is the only place that understands industry context;
    the scoring engine stays industry-agnostic.
"""

import logging
from dataclasses import replace
from typing import Dict, List, Optional, Set, Tuple

from config.scoring_rules import (
    ALL_SCORING_RULES,
    EPS_GROWTH_TIER_PRESETS,
    ENERGY_OPERATING_MARGIN,
    ENERGY_ROCE,
    ENERGY_ROE,
    FIN_NET_MARGIN,
    FIN_PROFIT_GROWTH,
    FIN_ROE,
    GROWTH_TIER_PRESETS,
    MetricRule,
    TECH_PROFIT_GROWTH,
    TECH_REVENUE_GROWTH,
    TECH_ROE,
)
from config.sector_rules import (
    CYCLICAL_INDUSTRIES,
    INDUSTRY_METRIC_PROFILES,
    SECTOR_WEIGHT_BOOST,
    IndustryMetricProfile,
    IndustrySubType,
    SectorProfile,
    SectorType,
    classify_sector,
)
from services.data_cleaner import CleanedFinancialData

logger = logging.getLogger(__name__)


class SectorEngine:
    """
    Provides sector/industry classification and scoring rule adjustments.

    Exposes:
    - profile()             → SectorProfile (includes IndustrySubType + metric profile)
    - get_effective_rules() → (rules, profile, excluded_set)
    """

    def profile(self, data: CleanedFinancialData) -> SectorProfile:
        return classify_sector(data.sector, data.industry, data.ticker)

    def get_effective_rules(
        self,
        data: CleanedFinancialData,
    ) -> Tuple[List[MetricRule], SectorProfile, Set[str]]:
        """
        Returns scoring rules with:
        1. Sector-specific rule overrides (ENERGY_ROE, FIN_ROE, etc.)
        2. Industry-specific growth tier presets applied to all growth metrics
        3. Excluded metrics zeroed-out (kept for display)
        4. Weight redistribution for excluded metrics

        Excluded metrics keep weight=0 (informational display only).
        Their weight is redistributed to sector-relevant metrics.
        """
        profile = self.profile(data)
        sub_type = profile.industry_sub_type
        metric_profile = profile.metric_profile

        excluded = set(metric_profile.excluded_metrics)
        boosts = SECTOR_WEIGHT_BOOST.get(profile.sector_type, {})
        industry_overrides = profile.industry_profile.weight_overrides

        # ── Step 1: Start with base rules ──────────────────────────
        base_rules = list(ALL_SCORING_RULES)

        # ── Step 2: Apply sector-type absolute overrides ────────────
        base_rules = self._apply_sector_overrides(base_rules, profile.sector_type)

        # ── Step 3: Apply industry-specific growth tier presets ─────
        growth_preset = metric_profile.scoring_tier_preset
        base_rules = self._apply_growth_tier_preset(base_rules, growth_preset, sub_type)

        # ── Step 4: Handle exclusions ───────────────────────────────
        if not excluded:
            logger.info(
                "SectorEngine: %s → %s / %s — no exclusions.",
                data.ticker, profile.sector_type.value, sub_type.value,
            )
            return base_rules, profile, excluded

        # Calculate removed weight
        removed_weight = sum(r.weight for r in base_rules if r.key in excluded)
        boost_total = sum(boosts.values())

        adjusted: List[MetricRule] = []
        for rule in base_rules:
            if rule.key in excluded:
                # Zero weight — kept for display only
                adjusted.append(MetricRule(
                    key=rule.key, display_name=rule.display_name, weight=0.0,
                    tiers=rule.tiers, lower_is_better=rule.lower_is_better,
                    display_unit=rule.display_unit,
                    description=rule.description + f" (Excluded for {profile.display_name})",
                    is_relative_eligible=False,
                    is_growth_metric=rule.is_growth_metric,
                ))
            else:
                # Boost weight for remaining metrics
                if boost_total > 0 and rule.key in boosts:
                    scale = removed_weight / boost_total
                    new_weight = round(rule.weight + boosts[rule.key] * scale, 4)
                else:
                    new_weight = rule.weight

                if rule.key in industry_overrides:
                    new_weight = round(industry_overrides[rule.key], 4)

                adjusted.append(MetricRule(
                    key=rule.key, display_name=rule.display_name, weight=new_weight,
                    tiers=rule.tiers, lower_is_better=rule.lower_is_better,
                    display_unit=rule.display_unit, description=rule.description,
                    is_relative_eligible=rule.is_relative_eligible,
                    is_growth_metric=rule.is_growth_metric,
                ))

        # ── Step 5: Normalize scored metric weights to sum=1.0 ──────
        scored = [r for r in adjusted if r.weight > 0 and r.key not in excluded]
        total = sum(r.weight for r in scored)
        if total > 0 and abs(total - 1.0) > 0.001:
            normalized: List[MetricRule] = []
            for rule in adjusted:
                if rule.weight > 0 and rule.key not in excluded:
                    normalized.append(MetricRule(
                        key=rule.key, display_name=rule.display_name,
                        weight=round(rule.weight / total, 4),
                        tiers=rule.tiers, lower_is_better=rule.lower_is_better,
                        display_unit=rule.display_unit, description=rule.description,
                        is_relative_eligible=rule.is_relative_eligible,
                        is_growth_metric=rule.is_growth_metric,
                    ))
                else:
                    normalized.append(rule)
            adjusted = normalized

        logger.info(
            "SectorEngine: %s → %s / %s — excluded %d metrics, preset=%s, cyclical=%s.",
            data.ticker, profile.sector_type.value, sub_type.value,
            len(excluded), metric_profile.scoring_tier_preset, metric_profile.cyclical,
        )
        return adjusted, profile, excluded

    # ──────────────────────────────────────────────────────────────────
    # SECTOR-TYPE ABSOLUTE OVERRIDES
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_sector_overrides(
        base_rules: List[MetricRule],
        sector_type: SectorType,
    ) -> List[MetricRule]:
        """Replaces specific rules with sector-calibrated absolute benchmarks."""
        if sector_type == SectorType.ENERGY:
            overrides = {
                "roe": ENERGY_ROE,
                "roce": ENERGY_ROCE,
                "operating_margin": ENERGY_OPERATING_MARGIN,
            }
        elif sector_type == SectorType.TECHNOLOGY:
            overrides = {
                "roe": TECH_ROE,
                "revenue_growth": TECH_REVENUE_GROWTH,
                "profit_growth": TECH_PROFIT_GROWTH,
            }
        elif sector_type in (SectorType.BANK, SectorType.NBFC, SectorType.INSURANCE):
            overrides = {
                "roe": FIN_ROE,
                "net_margin": FIN_NET_MARGIN,
                "profit_growth": FIN_PROFIT_GROWTH,
            }
        else:
            return base_rules

        return [
            overrides.get(rule.key, rule) if rule.key in overrides else rule
            for rule in base_rules
        ]

    # ──────────────────────────────────────────────────────────────────
    # INDUSTRY GROWTH TIER PRESET APPLICATION
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_growth_tier_preset(
        rules: List[MetricRule],
        preset_key: str,
        sub_type: IndustrySubType,
    ) -> List[MetricRule]:
        """
        Replaces growth metric tier lists with industry-specific presets.
        Only affects metrics where is_growth_metric=True.
        Preserves all other rule attributes unchanged.
        """
        # Determine which preset tables to use
        standard_preset = GROWTH_TIER_PRESETS.get(preset_key, GROWTH_TIER_PRESETS["general"])
        eps_preset = EPS_GROWTH_TIER_PRESETS.get(preset_key, EPS_GROWTH_TIER_PRESETS["general"])

        # Per-metric preset selection
        per_metric_presets = {
            "revenue_growth":    standard_preset,
            "profit_growth":     standard_preset,
            "cash_flow_growth":  standard_preset,
            "book_value_growth": eps_preset,     # slightly lower bar
            "eps_growth":        eps_preset,
            "dividend_growth":   None,            # uses universal tiers
        }

        result: List[MetricRule] = []
        for rule in rules:
            if rule.is_growth_metric and rule.key in per_metric_presets:
                new_tiers = per_metric_presets[rule.key]
                if new_tiers is not None:
                    result.append(MetricRule(
                        key=rule.key, display_name=rule.display_name,
                        weight=rule.weight, tiers=new_tiers,
                        lower_is_better=rule.lower_is_better,
                        display_unit=rule.display_unit,
                        description=rule.description,
                        is_relative_eligible=rule.is_relative_eligible,
                        is_growth_metric=True,
                    ))
                    continue
            result.append(rule)
        return result
