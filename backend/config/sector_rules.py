"""
config/sector_rules.py — Industry Sub-Classification & Metric Profiles
=======================================================================
Architecture:
    Two-level classification:
    1. SectorType   — coarse (BANK / NBFC / INSURANCE / ENERGY / TECHNOLOGY / INDUSTRIAL)
                      Controls which metrics are EXCLUDED entirely (e.g. D/E for banks).
    2. IndustrySubType — granular (PRIVATE_BANK / IT_SERVICES / OIL_MARKETING / …)
                      Controls which SCORING TIER PRESET applies to growth metrics.

Industry Metric Visibility:
    Each IndustryMetricProfile defines:
    - excluded_metrics   : removed from both scoring AND display
    - informational_only : shown but do not affect score
    - scoring_tier_preset: key into GROWTH_TIER_PRESETS (in scoring_rules.py)
    - cyclical           : bool — enables multi-cycle normalization in consistency engine

Peer Data Architecture:
    All peer-relative fields are preserved in models.
    When peer_data is absent, industry_relative_score = None and contributes 0%
    to the Business Quality Score. Plug in a benchmark DB later without touching
    any scoring logic.

Cyclical Industries (approved list):
    Oil & Gas, Oil Marketing, Exploration, Refining, Steel, Metals, Mining,
    Cement, Shipping, Commodity Chemicals, Fertilizers, Paper, Power Utilities
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# COARSE SECTOR TYPE  (controls metric exclusions)
# ─────────────────────────────────────────────────────────────────────

class SectorType(str, Enum):
    INDUSTRIAL   = "industrial"
    BANK         = "bank"
    NBFC         = "nbfc"
    INSURANCE    = "insurance"
    ENERGY       = "energy"
    TECHNOLOGY   = "technology"
    HEALTHCARE   = "healthcare"
    FMCG         = "fmcg"
    UNKNOWN      = "unknown"


# ─────────────────────────────────────────────────────────────────────
# INDUSTRY SUB-TYPE  (controls growth tier presets + cyclical flag)
# ─────────────────────────────────────────────────────────────────────

class IndustrySubType(str, Enum):
    # ── Banking ──────────────────────────────────────────────────────
    PRIVATE_BANK        = "private_bank"
    PSU_BANK            = "psu_bank"
    SMALL_FINANCE_BANK  = "small_finance_bank"

    # ── Financial Services ───────────────────────────────────────────
    NBFC                = "nbfc"
    INSURANCE_LIFE      = "insurance_life"
    INSURANCE_GENERAL   = "insurance_general"
    ASSET_MANAGEMENT    = "asset_management"

    # ── Technology ───────────────────────────────────────────────────
    IT_SERVICES         = "it_services"
    SOFTWARE_PRODUCTS   = "software_products"
    SEMICONDUCTORS      = "semiconductors"
    ENGINEERING_SOFTWARE= "engineering_software"

    # ── Energy ───────────────────────────────────────────────────────
    OIL_MARKETING       = "oil_marketing"
    INTEGRATED_ENERGY   = "integrated_energy"
    EXPLORATION         = "exploration"
    REFINING            = "refining"
    POWER_UTILITIES     = "power_utilities"

    # ── Industrials / Cyclicals ───────────────────────────────────────
    CEMENT              = "cement"
    STEEL               = "steel"
    METALS              = "metals"
    MINING              = "mining"
    SHIPPING            = "shipping"
    COMMODITY_CHEMICALS = "commodity_chemicals"
    FERTILIZERS         = "fertilizers"
    PAPER               = "paper"
    ENGINEERING         = "engineering"

    # ── Consumer ─────────────────────────────────────────────────────
    FMCG_LARGE_CAP      = "fmcg_large_cap"
    FMCG_MID_CAP        = "fmcg_mid_cap"
    CONSUMER_DURABLES   = "consumer_durables"

    # ── Healthcare ────────────────────────────────────────────────────
    PHARMA              = "pharma"
    HOSPITALS           = "hospitals"
    DIAGNOSTICS         = "diagnostics"

    # ── Real Estate & Others ─────────────────────────────────────────
    REAL_ESTATE         = "real_estate"
    TELECOM             = "telecom"
    MEDIA               = "media"

    # ── Fallback ─────────────────────────────────────────────────────
    GENERAL             = "general"


# ─────────────────────────────────────────────────────────────────────
# CYCLICAL INDUSTRY SET
# ─────────────────────────────────────────────────────────────────────

CYCLICAL_INDUSTRIES: FrozenSet[IndustrySubType] = frozenset({
    IndustrySubType.OIL_MARKETING,
    IndustrySubType.INTEGRATED_ENERGY,
    IndustrySubType.EXPLORATION,
    IndustrySubType.REFINING,
    IndustrySubType.POWER_UTILITIES,
    IndustrySubType.STEEL,
    IndustrySubType.METALS,
    IndustrySubType.MINING,
    IndustrySubType.CEMENT,
    IndustrySubType.SHIPPING,
    IndustrySubType.COMMODITY_CHEMICALS,
    IndustrySubType.FERTILIZERS,
    IndustrySubType.PAPER,
})

NON_CYCLICAL_INDUSTRIES: FrozenSet[IndustrySubType] = frozenset({
    IndustrySubType.IT_SERVICES,
    IndustrySubType.SOFTWARE_PRODUCTS,
    IndustrySubType.FMCG_LARGE_CAP,
    IndustrySubType.FMCG_MID_CAP,
    IndustrySubType.PHARMA,
    IndustrySubType.HOSPITALS,
    IndustrySubType.DIAGNOSTICS,
    IndustrySubType.PRIVATE_BANK,
    IndustrySubType.PSU_BANK,
    IndustrySubType.SMALL_FINANCE_BANK,
})


# ─────────────────────────────────────────────────────────────────────
# INDUSTRY METRIC PROFILE
# Defines per-industry metric visibility + growth context
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IndustryMetricProfile:
    """
    Controls metric scoring behaviour for a specific industry sub-type.

    excluded_metrics   : Completely hidden — not scored, not displayed.
    informational_only : Displayed with value but not scored.
    scoring_tier_preset: Key into GROWTH_TIER_PRESETS (scoring_rules.py).
    cyclical           : True enables multi-period normalization in ConsistencyEngine.
    growth_context_text: Human-readable growth benchmark for ExplanationEngine.
    display_name       : Shown in explanations ("IT Services", "Private Bank", …)
    """
    key: str
    display_name: str
    excluded_metrics: FrozenSet[str] = field(default_factory=frozenset)
    informational_only: FrozenSet[str] = field(default_factory=frozenset)
    scoring_tier_preset: str = "general"
    cyclical: bool = False
    growth_context_text: str = ""


# ─────────────────────────────────────────────────────────────────────
# METRIC EXCLUSION SETS  (by sector type)
# ─────────────────────────────────────────────────────────────────────

_BANK_EXCLUDED = frozenset({
    "debt_to_equity",
    "current_ratio",
    "operating_margin",
    "interest_coverage",
    "roce",
})

_INSURANCE_EXCLUDED = frozenset({
    "debt_to_equity",
    "current_ratio",
    "operating_margin",
    "interest_coverage",
    "roce",
})

_NBFC_EXCLUDED = frozenset({
    "current_ratio",
    "operating_margin",
    "interest_coverage",
})

_EMPTY: FrozenSet[str] = frozenset()


# ─────────────────────────────────────────────────────────────────────
# INDUSTRY METRIC PROFILES — one per IndustrySubType
# ─────────────────────────────────────────────────────────────────────

INDUSTRY_METRIC_PROFILES: Dict[IndustrySubType, IndustryMetricProfile] = {

    # ── BANKING ──────────────────────────────────────────────────────

    IndustrySubType.PRIVATE_BANK: IndustryMetricProfile(
        key="private_bank",
        display_name="Private Sector Bank",
        excluded_metrics=_BANK_EXCLUDED,
        scoring_tier_preset="banking",
        growth_context_text=(
            "Private banks are expected to grow revenue 12–20% annually. "
            "ROE above 15% is considered excellent in this industry."
        ),
    ),
    IndustrySubType.PSU_BANK: IndustryMetricProfile(
        key="psu_bank",
        display_name="Public Sector Bank",
        excluded_metrics=_BANK_EXCLUDED,
        scoring_tier_preset="psu_bank",
        growth_context_text=(
            "PSU banks typically grow 8–15% annually with lower ROE expectations (10–15%). "
            "Capital adequacy and NPA management are critical indicators."
        ),
    ),
    IndustrySubType.SMALL_FINANCE_BANK: IndustryMetricProfile(
        key="small_finance_bank",
        display_name="Small Finance Bank",
        excluded_metrics=_BANK_EXCLUDED,
        scoring_tier_preset="banking",
        growth_context_text=(
            "Small finance banks target 20–30% growth given their smaller base. "
            "High ROE (>15%) and consistent margin expansion are the key indicators."
        ),
    ),

    # ── FINANCIAL SERVICES ───────────────────────────────────────────

    IndustrySubType.NBFC: IndustryMetricProfile(
        key="nbfc",
        display_name="NBFC / Financial Services",
        excluded_metrics=_NBFC_EXCLUDED,
        scoring_tier_preset="banking",
        growth_context_text=(
            "NBFCs are evaluated on profitability, growth, and adjusted leverage. "
            "Revenue growth of 15–25% is strong for consumer-focused NBFCs."
        ),
    ),
    IndustrySubType.INSURANCE_LIFE: IndustryMetricProfile(
        key="insurance_life",
        display_name="Life Insurance",
        excluded_metrics=_INSURANCE_EXCLUDED,
        scoring_tier_preset="insurance",
        growth_context_text=(
            "Life insurers are evaluated on new business premium growth and ROE. "
            "15%+ VNB growth is considered excellent."
        ),
    ),
    IndustrySubType.INSURANCE_GENERAL: IndustryMetricProfile(
        key="insurance_general",
        display_name="General Insurance",
        excluded_metrics=_INSURANCE_EXCLUDED,
        scoring_tier_preset="insurance",
        growth_context_text=(
            "General insurers are evaluated on combined ratio and premium growth. "
            "10–15% premium growth with combined ratio below 100% is excellent."
        ),
    ),
    IndustrySubType.ASSET_MANAGEMENT: IndustryMetricProfile(
        key="asset_management",
        display_name="Asset Management",
        excluded_metrics=frozenset({"current_ratio", "interest_coverage"}),
        scoring_tier_preset="asset_management",
        growth_context_text=(
            "Asset managers are evaluated on AUM growth and margins. "
            "15%+ revenue growth with 30%+ operating margins is excellent."
        ),
    ),

    # ── TECHNOLOGY ───────────────────────────────────────────────────

    IndustrySubType.IT_SERVICES: IndustryMetricProfile(
        key="it_services",
        display_name="IT Services",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="it_services",
        growth_context_text=(
            "Large-cap IT services companies typically grow 8–18% annually. "
            "Revenue growth above 12% CAGR is considered excellent. "
            "ROE above 25% and operating margins above 20% are hallmarks of top-tier IT firms."
        ),
    ),
    IndustrySubType.SOFTWARE_PRODUCTS: IndustryMetricProfile(
        key="software_products",
        display_name="Software Products",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="software_products",
        growth_context_text=(
            "Software product companies can sustain 20–35% growth given network effects. "
            "Revenue growth above 20% CAGR is excellent. High FCF margins (>20%) indicate "
            "strong product-market fit."
        ),
    ),
    IndustrySubType.SEMICONDUCTORS: IndustryMetricProfile(
        key="semiconductors",
        display_name="Semiconductors",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="semiconductors",
        cyclical=True,
        growth_context_text=(
            "Semiconductor companies are cyclical with high capital intensity. "
            "Growth above 15% CAGR is strong; ROCE above 15% despite heavy CapEx is excellent."
        ),
    ),
    IndustrySubType.ENGINEERING_SOFTWARE: IndustryMetricProfile(
        key="engineering_software",
        display_name="Engineering & Technology Software",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="it_services",
        growth_context_text=(
            "Engineering software companies target 12–20% growth. "
            "Subscription revenue and expanding margins indicate quality."
        ),
    ),

    # ── ENERGY ───────────────────────────────────────────────────────

    IndustrySubType.OIL_MARKETING: IndustryMetricProfile(
        key="oil_marketing",
        display_name="Oil Marketing",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="oil_marketing",
        cyclical=True,
        growth_context_text=(
            "Oil marketing companies operate on thin margins tied to commodity cycles. "
            "Revenue growth of 5–8% over a cycle is solid. ROCE above 10% is considered "
            "good for this capital-intensive, regulated sector."
        ),
    ),
    IndustrySubType.INTEGRATED_ENERGY: IndustryMetricProfile(
        key="integrated_energy",
        display_name="Integrated Energy",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="integrated_energy",
        cyclical=True,
        growth_context_text=(
            "Integrated energy companies are evaluated across cycles. "
            "Revenue growth of 8–12% is strong. Cash generation and disciplined CapEx "
            "matter more than short-term growth in this sector."
        ),
    ),
    IndustrySubType.EXPLORATION: IndustryMetricProfile(
        key="exploration",
        display_name="Oil & Gas Exploration",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="energy",
        cyclical=True,
        growth_context_text=(
            "E&P companies are highly cyclical. Multi-year reserves growth and FCF "
            "through commodity cycles matter more than single-year revenue growth."
        ),
    ),
    IndustrySubType.REFINING: IndustryMetricProfile(
        key="refining",
        display_name="Oil Refining",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="oil_marketing",
        cyclical=True,
        growth_context_text=(
            "Refining margins are cyclical. A 5–8% revenue CAGR over a cycle is solid. "
            "Complexity index and Nelson complexity drive long-term margin expansion."
        ),
    ),
    IndustrySubType.POWER_UTILITIES: IndustryMetricProfile(
        key="power_utilities",
        display_name="Power Utilities",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="utilities",
        cyclical=True,
        growth_context_text=(
            "Power utilities target regulated or contracted returns. "
            "Revenue growth of 5–10% is solid. Capital allocation and debt discipline "
            "matter more than top-line growth in this sector."
        ),
    ),

    # ── INDUSTRIALS / CYCLICALS ───────────────────────────────────────

    IndustrySubType.CEMENT: IndustryMetricProfile(
        key="cement",
        display_name="Cement",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="cyclical_industrial",
        cyclical=True,
        growth_context_text=(
            "Cement volumes track infrastructure and construction cycles. "
            "Revenue CAGR of 8–12% is strong. EBITDA/tonne and volume growth are "
            "more meaningful than nominal revenue growth."
        ),
    ),
    IndustrySubType.STEEL: IndustryMetricProfile(
        key="steel",
        display_name="Steel",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="cyclical_industrial",
        cyclical=True,
        growth_context_text=(
            "Steel is deeply cyclical. Revenue is driven by global commodity prices. "
            "ROCE of 10%+ through a cycle is excellent. Leverage and capacity utilization "
            "are critical indicators."
        ),
    ),
    IndustrySubType.METALS: IndustryMetricProfile(
        key="metals",
        display_name="Metals & Mining",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="cyclical_industrial",
        cyclical=True,
        growth_context_text=(
            "Metal prices drive cyclical swings. Evaluate ROCE on a 5-year average basis. "
            "Consistent cash generation through cycles indicates structural strength."
        ),
    ),
    IndustrySubType.MINING: IndustryMetricProfile(
        key="mining",
        display_name="Mining",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="cyclical_industrial",
        cyclical=True,
        growth_context_text=(
            "Mining profitability depends on commodity prices. "
            "Multi-cycle FCF and reserve replacement are critical measures of quality."
        ),
    ),
    IndustrySubType.SHIPPING: IndustryMetricProfile(
        key="shipping",
        display_name="Shipping",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="cyclical_industrial",
        cyclical=True,
        growth_context_text=(
            "Shipping is highly cyclical. Freight rates and fleet utilization drive margins. "
            "Evaluate performance across full freight cycles rather than single years."
        ),
    ),
    IndustrySubType.COMMODITY_CHEMICALS: IndustryMetricProfile(
        key="commodity_chemicals",
        display_name="Commodity Chemicals",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="cyclical_industrial",
        cyclical=True,
        growth_context_text=(
            "Commodity chemical margins are feedstock and demand driven. "
            "Revenue CAGR of 8–12% is solid. Specialty-to-commodity mix shift is a key quality indicator."
        ),
    ),
    IndustrySubType.FERTILIZERS: IndustryMetricProfile(
        key="fertilizers",
        display_name="Fertilizers",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="cyclical_industrial",
        cyclical=True,
        growth_context_text=(
            "Fertilizer profitability is linked to global commodity prices and subsidies. "
            "Evaluate cash generation over cycles rather than peak-year profitability."
        ),
    ),
    IndustrySubType.PAPER: IndustryMetricProfile(
        key="paper",
        display_name="Paper & Forest Products",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="cyclical_industrial",
        cyclical=True,
        growth_context_text=(
            "Paper is a cyclical commodity sector with low structural growth. "
            "Revenue CAGR of 5–8% is solid. Cost efficiency and vertical integration "
            "drive quality differentiation."
        ),
    ),
    IndustrySubType.ENGINEERING: IndustryMetricProfile(
        key="engineering",
        display_name="Industrial Engineering",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="industrial",
        growth_context_text=(
            "Industrial engineering companies target 10–15% revenue growth. "
            "Order book quality and ROCE are critical performance indicators."
        ),
    ),

    # ── CONSUMER ─────────────────────────────────────────────────────

    IndustrySubType.FMCG_LARGE_CAP: IndustryMetricProfile(
        key="fmcg_large_cap",
        display_name="FMCG (Large Cap)",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="fmcg_large_cap",
        growth_context_text=(
            "Large-cap FMCG companies typically grow 8–12% annually on a high base. "
            "Revenue growth above 10% CAGR is excellent. ROE above 30% and FCF margins "
            "above 15% are hallmarks of quality FMCG businesses."
        ),
    ),
    IndustrySubType.FMCG_MID_CAP: IndustryMetricProfile(
        key="fmcg_mid_cap",
        display_name="FMCG (Mid Cap)",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="fmcg_mid_cap",
        growth_context_text=(
            "Mid-cap FMCG companies can target 12–20% growth given a smaller base. "
            "Consistent margin expansion alongside revenue growth indicates quality."
        ),
    ),
    IndustrySubType.CONSUMER_DURABLES: IndustryMetricProfile(
        key="consumer_durables",
        display_name="Consumer Durables",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="industrial",
        growth_context_text=(
            "Consumer durables grow 12–18% driven by premiumization and market expansion. "
            "Working capital efficiency and brand investment are key quality signals."
        ),
    ),

    # ── HEALTHCARE ────────────────────────────────────────────────────

    IndustrySubType.PHARMA: IndustryMetricProfile(
        key="pharma",
        display_name="Pharmaceuticals",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="pharma",
        growth_context_text=(
            "Pharma companies target 10–18% revenue growth. "
            "R&D pipeline quality and US generics performance are critical. "
            "ROCE above 15% demonstrates strong capital allocation in this R&D-heavy sector."
        ),
    ),
    IndustrySubType.HOSPITALS: IndustryMetricProfile(
        key="hospitals",
        display_name="Hospitals & Healthcare",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="healthcare",
        growth_context_text=(
            "Hospital chains grow 12–20% driven by bed additions and ARPOB improvement. "
            "ROCE above 12% is solid given high CapEx for greenfield expansion."
        ),
    ),
    IndustrySubType.DIAGNOSTICS: IndustryMetricProfile(
        key="diagnostics",
        display_name="Diagnostics",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="healthcare",
        growth_context_text=(
            "Diagnostics companies target 12–18% growth. "
            "Test mix toward specialized tests and franchise network quality "
            "are key quality differentiators."
        ),
    ),

    # ── OTHERS ───────────────────────────────────────────────────────

    IndustrySubType.REAL_ESTATE: IndustryMetricProfile(
        key="real_estate",
        display_name="Real Estate",
        excluded_metrics=frozenset({"current_ratio", "interest_coverage"}),
        scoring_tier_preset="industrial",
        cyclical=True,
        growth_context_text=(
            "Real estate revenue follows project completion cycles. "
            "Pre-sales and collections are more meaningful than reported revenue. "
            "Net debt and land bank quality are critical balance sheet indicators."
        ),
    ),
    IndustrySubType.TELECOM: IndustryMetricProfile(
        key="telecom",
        display_name="Telecom",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="utilities",
        growth_context_text=(
            "Telecom is a capital-intensive, oligopolistic sector. "
            "ARPU growth and subscriber additions are key operating metrics. "
            "Revenue growth of 8–12% is solid given high fixed costs."
        ),
    ),
    IndustrySubType.MEDIA: IndustryMetricProfile(
        key="media",
        display_name="Media & Entertainment",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="industrial",
        growth_context_text=(
            "Media companies target 8–15% growth driven by digital transition. "
            "Subscription revenue mix and content investment efficiency are key."
        ),
    ),

    # ── FALLBACK ─────────────────────────────────────────────────────

    IndustrySubType.GENERAL: IndustryMetricProfile(
        key="general",
        display_name="General",
        excluded_metrics=_EMPTY,
        scoring_tier_preset="general",
        growth_context_text=(
            "Standard cross-sector benchmarks apply. "
            "Revenue growth above 10% CAGR is considered strong."
        ),
    ),
}


# ─────────────────────────────────────────────────────────────────────
# SECTOR EXCLUDED METRICS  (coarse level — sector-wide exclusions)
# ─────────────────────────────────────────────────────────────────────

SECTOR_EXCLUDED_METRICS: Dict[SectorType, FrozenSet[str]] = {
    SectorType.BANK:      _BANK_EXCLUDED,
    SectorType.NBFC:      _NBFC_EXCLUDED,
    SectorType.INSURANCE: _INSURANCE_EXCLUDED,
    SectorType.ENERGY:    _EMPTY,
    SectorType.TECHNOLOGY:_EMPTY,
    SectorType.HEALTHCARE:_EMPTY,
    SectorType.FMCG:      _EMPTY,
    SectorType.INDUSTRIAL:_EMPTY,
    SectorType.UNKNOWN:   _EMPTY,
}

# Weight boosts when coarse-level metrics are excluded (legacy; still used)
SECTOR_WEIGHT_BOOST: Dict[SectorType, Dict[str, float]] = {
    SectorType.BANK: {
        "roe": 0.04,
        "net_margin": 0.03,
        "profit_growth": 0.03,
        "eps_growth": 0.03,
        "book_value_growth": 0.02,
    },
    SectorType.NBFC: {
        "roe": 0.02,
        "debt_to_equity": 0.02,
        "net_margin": 0.02,
        "profit_growth": 0.02,
    },
    SectorType.INSURANCE: {
        "roe": 0.04,
        "net_margin": 0.03,
        "profit_growth": 0.03,
    },
}


# ─────────────────────────────────────────────────────────────────────
# LEGACY PROFILE DATACLASSES  (preserved for backward compatibility)
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IndustryProfile:
    key: str
    display_name: str
    excluded_metrics: FrozenSet[str] = field(default_factory=frozenset)
    weight_overrides: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


@dataclass(frozen=True)
class SectorProfile:
    sector_type: SectorType
    display_name: str
    excluded_metrics: FrozenSet[str] = field(default_factory=frozenset)
    explanation: str = ""
    industry_profile: IndustryProfile = field(
        default_factory=lambda: IndustryProfile(key="general", display_name="General")
    )
    industry_sub_type: IndustrySubType = IndustrySubType.GENERAL
    metric_profile: IndustryMetricProfile = field(
        default_factory=lambda: INDUSTRY_METRIC_PROFILES[IndustrySubType.GENERAL]
    )


# Legacy INDUSTRY_PROFILES (backward compat)
INDUSTRY_PROFILES: Dict[str, IndustryProfile] = {
    "banking": IndustryProfile(
        key="banking", display_name="Banking",
        excluded_metrics=_BANK_EXCLUDED,
        weight_overrides={
            "roe": 0.20, "net_margin": 0.14, "profit_growth": 0.14,
            "eps_growth": 0.12, "book_value_growth": 0.10, "dividend_growth": 0.10,
        },
        explanation="Banking peers evaluated on profitability, growth, and capital strength.",
    ),
    "technology": IndustryProfile(
        key="technology", display_name="Technology",
        excluded_metrics=frozenset(),
        weight_overrides={
            "roe": 0.16, "roce": 0.12, "operating_margin": 0.12,
            "revenue_growth": 0.12, "profit_growth": 0.10, "cash_flow_growth": 0.08,
        },
        explanation="Technology peers weighted toward growth, profitability, and capital efficiency.",
    ),
    "energy": IndustryProfile(
        key="energy", display_name="Energy", excluded_metrics=frozenset(),
        weight_overrides={
            "fcf_margin": 0.12, "cash_flow_growth": 0.12, "debt_to_equity": 0.10,
            "roe": 0.12, "roce": 0.10, "operating_margin": 0.10,
        },
        explanation="Energy peers weighted toward cash generation, leverage discipline, and capital efficiency.",
    ),
    "fmcg": IndustryProfile(
        key="fmcg", display_name="FMCG / Consumer Staples",
        excluded_metrics=frozenset(),
        weight_overrides={
            "operating_margin": 0.14, "net_margin": 0.12, "fcf_margin": 0.10,
            "roe": 0.12, "roce": 0.10, "revenue_growth": 0.08,
        },
        explanation="Consumer staples peers evaluated on margins, cash conversion, and returns.",
    ),
    "general": IndustryProfile(
        key="general", display_name="General",
        excluded_metrics=frozenset(),
        explanation="Standard peer-relative scoring applies.",
    ),
}


# ─────────────────────────────────────────────────────────────────────
# KEYWORD MATCHING TABLES
# ─────────────────────────────────────────────────────────────────────

# Format: list of (keyword_tuple, IndustrySubType) — evaluated top-down; first match wins
_SUB_TYPE_KEYWORD_RULES: List[Tuple[Tuple[str, ...], IndustrySubType]] = [
    # Banking
    (("small finance bank",),                      IndustrySubType.SMALL_FINANCE_BANK),
    (("private bank", "private sector bank"),       IndustrySubType.PRIVATE_BANK),
    (("public sector bank", "psu bank", "state bank"), IndustrySubType.PSU_BANK),
    (("bank", "banking"),                           IndustrySubType.PRIVATE_BANK),  # fallback

    # Financial Services
    (("life insurance",),                          IndustrySubType.INSURANCE_LIFE),
    (("general insurance", "non-life insurance"),  IndustrySubType.INSURANCE_GENERAL),
    (("insurance",),                               IndustrySubType.INSURANCE_LIFE),  # fallback
    (("asset management", "mutual fund", "wealth management"), IndustrySubType.ASSET_MANAGEMENT),
    (("nbfc", "microfinance", "housing finance", "consumer finance"), IndustrySubType.NBFC),
    (("financial services",),                      IndustrySubType.NBFC),

    # Technology
    (("semiconductor", "chip", "electronic components"), IndustrySubType.SEMICONDUCTORS),
    (("software product", "saas", "cloud software"), IndustrySubType.SOFTWARE_PRODUCTS),
    (("engineering software", "cad", "simulation", "plm"), IndustrySubType.ENGINEERING_SOFTWARE),
    (("information technology", "it services", "it consulting", "outsourcing"), IndustrySubType.IT_SERVICES),
    (("technology", "software", "internet"),       IndustrySubType.IT_SERVICES),

    # Energy
    (("oil marketing", "petroleum marketing"),     IndustrySubType.OIL_MARKETING),
    (("oil refin", "refinery"),                    IndustrySubType.REFINING),
    (("exploration", "e&p", "upstream oil"),       IndustrySubType.EXPLORATION),
    (("power", "electricity", "utilities"),        IndustrySubType.POWER_UTILITIES),
    (("oil", "gas", "petroleum", "energy"),        IndustrySubType.INTEGRATED_ENERGY),

    # Industrials / Cyclicals
    (("cement", "concrete"),                       IndustrySubType.CEMENT),
    (("steel", "iron and steel"),                  IndustrySubType.STEEL),
    (("mining", "quarry", "mineral extraction"),   IndustrySubType.MINING),
    (("metal", "aluminium", "copper", "zinc"),     IndustrySubType.METALS),
    (("shipping", "maritime", "freight"),          IndustrySubType.SHIPPING),
    (("fertilizer", "agrochemical"),               IndustrySubType.FERTILIZERS),
    (("commodity chemical", "chlor-alkali", "petrochemical basic"), IndustrySubType.COMMODITY_CHEMICALS),
    (("paper", "pulp", "packaging"),               IndustrySubType.PAPER),
    (("chemicals",),                               IndustrySubType.COMMODITY_CHEMICALS),  # fallback

    # Consumer
    (("personal care", "beauty", "grooming"),      IndustrySubType.FMCG_LARGE_CAP),
    (("food", "beverages", "household products", "consumer staples", "fmcg"), IndustrySubType.FMCG_LARGE_CAP),
    (("consumer discretionary", "consumer durables", "appliances"), IndustrySubType.CONSUMER_DURABLES),

    # Healthcare
    (("pharmaceuticals", "pharma", "drug"),        IndustrySubType.PHARMA),
    (("hospital", "healthcare services"),          IndustrySubType.HOSPITALS),
    (("diagnostic", "pathology", "laboratory"),    IndustrySubType.DIAGNOSTICS),

    # Others
    (("real estate", "reit", "property"),          IndustrySubType.REAL_ESTATE),
    (("telecom", "telecommunication", "wireless"),  IndustrySubType.TELECOM),
    (("media", "entertainment", "broadcasting"),   IndustrySubType.MEDIA),
    (("engineering", "capital goods", "heavy machinery"), IndustrySubType.ENGINEERING),
]

# Hard-coded overrides for known edge-case companies (Yahoo Finance misclassifications)
_TICKER_OVERRIDES: Dict[str, IndustrySubType] = {
    "RELIANCE": IndustrySubType.INTEGRATED_ENERGY,
    "RELIANCEIND": IndustrySubType.INTEGRATED_ENERGY,
    "ONGC": IndustrySubType.EXPLORATION,
    "BPCL": IndustrySubType.OIL_MARKETING,
    "HPCL": IndustrySubType.OIL_MARKETING,
    "IOC": IndustrySubType.OIL_MARKETING,
    "IOCL": IndustrySubType.OIL_MARKETING,
    "GAIL": IndustrySubType.INTEGRATED_ENERGY,
    "COALINDIA": IndustrySubType.MINING,
    "HINDALCO": IndustrySubType.METALS,
    "JSWSTEEL": IndustrySubType.STEEL,
    "TATASTEEL": IndustrySubType.STEEL,
    "SAIL": IndustrySubType.STEEL,
    "NTPC": IndustrySubType.POWER_UTILITIES,
    "POWERGRID": IndustrySubType.POWER_UTILITIES,
    "BAJAJFINSV": IndustrySubType.INSURANCE_LIFE,
    "SBILIFE": IndustrySubType.INSURANCE_LIFE,
    "HDFCLIFE": IndustrySubType.INSURANCE_LIFE,
    "ICICIPRULI": IndustrySubType.INSURANCE_LIFE,
    "GICRE": IndustrySubType.INSURANCE_GENERAL,
    "NIACL": IndustrySubType.INSURANCE_GENERAL,
    "BAJAJFINANCE": IndustrySubType.NBFC,
    "BAJFINANCE": IndustrySubType.NBFC,
    "HDFC": IndustrySubType.NBFC,
    "MUTHOOTFIN": IndustrySubType.NBFC,
    "CHOLAFIN": IndustrySubType.NBFC,
    "HDFCBANK": IndustrySubType.PRIVATE_BANK,
    "ICICIBANK": IndustrySubType.PRIVATE_BANK,
    "AXISBANK": IndustrySubType.PRIVATE_BANK,
    "KOTAKBANK": IndustrySubType.PRIVATE_BANK,
    "INDUSIND": IndustrySubType.PRIVATE_BANK,
    "FEDERALBNK": IndustrySubType.PRIVATE_BANK,
    "SBIN": IndustrySubType.PSU_BANK,
    "BANKBARODA": IndustrySubType.PSU_BANK,
    "CANARABANK": IndustrySubType.PSU_BANK,
    "PNB": IndustrySubType.PSU_BANK,
    "NESTLEIND": IndustrySubType.FMCG_LARGE_CAP,
    "HINDUNILVR": IndustrySubType.FMCG_LARGE_CAP,
    "DABUR": IndustrySubType.FMCG_LARGE_CAP,
    "GODREJCP": IndustrySubType.FMCG_LARGE_CAP,
    "MARICO": IndustrySubType.FMCG_LARGE_CAP,
    "BRITANNIA": IndustrySubType.FMCG_LARGE_CAP,
    "ITC": IndustrySubType.FMCG_LARGE_CAP,
    "TCS": IndustrySubType.IT_SERVICES,
    "INFY": IndustrySubType.IT_SERVICES,
    "WIPRO": IndustrySubType.IT_SERVICES,
    "HCLTECH": IndustrySubType.IT_SERVICES,
    "TECHM": IndustrySubType.IT_SERVICES,
    "MPHASIS": IndustrySubType.IT_SERVICES,
    "LTIMINDTREE": IndustrySubType.IT_SERVICES,
    "PERSISTENT": IndustrySubType.SOFTWARE_PRODUCTS,
    "COFORGE": IndustrySubType.IT_SERVICES,
    "DRREDDY": IndustrySubType.PHARMA,
    "SUNPHARMA": IndustrySubType.PHARMA,
    "CIPLA": IndustrySubType.PHARMA,
    "LUPIN": IndustrySubType.PHARMA,
    "DIVISLAB": IndustrySubType.PHARMA,
    "APOLLOHOSP": IndustrySubType.HOSPITALS,
    "MAXHEALTH": IndustrySubType.HOSPITALS,
    "METROPOLIS": IndustrySubType.DIAGNOSTICS,
    "DRPATH": IndustrySubType.DIAGNOSTICS,
    "ULTRACEMCO": IndustrySubType.CEMENT,
    "AMBUJACEM": IndustrySubType.CEMENT,
    "ACCCEMENT": IndustrySubType.CEMENT,
    "SHREECEM": IndustrySubType.CEMENT,
}


# ─────────────────────────────────────────────────────────────────────
# CLASSIFICATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def classify_industry_sub_type(
    sector: str,
    industry: str,
    ticker: str = "",
) -> IndustrySubType:
    """
    Classifies a company into an IndustrySubType using:
    1. Known ticker overrides (edge cases / conglomerates)
    2. Keyword matching on Yahoo Finance sector + industry strings
    3. Fallback to GENERAL

    Never forces an incorrect classification — if confidence is low,
    returns GENERAL and lets the system fall back to sector-level rules.
    """
    # Step 1: ticker override
    ticker_clean = (ticker or "").upper().replace(".NS", "").replace(".BO", "")
    if ticker_clean in _TICKER_OVERRIDES:
        return _TICKER_OVERRIDES[ticker_clean]

    # Step 2: keyword matching (case-insensitive, combined sector+industry)
    combined = f"{(sector or '').lower()} {(industry or '').lower()}"

    for keywords, sub_type in _SUB_TYPE_KEYWORD_RULES:
        if any(kw in combined for kw in keywords):
            return sub_type

    return IndustrySubType.GENERAL


def classify_sector(sector: str, industry: str, ticker: str = "") -> SectorProfile:
    """
    Returns the SectorProfile for a company, including its IndustrySubType and
    IndustryMetricProfile. Used by SectorEngine.
    """
    sub_type = classify_industry_sub_type(sector, industry, ticker)
    metric_profile = INDUSTRY_METRIC_PROFILES.get(sub_type, INDUSTRY_METRIC_PROFILES[IndustrySubType.GENERAL])

    sector_l   = (sector or "").lower()
    industry_l = (industry or "").lower()
    combined   = f"{sector_l} {industry_l}"

    # Map sub_type → coarse SectorType
    bank_subs      = {IndustrySubType.PRIVATE_BANK, IndustrySubType.PSU_BANK, IndustrySubType.SMALL_FINANCE_BANK}
    insurance_subs = {IndustrySubType.INSURANCE_LIFE, IndustrySubType.INSURANCE_GENERAL}
    energy_subs    = {IndustrySubType.OIL_MARKETING, IndustrySubType.INTEGRATED_ENERGY,
                      IndustrySubType.EXPLORATION, IndustrySubType.REFINING, IndustrySubType.POWER_UTILITIES}
    tech_subs      = {IndustrySubType.IT_SERVICES, IndustrySubType.SOFTWARE_PRODUCTS,
                      IndustrySubType.SEMICONDUCTORS, IndustrySubType.ENGINEERING_SOFTWARE}
    fmcg_subs      = {IndustrySubType.FMCG_LARGE_CAP, IndustrySubType.FMCG_MID_CAP}
    health_subs    = {IndustrySubType.PHARMA, IndustrySubType.HOSPITALS, IndustrySubType.DIAGNOSTICS}

    if sub_type in bank_subs:
        sector_type = SectorType.BANK
        display_name = "Banking"
        explanation = (
            f"{metric_profile.display_name} companies are scored on ROE, net margin, and growth. "
            "Industrial leverage ratios are excluded as bank balance sheets follow different accounting norms."
        )
    elif sub_type == IndustrySubType.NBFC:
        sector_type = SectorType.NBFC
        display_name = "NBFC / Financial Services"
        explanation = "NBFCs use adjusted leverage benchmarks; industrial liquidity ratios are excluded."
    elif sub_type in insurance_subs:
        sector_type = SectorType.INSURANCE
        display_name = "Insurance"
        explanation = "Insurance companies are scored on ROE, margins, and growth — not industrial leverage."
    elif sub_type in energy_subs:
        sector_type = SectorType.ENERGY
        display_name = "Energy"
        explanation = f"{metric_profile.display_name} companies are evaluated with adjusted expectations for capital-intensive, cyclical operations."
    elif sub_type in tech_subs:
        sector_type = SectorType.TECHNOLOGY
        display_name = "Technology"
        explanation = f"{metric_profile.display_name} companies are evaluated with emphasis on growth and capital efficiency."
    elif sub_type in fmcg_subs or sub_type == IndustrySubType.CONSUMER_DURABLES:
        sector_type = SectorType.FMCG
        display_name = "Consumer"
        explanation = f"{metric_profile.display_name} companies are evaluated on margins, cash conversion, and returns."
    elif sub_type in health_subs:
        sector_type = SectorType.HEALTHCARE
        display_name = "Healthcare"
        explanation = f"{metric_profile.display_name} companies are evaluated on growth, margins, and R&D efficiency."
    elif sub_type == IndustrySubType.ASSET_MANAGEMENT:
        sector_type = SectorType.NBFC
        display_name = "Asset Management"
        explanation = "Asset managers evaluated on AUM growth, margins, and ROE."
    else:
        sector_type = SectorType.INDUSTRIAL
        display_name = sector or "Industrial"
        explanation = f"Standard industrial benchmarks apply for {metric_profile.display_name}."

    # Legacy IndustryProfile (backward compat — map from sector_type)
    _legacy_map = {
        SectorType.BANK: "banking",
        SectorType.NBFC: "banking",
        SectorType.INSURANCE: "banking",
        SectorType.ENERGY: "energy",
        SectorType.TECHNOLOGY: "technology",
        SectorType.FMCG: "fmcg",
    }
    legacy_key = _legacy_map.get(sector_type, "general")
    legacy_profile = INDUSTRY_PROFILES[legacy_key]

    return SectorProfile(
        sector_type=sector_type,
        display_name=display_name,
        excluded_metrics=metric_profile.excluded_metrics,
        explanation=explanation,
        industry_profile=legacy_profile,
        industry_sub_type=sub_type,
        metric_profile=metric_profile,
    )


# Public exports
SUPPORTED_SECTORS: List[str] = [e.value for e in SectorType]
