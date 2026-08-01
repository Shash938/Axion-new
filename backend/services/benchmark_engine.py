"""
services/benchmark_engine.py — Benchmark Evaluation Engine
==========================================================
Evaluates metric values against professional benchmark bands.
Supports sector-specific overrides via config/benchmark_rules.py.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from config.benchmark_rules import (
    DEFAULT_BENCHMARKS,
    SECTOR_BENCHMARK_OVERRIDES,
    BenchmarkBand,
    MetricBenchmark,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of benchmark evaluation for a single metric."""
    label: str
    description: str
    band_summary: str
    meets_excellent: bool = False


class BenchmarkEngine:
    """Looks up and evaluates metrics against benchmark tables."""

    def get_benchmark(self, metric_key: str, sector: str = "Unknown", industry: str = "Unknown") -> Optional[MetricBenchmark]:
        """Returns sector/industry-specific benchmark if available, else default."""
        profile = self._profile_key(sector, industry)
        if profile == "banking":
            profile_overrides = {
                "roe": self._build_benchmark("roe", "ROE", "%", ((16.0, "Excellent"), (13.0, "Good"), (10.0, "Average"), (None, "Weak")), False),
                "net_margin": self._build_benchmark("net_margin", "Net Margin", "%", ((14.0, "Excellent"), (11.0, "Good"), (8.0, "Average"), (None, "Weak")), False),
                "profit_growth": self._build_benchmark("profit_growth", "Profit Growth", "%", ((12.0, "Excellent"), (8.0, "Good"), (5.0, "Average"), (None, "Weak")), False),
                "revenue_growth": self._build_benchmark("revenue_growth", "Revenue Growth", "%", ((9.0, "Excellent"), (6.0, "Good"), (3.0, "Average"), (None, "Weak")), False),
            }
            if metric_key in profile_overrides:
                return profile_overrides[metric_key]
        elif profile == "technology":
            profile_overrides = {
                "roe": self._build_benchmark("roe", "ROE", "%", ((18.0, "Excellent"), (14.0, "Good"), (10.0, "Average"), (None, "Weak")), False),
                "revenue_growth": self._build_benchmark("revenue_growth", "Revenue Growth", "%", ((12.0, "Excellent"), (8.0, "Good"), (4.0, "Average"), (None, "Weak")), False),
                "profit_growth": self._build_benchmark("profit_growth", "Profit Growth", "%", ((12.0, "Excellent"), (8.0, "Good"), (4.0, "Average"), (None, "Weak")), False),
            }
            if metric_key in profile_overrides:
                return profile_overrides[metric_key]
        elif profile == "energy":
            profile_overrides = {
                "roe": self._build_benchmark("roe", "ROE", "%", ((12.0, "Excellent"), (10.0, "Good"), (8.0, "Average"), (None, "Weak")), False),
                "roce": self._build_benchmark("roce", "ROCE", "%", ((12.0, "Excellent"), (10.0, "Good"), (8.0, "Average"), (None, "Weak")), False),
                "fcf_margin": self._build_benchmark("fcf_margin", "FCF Margin", "%", ((10.0, "Excellent"), (7.0, "Good"), (3.0, "Average"), (None, "Weak")), False),
            }
            if metric_key in profile_overrides:
                return profile_overrides[metric_key]
        elif profile == "fmcg":
            profile_overrides = {
                "roe": self._build_benchmark("roe", "ROE", "%", ((18.0, "Excellent"), (14.0, "Good"), (10.0, "Average"), (None, "Weak")), False),
                "roce": self._build_benchmark("roce", "ROCE", "%", ((16.0, "Excellent"), (12.0, "Good"), (8.0, "Average"), (None, "Weak")), False),
                "operating_margin": self._build_benchmark("operating_margin", "Operating Margin", "%", ((18.0, "Excellent"), (14.0, "Good"), (10.0, "Average"), (None, "Weak")), False),
            }
            if metric_key in profile_overrides:
                return profile_overrides[metric_key]

        sector_overrides = SECTOR_BENCHMARK_OVERRIDES.get(sector, {})
        if metric_key in sector_overrides:
            return sector_overrides[metric_key]
        return DEFAULT_BENCHMARKS.get(metric_key)

    def evaluate(
        self,
        metric_key: str,
        value: Optional[float],
        sector: str = "Unknown",
        industry: str = "Unknown",
    ) -> Optional[BenchmarkResult]:
        """Evaluates a value against benchmark bands. Returns None if no benchmark defined."""
        benchmark = self.get_benchmark(metric_key, sector, industry)
        if benchmark is None or value is None:
            return None

        band_summary = " | ".join(b.description for b in benchmark.bands if b.description)
        label = self._classify_value(benchmark, value)
        meets_excellent = label == "Excellent"

        return BenchmarkResult(
            label=label,
            description=benchmark.bands[0].description if benchmark.bands else "",
            band_summary=band_summary,
            meets_excellent=meets_excellent,
        )

    @staticmethod
    def _profile_key(sector: str, industry: str) -> str:
        combined = f"{(sector or '').lower()} {(industry or '').lower()}"
        if any(token in combined for token in ("bank", "banks", "regional bank")):
            return "banking"
        if any(token in combined for token in ("technology", "it services", "software", "semiconductor")):
            return "technology"
        if any(token in combined for token in ("energy", "oil", "gas", "petroleum", "refining", "power")):
            return "energy"
        if any(token in combined for token in ("fmcg", "consumer staples", "packaged foods", "personal care", "paint", "coatings")):
            return "fmcg"
        return "general"

    @staticmethod
    def _build_benchmark(key: str, display_name: str, unit: str, thresholds, lower_is_better: bool) -> MetricBenchmark:
        bands = []
        for threshold, label in thresholds:
            if threshold is None:
                bands.append(BenchmarkBand(label, description=label))
            else:
                bands.append(BenchmarkBand(label, min_value=threshold, description=f">={threshold}"))
        return MetricBenchmark(key=key, display_name=display_name, unit=unit, bands=tuple(bands), lower_is_better=lower_is_better)

    @staticmethod
    def _classify_value(benchmark: MetricBenchmark, value: float) -> str:
        """Returns the label of the matching benchmark band."""
        if benchmark.lower_is_better:
            for band in benchmark.bands:
                if band.max_value is not None and value < band.max_value:
                    return band.label
                if band.min_value is not None and value >= band.min_value:
                    return band.label
            return benchmark.bands[-1].label if benchmark.bands else "Unknown"

        for band in benchmark.bands:
            if band.min_value is not None and value >= band.min_value:
                if band.max_value is None or value < band.max_value:
                    return band.label
            if band.max_value is not None and band.min_value is None and value < band.max_value:
                return band.label
        return benchmark.bands[-1].label if benchmark.bands else "Unknown"
