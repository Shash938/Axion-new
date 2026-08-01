"""
services/peer_engine.py — Industry Peer Analysis Engine
=========================================================
Why this file exists:
    Provides dynamic peer comparison metrics (percentiles, quartiles, averages).
    In Stage 1, it defines the interface and math but defaults to returning None
    since no real peer database is connected yet.
    In Stage 2, it will connect to a peer data fetcher to compute live relative rankings.
"""

import numpy as np
from typing import List, Optional
import logging

from models.fundamental import PeerMetrics

logger = logging.getLogger(__name__)


class PeerAnalysisEngine:
    """
    Evaluates target company metrics against a list of peers.
    """

    def __init__(self, peer_data: Optional[List[dict]] = None) -> None:
        """
        Initialise with optional pre-fetched peer data.
        """
        self.peer_data = peer_data or []

    def extract_peer_values(self, metric_key: str) -> List[float]:
        """
        Extracts the values for a specific metric key from the peer data.
        (Placeholder for Stage 2).
        """
        if not self.peer_data:
            return []
            
        values = []
        for peer in self.peer_data:
            val = peer.get(metric_key)
            if val is not None:
                values.append(float(val))
        return values

    def evaluate(self, target_value: Optional[float], peer_values: List[float], lower_is_better: bool = False) -> Optional[PeerMetrics]:
        """
        Computes percentile, quartiles, average, and median against peer group.
        
        Returns:
            PeerMetrics object if peer data exists, otherwise None.
        """
        if target_value is None or not peer_values:
            return None
            
        clean_peers = [v for v in peer_values if v is not None]
        if not clean_peers:
            return None
            
        peer_array = np.array(clean_peers)
        total_peers = len(clean_peers)
        peer_average = float(np.mean(peer_array))
        peer_median = float(np.median(peer_array))
        
        if not lower_is_better:
            better_than = sum(1 for v in clean_peers if target_value > v)
            rank = sum(1 for v in clean_peers if target_value < v) + 1
        else:
            better_than = sum(1 for v in clean_peers if target_value < v)
            rank = sum(1 for v in clean_peers if target_value > v) + 1
            
        percentile = (better_than / total_peers) * 100.0
        
        # Quartile: 1 is top 25%, 4 is bottom 25%
        if percentile >= 75:
            quartile = 1
        elif percentile >= 50:
            quartile = 2
        elif percentile >= 25:
            quartile = 3
        else:
            quartile = 4
            
        difference = target_value - peer_average
        
        return PeerMetrics(
            industry_rank=rank,
            total_peers=total_peers,
            percentile=round(percentile, 2),
            quartile=quartile,
            peer_average=round(peer_average, 4),
            peer_median=round(peer_median, 4),
            difference_from_average=round(difference, 4)
        )
