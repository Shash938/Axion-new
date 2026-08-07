import pytest
from pydantic import ValidationError
from models.fundamental import AnalysisRequest, Exchange, FundamentalScore


def test_analysis_request_valid_ticker():
    req = AnalysisRequest(ticker="tcs", exchange=Exchange.NSE)
    assert req.ticker == "TCS"
    assert req.exchange == Exchange.NSE


def test_analysis_request_strips_suffix():
    req = AnalysisRequest(ticker="RELIANCE.NS", exchange=Exchange.NSE)
    assert req.ticker == "RELIANCE"


def test_analysis_request_empty_ticker_rejected():
    with pytest.raises(ValidationError):
        AnalysisRequest(ticker="   ")


def test_analysis_request_malicious_ticker_rejected():
    with pytest.raises(ValidationError):
        AnalysisRequest(ticker="<script>alert(1)</script>")

    with pytest.raises(ValidationError):
        AnalysisRequest(ticker="INVALID TICKER!")


def test_fundamental_score_defaults():
    from models.fundamental import Grade, Recommendation

    score = FundamentalScore(
        total_score=5.5,
        grade=Grade.B,
        recommendation=Recommendation.HOLD,
        metrics_evaluated=10,
        metrics_total=14,
        coverage_pct=71.4,
    )
    assert score.total_score == 5.5
    assert score.business_quality_score == 0.0
    assert score.valuation_score == 5.0
    assert score.risk_score == 5.0
    assert score.grade == Grade.B
