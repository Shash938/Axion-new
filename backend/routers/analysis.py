"""
routers/analysis.py — Analysis API Router
==========================================
Why this file exists:
    Keeps all analysis-related route definitions in a dedicated module,
    separate from app setup. FastAPI's APIRouter provides clean modularity
    — adding a new router (e.g. /portfolio) never requires touching this file.

How it connects:
    - Included in app.py via `app.include_router(analysis_router)`.
    - Instantiates `FundamentalAnalyzerService` once at module level (singleton).
    - Maps `DataFetcherService` custom exceptions → HTTP error responses.

Error mapping:
    TickerNotFoundError  → HTTP 404
    DataUnavailableError → HTTP 422 (Unprocessable Entity)
    NetworkError         → HTTP 503 (Service Unavailable)
    Exception            → HTTP 500

Possible improvements:
    - Use FastAPI's Depends() for service injection when the app grows.
    - Add response caching (Redis) at the router level with cache-control headers.
    - Add rate limiting middleware per-IP when exposed publicly.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from security.auth import verify_api_key, get_current_user_optional
from security.rate_limiter import rate_limit_check
from database.db import record_search_history

from models.fundamental import AnalysisRequest, AnalysisResponse, ErrorDetail, Exchange
from services.data_fetcher import DataUnavailableError, NetworkError, TickerNotFoundError
from services.fundamental_analyzer import FundamentalAnalyzerService

logger = logging.getLogger(__name__)

# Single shared instance — services are stateless and safe to share
_analyzer = FundamentalAnalyzerService()

analysis_router = APIRouter(
    prefix="/api/v1",
    tags=["Fundamental Analysis"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit_check)],
)


# ==============================================================================
# POST /api/v1/analyze
# ==============================================================================


@analysis_router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse a stock's fundamentals",
    description=(
        "Fetches financial data for the given ticker, calculates 14 fundamental ratios, "
        "scores each metric, and returns a structured JSON response with an investment "
        "recommendation and beginner-friendly explanations."
    ),
    responses={
        404: {"model": ErrorDetail, "description": "Ticker not found on the exchange."},
        422: {"model": ErrorDetail, "description": "Ticker exists but financial data is unavailable."},
        503: {"model": ErrorDetail, "description": "Unable to connect to market data provider."},
        500: {"model": ErrorDetail, "description": "Unexpected server error."},
    },
)
def analyze_stock(
    request: AnalysisRequest,
    current_user: dict = Depends(get_current_user_optional),
) -> AnalysisResponse:
    """
    Main analysis endpoint.

    Accepts a JSON body with `ticker` and optional `exchange` (default: NSE).
    Example request:
        POST /api/v1/analyze
        {"ticker": "RELIANCE", "exchange": "NSE"}
    """
    logger.info("POST /api/v1/analyze — ticker=%s exchange=%s", request.ticker, request.exchange)
    return _run_analysis(request, current_user)


# ==============================================================================
# GET /api/v1/analyze/{ticker}
# ==============================================================================


@analysis_router.get(
    "/analyze/{ticker}",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse a stock's fundamentals (GET)",
    description=(
        "Convenience GET endpoint. Accepts the ticker as a path parameter "
        "and optionally the exchange as a query parameter."
    ),
    responses={
        404: {"model": ErrorDetail, "description": "Ticker not found on the exchange."},
        422: {"model": ErrorDetail, "description": "Ticker exists but financial data is unavailable."},
        503: {"model": ErrorDetail, "description": "Unable to connect to market data provider."},
        500: {"model": ErrorDetail, "description": "Unexpected server error."},
    },
)
def analyze_stock_get(
    ticker: str = Path(
        ...,
        min_length=1,
        max_length=20,
        description="NSE/BSE stock ticker (e.g. TCS, INFY, HDFC).",
        openapi_examples={"default": {"summary": "TCS", "value": "TCS"}},
    ),
    exchange: Exchange = Query(
        default=Exchange.NSE,
        description="Stock exchange (NSE or BSE).",
    ),
    current_user: dict = Depends(get_current_user_optional),
) -> AnalysisResponse:
    """
    Convenience GET endpoint for browser / extension quick lookups.
    Example: GET /api/v1/analyze/TCS?exchange=NSE
    """
    logger.info("GET /api/v1/analyze/%s?exchange=%s", ticker, exchange)
    request = AnalysisRequest(ticker=ticker, exchange=exchange)
    return _run_analysis(request, current_user)


# ==============================================================================
# Shared execution helper
# ==============================================================================


def _run_analysis(request: AnalysisRequest, current_user: Optional[dict] = None) -> AnalysisResponse:
    """
    Centralised error handling wrapper for both endpoints.
    Maps domain exceptions to appropriate HTTP status codes.
    """
    try:
        response = _analyzer.analyze(request)
        
        # Record successful search in history database
        user_id = current_user["id"] if current_user else None
        try:
            record_search_history(
                user_id=user_id,
                ticker=response.company.ticker,
                exchange=response.company.exchange.value,
                company_name=response.company.company_name,
                score=response.fundamental_score.total_score,
                grade=response.fundamental_score.grade.value,
                recommendation=response.fundamental_score.recommendation.value,
            )
        except Exception as hist_err:
            logger.warning("Failed to record search history: %s", hist_err)
            
        return response

    except TickerNotFoundError as exc:
        logger.warning("Ticker not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "type": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404",
                "title": "Ticker Not Found",
                "status": 404,
                "detail": str(exc),
                "ticker": exc.ticker,
            },
        ) from exc

    except DataUnavailableError as exc:
        logger.warning("Data unavailable for ticker: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "type": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/422",
                "title": "Financial Data Unavailable",
                "status": 422,
                "detail": str(exc),
                "ticker": exc.ticker,
            },
        ) from exc

    except NetworkError as exc:
        logger.error("Network error fetching data for ticker %s: %s", exc.ticker, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "type": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/503",
                "title": "Market Data Provider Unavailable",
                "status": 503,
                "detail": (
                    "Unable to connect to the market data provider. "
                    "Please try again in a few moments."
                ),
                "ticker": exc.ticker,
            },
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error during analysis for %s: %s", request.ticker, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "type": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500",
                "title": "Internal Server Error",
                "status": 500,
                "detail": "An unexpected error occurred. Our team has been notified.",
                "ticker": request.ticker,
            },
        ) from exc
