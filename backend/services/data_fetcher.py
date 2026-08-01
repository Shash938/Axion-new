"""
services/data_fetcher.py — Financial Data Fetching Service
===========================================================
Why this file exists:
    Single-responsibility service for ALL external data retrieval.
    Abstracts yfinance's API behind a clean interface so that:
      a) The rest of the codebase never imports yfinance directly.
      b) yfinance can be swapped for another data provider with zero changes
         to any other service.

How it connects:
    - Called by services/fundamental_analyzer.py (the pipeline orchestrator).
    - Returns raw Pandas DataFrames and a dict of stock metadata.
    - The output is passed to services/data_cleaner.py for validation.

Error hierarchy:
    DataFetchError (base)
    ├── TickerNotFoundError   — ticker symbol doesn't exist on the exchange
    ├── DataUnavailableError  — ticker is valid but financials are not available
    └── NetworkError          — connectivity or timeout issues with yfinance

Possible improvements:
    - Add Redis caching layer to avoid hitting yfinance on every request.
    - Add a fallback provider (e.g. Alpha Vantage, Screener.in) when yfinance fails.
    - Rate-limit using a token bucket to avoid yfinance throttling in production.
"""

import logging
import time
from typing import Any, Dict, Optional

import pandas as pd
import urllib3
import yfinance as yf
from curl_cffi.requests import Session as CurlSession

from config import get_settings

# Disable SSL warning messages in console for verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


# ==============================================================================
# Custom Exceptions
# ==============================================================================


class DataFetchError(Exception):
    """Base exception for all data fetching failures."""
    def __init__(self, message: str, ticker: Optional[str] = None) -> None:
        self.ticker = ticker
        super().__init__(message)


class TickerNotFoundError(DataFetchError):
    """Raised when no yfinance data exists for the supplied ticker symbol."""
    pass


class DataUnavailableError(DataFetchError):
    """Raised when a ticker is found but required financial statements are missing."""
    pass


class NetworkError(DataFetchError):
    """Raised when yfinance HTTP requests fail due to connectivity issues."""
    pass


# ==============================================================================
# Raw Financial Data Container
# ==============================================================================


class RawFinancialData:
    """
    Plain container for all raw data returned by DataFetcherService.
    Using a class (rather than a TypedDict or dataclass) allows optional
    DataFrames without complex Optional[DataFrame] typing noise.
    """

    __slots__ = (
        "ticker",
        "exchange",
        "yf_ticker",
        "info",
        "fast_info",
        "income_stmt",
        "balance_sheet",
        "cash_flow",
        "quarterly_income_stmt",
        "quarterly_balance_sheet",
        "quarterly_cash_flow",
        "dividends",
    )

    def __init__(
        self,
        ticker: str,
        exchange: str,
        yf_ticker: str,
        info: Dict[str, Any],
        fast_info: Dict[str, Any],
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
        quarterly_income_stmt: pd.DataFrame,
        quarterly_balance_sheet: pd.DataFrame,
        quarterly_cash_flow: pd.DataFrame,
        dividends: pd.Series,
    ) -> None:
        self.ticker = ticker
        self.exchange = exchange
        self.yf_ticker = yf_ticker
        self.info = info
        self.fast_info = fast_info
        self.income_stmt = income_stmt
        self.balance_sheet = balance_sheet
        self.cash_flow = cash_flow
        self.quarterly_income_stmt = quarterly_income_stmt
        self.quarterly_balance_sheet = quarterly_balance_sheet
        self.quarterly_cash_flow = quarterly_cash_flow
        self.dividends = dividends


# ==============================================================================
# Service
# ==============================================================================


class DataFetcherService:
    """
    Fetches all required financial data for a given NSE/BSE stock ticker.

    Instantiation:
        fetcher = DataFetcherService()

    Usage:
        raw_data = fetcher.fetch(ticker="RELIANCE", exchange="NSE")

    Retry Strategy:
        yfinance can be flaky. We implement simple exponential backoff with
        a configurable maximum number of retries (default: 3).
    """

    _MAX_RETRIES: int = 3
    _BACKOFF_BASE: float = 1.5  # seconds; delay = base ^ attempt

    def __init__(self) -> None:
        self._settings = get_settings()
        self._session = CurlSession(impersonate="chrome")
        self._session.verify = False
        logger.info("DataFetcherService initialised.")

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def fetch(self, ticker: str, exchange: str) -> RawFinancialData:
        """
        Main entry point. Fetches all financial data for `ticker`.

        Args:
            ticker:   Normalised ticker symbol without exchange suffix (e.g. "RELIANCE").
            exchange: Exchange string from the AnalysisRequest ("NSE" or "BSE").

        Returns:
            RawFinancialData container with all statements and metadata.

        Raises:
            TickerNotFoundError: If yfinance returns no data for the ticker.
            DataUnavailableError: If financial statements are empty.
            NetworkError: If an HTTP/timeout error occurs after all retries.
        """
        yf_ticker = self._build_yf_ticker(ticker, exchange)
        logger.info("Fetching data for %s (yfinance: %s)", ticker, yf_ticker)

        ticker_obj = self._get_ticker_with_retry(yf_ticker, ticker)
        info = self._extract_info(ticker_obj, ticker, yf_ticker)

        # Safe extraction of fast_info — primary source for market cap
        fast_info: Dict[str, Any] = {}
        try:
            fi = ticker_obj.fast_info
            # fast_info may be a dict-like object or attribute bag
            if hasattr(fi, "items"):
                for k, v in fi.items():
                    if v is not None:
                        fast_info[k] = v
            mc = (
                fast_info.get("marketCap")
                or fast_info.get("market_cap")
                or getattr(fi, "market_cap", None)
                or getattr(fi, "marketCap", None)
            )
            shares = (
                fast_info.get("sharesOutstanding")
                or fast_info.get("shares")
                or getattr(fi, "shares", None)
                or getattr(fi, "shares_outstanding", None)
            )
            last_price = (
                fast_info.get("lastPrice")
                or fast_info.get("last_price")
                or getattr(fi, "last_price", None)
                or getattr(fi, "lastPrice", None)
            )
            currency = fast_info.get("currency") or getattr(fi, "currency", None)

            if mc is not None and float(mc) > 0:
                fast_info["marketCap"] = float(mc)
            if shares is not None and float(shares) > 0:
                fast_info["sharesOutstanding"] = float(shares)
            if last_price is not None:
                fast_info["lastPrice"] = float(last_price)
            if currency:
                fast_info["currency"] = currency
        except Exception as exc:
            logger.warning("Could not retrieve fast_info for %s: %s", ticker, exc)

        income_stmt = self._safe_get_df(ticker_obj, "income_stmt", ticker)
        balance_sheet = self._safe_get_df(ticker_obj, "balance_sheet", ticker)
        cash_flow = self._safe_get_df(ticker_obj, "cashflow", ticker)
        quarterly_income_stmt = self._safe_get_df(ticker_obj, "quarterly_income_stmt", ticker)
        quarterly_balance_sheet = self._safe_get_df(ticker_obj, "quarterly_balance_sheet", ticker)
        quarterly_cash_flow = self._safe_get_df(ticker_obj, "quarterly_cashflow", ticker)
        dividends = self._fetch_dividends(ticker_obj, ticker)

        # Verify at least income statement and balance sheet are present
        if income_stmt.empty and balance_sheet.empty:
            raise DataUnavailableError(
                f"No annual financial statements available for '{ticker}' on {exchange}. "
                "This can happen for very new listings, ETFs, or index funds.",
                ticker=ticker,
            )

        logger.info(
            "Successfully fetched data for %s — %d years of income data, "
            "%d years of balance sheet data.",
            yf_ticker,
            len(income_stmt.columns) if not income_stmt.empty else 0,
            len(balance_sheet.columns) if not balance_sheet.empty else 0,
        )

        return RawFinancialData(
            ticker=ticker,
            exchange=exchange,
            yf_ticker=yf_ticker,
            info=info,
            fast_info=fast_info,
            income_stmt=income_stmt,
            balance_sheet=balance_sheet,
            cash_flow=cash_flow,
            quarterly_income_stmt=quarterly_income_stmt,
            quarterly_balance_sheet=quarterly_balance_sheet,
            quarterly_cash_flow=quarterly_cash_flow,
            dividends=dividends,
        )

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _build_yf_ticker(self, ticker: str, exchange: str) -> str:
        """
        Appends the correct yfinance exchange suffix.

        NSE → RELIANCE.NS
        BSE → RELIANCE.BO
        """
        exchange_upper = exchange.upper()
        if exchange_upper == "BSE":
            return f"{ticker}{self._settings.BSE_SUFFIX}"
        return f"{ticker}{self._settings.NSE_SUFFIX}"

    def _get_ticker_with_retry(self, yf_ticker: str, original_ticker: str) -> yf.Ticker:
        """
        Creates a yfinance Ticker object with retry on transient failures.

        Note: yf.Ticker() itself doesn't make a network call; the call happens
        when accessing properties. We intentionally do a lightweight property
        access here to trigger and validate connectivity.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                ticker_obj = yf.Ticker(yf_ticker, session=self._session)
                # Trigger a lightweight request to validate connectivity
                _ = ticker_obj.fast_info
                return ticker_obj
            except Exception as exc:
                last_exception = exc
                wait = self._BACKOFF_BASE ** attempt
                logger.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                    attempt,
                    self._MAX_RETRIES,
                    yf_ticker,
                    exc,
                    wait,
                )
                if attempt < self._MAX_RETRIES:
                    time.sleep(wait)

        raise NetworkError(
            f"Failed to connect to yfinance for '{original_ticker}' after "
            f"{self._MAX_RETRIES} attempts. Last error: {last_exception}",
            ticker=original_ticker,
        ) from last_exception

    def _fetch_dividends(self, ticker_obj: yf.Ticker, ticker: str) -> pd.Series:
        """
        Fetches dividend history from ticker.dividends.
        Falls back to the Dividends column in ticker.actions when the primary
        series is empty (some Indian tickers expose data only via actions).
        """
        dividends = self._safe_get_series(ticker_obj, "dividends", ticker)
        if dividends is not None and not dividends.empty:
            return dividends

        try:
            actions = ticker_obj.actions
            if actions is not None and not actions.empty and "Dividends" in actions.columns:
                div_col = actions["Dividends"].dropna()
                div_col = div_col[div_col > 0]
                if not div_col.empty:
                    logger.info("Using actions.Dividends fallback for %s (%d payments).", ticker, len(div_col))
                    return div_col
        except Exception as exc:
            logger.warning("Could not fetch dividend actions for %s: %s", ticker, exc)

        return pd.Series(dtype=float)

    def _extract_info(
        self,
        ticker_obj: yf.Ticker,
        original_ticker: str,
        yf_ticker: str,
    ) -> Dict[str, Any]:
        """
        Extracts the .info dict from a yfinance Ticker.

        If .info is empty or has no 'symbol' key, the ticker is considered
        invalid / not found on the exchange.
        """
        try:
            info = ticker_obj.info
        except Exception as exc:
            raise DataUnavailableError(
                f"Could not retrieve company info for '{original_ticker}': {exc}",
                ticker=original_ticker,
            ) from exc

        if not info or not info.get("symbol"):
            raise TickerNotFoundError(
                f"Ticker '{original_ticker}' was not found on yfinance "
                f"(tried '{yf_ticker}'). Verify the symbol is correct and "
                "listed on NSE or BSE.",
                ticker=original_ticker,
            )

        return info

    @staticmethod
    def _safe_get_df(
        ticker_obj: yf.Ticker,
        attribute: str,
        ticker: str,
    ) -> pd.DataFrame:
        """
        Safely retrieves a DataFrame attribute from a yfinance Ticker.
        Returns an empty DataFrame on any failure rather than raising, because
        many individual statements can be missing (e.g. no dividend history).
        The DataCleanerService will decide which missing statements are fatal.
        """
        try:
            df = getattr(ticker_obj, attribute)
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as exc:
            logger.warning(
                "Could not fetch '%s' for %s: %s. Will use empty DataFrame.",
                attribute,
                ticker,
                exc,
            )
            return pd.DataFrame()

    @staticmethod
    def _safe_get_series(
        ticker_obj: yf.Ticker,
        attribute: str,
        ticker: str,
    ) -> pd.Series:
        """Safely retrieves a Series attribute (e.g. dividends)."""
        try:
            series = getattr(ticker_obj, attribute)
            if series is None or (hasattr(series, "empty") and series.empty):
                return pd.Series(dtype=float)
            return series
        except Exception as exc:
            logger.warning(
                "Could not fetch '%s' for %s: %s. Will use empty Series.",
                attribute,
                ticker,
                exc,
            )
            return pd.Series(dtype=float)
