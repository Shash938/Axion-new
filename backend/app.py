"""
app.py — FastAPI Application Entry Point
=========================================
Why this file exists:
    Creates and configures the FastAPI application instance. Responsible for:
      - Application metadata (name, version, docs URL)
      - Middleware (CORS, request logging)
      - Global exception handler
      - Lifespan events (startup / shutdown logging)
      - Health check endpoint
      - Router registration

How it connects:
    - Root of the application. Run with: `uvicorn app:app --reload`
    - Imports and registers `analysis_router` from routers/analysis.py.
    - Reads configuration via `get_settings()` from config.py.

Possible improvements:
    - Add Prometheus metrics middleware for production monitoring.
    - Add authentication/API-key middleware before routes go public.
    - Add Sentry SDK integration for error tracking.
    - Move to a factory function pattern (`create_app()`) for easier testing.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from config import get_settings
from database.db import init_db
from routers.analysis import analysis_router
from routers.auth import auth_router
from routers.history import history_router
from routers.webauthn import webauthn_router
from routers.face_auth import face_auth_router
from security.headers import SecurityHeadersMiddleware
from security.payload_limit import PayloadSizeLimitMiddleware

# ==============================================================================
# Logging Setup
# ==============================================================================

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Application Lifespan
# ==============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown lifecycle.

    On startup:
        - Logs configuration summary (safe fields only — never log secrets).
        - Future: warm up DB connections, load ML models, etc.

    On shutdown:
        - Logs graceful shutdown.
        - Future: flush async queues, close connection pools.
    """
    # === Startup ===
    init_db()
    logger.info("=" * 60)
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("Database initialized successfully.")
    logger.info("Debug mode : %s", settings.DEBUG)
    logger.info("Log level  : %s", settings.LOG_LEVEL)
    logger.info("Allowed origins: %s", settings.ALLOWED_ORIGINS)
    logger.info("=" * 60)
    yield
    # === Shutdown ===
    logger.info("Shutting down %s. Goodbye.", settings.APP_NAME)


# ==============================================================================
# Application Factory
# ==============================================================================


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI-powered fundamental analysis engine for NSE-listed Indian stocks. "
        "Calculates 14 financial ratios, scores each metric, and returns a "
        "structured investment recommendation with beginner-friendly explanations."
    ),
    docs_url="/docs" if settings.DEBUG else None,       # Hide Swagger in production
    redoc_url="/redoc" if settings.DEBUG else None,     # Hide ReDoc in production
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)


# ==============================================================================
# Middleware
# ==============================================================================


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(PayloadSizeLimitMiddleware)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Logs every incoming request and outgoing response with timing.
    Helps diagnose slow queries and unexpected traffic patterns.
    """
    start_time = time.perf_counter()
    logger.info("→ %s %s", request.method, request.url.path)

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "← %s %s | status=%d | %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ==============================================================================
# Global Exception Handler
# ==============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches any unhandled exception that escapes route handlers.
    Returns a safe RFC 7807 Problem Details response — never leaks
    stack traces or internal details to the client.
    """
    logger.exception(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "type": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An unexpected error occurred. Please try again later.",
        },
    )


# ==============================================================================
# Health Check
# ==============================================================================


@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    description="Returns application health status. Used by load balancers and monitoring systems.",
)
def health_check():
    """
    Lightweight health check endpoint.
    Returns 200 OK when the application is running.
    Does NOT check external dependencies (yfinance, DB) — use a readiness
    probe for that when those are added.
    """
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


# ==============================================================================
# UI Dashboard
# ==============================================================================


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    """Serves the frontend SPA dashboard."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "index.html"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")),
    ]
    for ui_path in possible_paths:
        if os.path.exists(ui_path):
            with open(ui_path, "r", encoding="utf-8") as f:
                return f.read()
    return "<h1>UI dashboard file not found</h1>"


# ==============================================================================
# Router Registration
# ==============================================================================

app.include_router(analysis_router)
app.include_router(auth_router)
app.include_router(history_router)
app.include_router(webauthn_router)
app.include_router(face_auth_router)


# ==============================================================================
# Dev Server Entry Point
# ==============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
