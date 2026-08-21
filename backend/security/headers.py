"""
security/headers.py — Security Headers ASGI Middleware
========================================================
Injects OWASP-recommended security headers into all HTTP responses.
"""

import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from fastapi import Request, Response

from config import get_settings

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds OWASP standard security headers to every response.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._settings = get_settings()

    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)

        if self._settings.ENABLE_SECURITY_HEADERS:
            # Prevent MIME sniffing
            response.headers["X-Content-Type-Options"] = "nosniff"

            # Prevent Clickjacking framing
            response.headers["X-Frame-Options"] = "DENY"

            # Cross-Site Scripting protection
            response.headers["X-XSS-Protection"] = "1; mode=block"

            # Strict Referrer Policy
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

            # Content Security Policy (CSP)
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' http: https:;"
            )

            # Restrict browser APIs
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

            # HTTP Strict Transport Security (HSTS) if connection is HTTPS or behind proxy
            if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        return response
