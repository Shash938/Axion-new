"""
security/payload_limit.py — Request Payload Size Limiting Middleware
====================================================================
Inspects incoming Content-Length headers and body size to prevent DoS attacks via
excessively large requests.
"""

import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

from config import get_settings

logger = logging.getLogger(__name__)


class PayloadSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware that checks Content-Length header to block payloads exceeding max_bytes.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._settings = get_settings()

    async def dispatch(self, request: Request, call_next) -> Response:
        content_length = request.headers.get("Content-Length")
        max_bytes = self._settings.MAX_PAYLOAD_SIZE_BYTES

        if content_length and request.method in ("POST", "PUT", "PATCH"):
            try:
                length = int(content_length)
                if length > max_bytes:
                    logger.warning(
                        "Payload size (%d bytes) exceeds maximum limit (%d bytes) from %s",
                        length,
                        max_bytes,
                        request.client.host if request.client else "unknown",
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "type": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/413",
                            "title": "Content Too Large",
                            "status": 413,
                            "detail": f"Request body size exceeds maximum allowed limit of {max_bytes} bytes.",
                        },
                    )
            except ValueError:
                pass

        return await call_next(request)
