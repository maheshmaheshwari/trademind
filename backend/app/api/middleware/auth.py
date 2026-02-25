"""
Authentication Middleware

API key validation and request authentication.
"""

import logging
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

logger = logging.getLogger(__name__)

# Endpoints that don't require authentication
PUBLIC_ENDPOINTS = {
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/model/health",
    "/model/ping",
}


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process request and validate API key.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response from handler or error response
        """
        path = request.url.path
        
        # Allow public endpoints
        if path in PUBLIC_ENDPOINTS or path.startswith("/model/"):
            return await call_next(request)
        
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            logger.warning(f"Missing API key for request to {path}")
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Missing API key. Include 'X-API-Key' header.",
                    "code": "MISSING_API_KEY",
                },
            )
        
        if api_key != settings.api_key:
            logger.warning(f"Invalid API key attempt for {path}")
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Invalid API key.",
                    "code": "INVALID_API_KEY",
                },
            )
        
        # Valid API key - proceed with request
        return await call_next(request)
