"""
Rate Limiting Middleware

Token bucket rate limiting per API key.
"""

import logging
import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.
        
        Args:
            rate: Tokens added per second
            capacity: Maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens consumed, False if rate limited
        """
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        
        # Add tokens based on elapsed time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    @property
    def remaining(self) -> int:
        """Get remaining tokens."""
        return int(self.tokens)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""
    
    def __init__(self, app, rate_per_minute: int = None):
        """
        Initialize rate limiter.
        
        Args:
            app: FastAPI application
            rate_per_minute: Requests allowed per minute per key
        """
        super().__init__(app)
        self.rate_per_minute = rate_per_minute or settings.rate_limit_per_minute
        self.rate_per_second = self.rate_per_minute / 60
        self.buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(self.rate_per_second, self.rate_per_minute)
        )
    
    def _get_client_key(self, request: Request) -> str:
        """Get unique client identifier."""
        api_key = request.headers.get("X-API-Key", "")
        client_ip = request.client.host if request.client else "unknown"
        return f"{api_key}:{client_ip}"
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process request and apply rate limiting.
        """
        # Skip rate limiting for health checks
        if request.url.path.startswith("/model/"):
            return await call_next(request)
        
        client_key = self._get_client_key(request)
        bucket = self.buckets[client_key]
        
        if not bucket.consume():
            logger.warning(f"Rate limit exceeded for {client_key}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please slow down.",
                    "code": "RATE_LIMIT_EXCEEDED",
                    "retry_after_seconds": 60,
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.rate_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.rate_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(bucket.remaining)
        
        return response
