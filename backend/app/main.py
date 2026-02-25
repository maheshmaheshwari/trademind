"""
TradeMind FastAPI Application

Main application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import init_db, close_db
from app.services.cache import close_redis
from app.api.routes import signals, market, stocks, health
from app.api.middleware.auth import AuthenticationMiddleware
from app.api.middleware.rate_limit import RateLimitMiddleware

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info(f"Starting {settings.app_name} in {settings.app_env} mode")
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await close_db()
    await close_redis()
    logger.info("Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="""
    AI-powered stock analysis platform for Indian equity markets (NIFTY 500).
    
    Generates swing and positional trade signals using machine learning.
    
    **Disclaimer**: This is not financial advice. Always do your own research.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(RateLimitMiddleware)
# Note: Auth middleware handled via dependencies for more granular control

# Include routers
app.include_router(signals.router)
app.include_router(market.router)
app.include_router(stocks.router)
app.include_router(health.router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "description": "AI-powered stock analysis for Indian markets",
        "docs": "/docs" if settings.is_development else None,
        "disclaimer": "This is not financial advice. Always do your own research.",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "code": "INTERNAL_ERROR",
            "disclaimer": "Please try again later.",
        },
    )


# Health check for load balancers (no auth required)
@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}
