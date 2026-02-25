"""
Cache Service

Redis caching for signals and market data.
"""

import json
import logging
from datetime import date
from typing import Optional

import redis.asyncio as redis

from app.config import settings
from app.schemas.signal import SignalsListResponse
from app.schemas.market import MarketOverview

logger = logging.getLogger(__name__)

# Redis client
_redis_client: Optional[redis.Redis] = None


async def get_redis_client() -> redis.Redis:
    """Get or create Redis client."""
    global _redis_client
    
    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    
    return _redis_client


async def close_redis() -> None:
    """Close Redis connection."""
    global _redis_client
    
    if _redis_client:
        await _redis_client.close()
        _redis_client = None


def _signals_cache_key(
    signal_date: date,
    signal_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
) -> str:
    """Generate cache key for signals."""
    key = f"signals:{signal_date.isoformat()}"
    
    if signal_type:
        key += f":type:{signal_type}"
    if min_confidence:
        key += f":conf:{min_confidence}"
    
    return key


def _market_cache_key(market_date: date) -> str:
    """Generate cache key for market overview."""
    return f"market:{market_date.isoformat()}"


async def get_cached_signals(
    signal_date: date,
    signal_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
) -> Optional[SignalsListResponse]:
    """
    Get cached signals response.
    
    Args:
        signal_date: Date of signals
        signal_type: Optional filter by type
        min_confidence: Optional minimum confidence filter
        
    Returns:
        Cached response or None
    """
    try:
        client = await get_redis_client()
        key = _signals_cache_key(signal_date, signal_type, min_confidence)
        
        cached = await client.get(key)
        
        if cached:
            data = json.loads(cached)
            return SignalsListResponse(**data)
        
        return None
        
    except Exception as e:
        logger.warning(f"Cache get failed: {e}")
        return None


async def cache_signals(
    signal_date: date,
    signal_type: Optional[str],
    min_confidence: Optional[float],
    response: SignalsListResponse,
) -> None:
    """
    Cache signals response.
    
    Args:
        signal_date: Date of signals
        signal_type: Filter applied
        min_confidence: Confidence filter applied
        response: Response to cache
    """
    try:
        client = await get_redis_client()
        key = _signals_cache_key(signal_date, signal_type, min_confidence)
        
        # Convert to JSON-serializable dict
        data = response.model_dump(mode="json")
        
        await client.setex(
            key,
            settings.redis_cache_ttl,
            json.dumps(data),
        )
        
        logger.debug(f"Cached signals with key: {key}")
        
    except Exception as e:
        logger.warning(f"Cache set failed: {e}")


async def get_cached_market_overview(
    market_date: date,
) -> Optional[MarketOverview]:
    """
    Get cached market overview.
    
    Args:
        market_date: Date of market data
        
    Returns:
        Cached response or None
    """
    try:
        client = await get_redis_client()
        key = _market_cache_key(market_date)
        
        cached = await client.get(key)
        
        if cached:
            data = json.loads(cached)
            return MarketOverview(**data)
        
        return None
        
    except Exception as e:
        logger.warning(f"Cache get failed: {e}")
        return None


async def cache_market_overview(
    market_date: date,
    response: MarketOverview,
) -> None:
    """
    Cache market overview response.
    
    Args:
        market_date: Date of market data
        response: Response to cache
    """
    try:
        client = await get_redis_client()
        key = _market_cache_key(market_date)
        
        data = response.model_dump(mode="json")
        
        await client.setex(
            key,
            settings.redis_cache_ttl,
            json.dumps(data),
        )
        
        logger.debug(f"Cached market overview with key: {key}")
        
    except Exception as e:
        logger.warning(f"Cache set failed: {e}")


async def invalidate_cache(pattern: str = "*") -> int:
    """
    Invalidate cache entries matching pattern.
    
    Args:
        pattern: Redis key pattern to match
        
    Returns:
        Number of keys deleted
    """
    try:
        client = await get_redis_client()
        
        keys = []
        async for key in client.scan_iter(match=pattern):
            keys.append(key)
        
        if keys:
            deleted = await client.delete(*keys)
            logger.info(f"Invalidated {deleted} cache entries matching '{pattern}'")
            return deleted
        
        return 0
        
    except Exception as e:
        logger.warning(f"Cache invalidation failed: {e}")
        return 0
