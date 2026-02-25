"""
API Dependencies

Common dependencies for API routes.
"""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_async_session


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """
    Verify API key from request header.
    
    Args:
        x_api_key: API key from X-API-Key header
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


async def get_optional_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key")
) -> str | None:
    """
    Get optional API key (for public endpoints with enhanced features for authenticated users).
    """
    if x_api_key and x_api_key == settings.api_key:
        return x_api_key
    return None


# Type aliases for dependency injection
DbSession = Annotated[AsyncSession, Depends(get_async_session)]
ApiKey = Annotated[str, Depends(verify_api_key)]
OptionalApiKey = Annotated[str | None, Depends(get_optional_api_key)]
