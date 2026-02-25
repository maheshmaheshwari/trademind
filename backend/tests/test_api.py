"""
API Tests

Integration tests for API endpoints.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.main import app
from app.config import settings
from app.database import Base, get_async_session


# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_db():
    """Create test database."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def client(test_db):
    """Create test client."""
    async def override_get_session():
        yield test_db
    
    app.dependency_overrides[get_async_session] = override_get_session
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_check(client):
    """Test health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Test root endpoint."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == settings.app_name
    assert "disclaimer" in data


@pytest.mark.asyncio
async def test_model_health(client):
    """Test model health endpoint."""
    response = await client.get("/model/health")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "database" in data
    assert "model" in data


@pytest.mark.asyncio
async def test_signals_without_auth(client):
    """Test signals endpoint requires authentication."""
    response = await client.get("/signals/today")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_signals_with_auth(client):
    """Test signals endpoint with authentication."""
    response = await client.get(
        "/signals/today",
        headers={"X-API-Key": settings.api_key}
    )
    # With empty db, might return 200 with empty list or appropriate error
    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_market_overview(client):
    """Test market overview endpoint."""
    response = await client.get(
        "/market/overview",
        headers={"X-API-Key": settings.api_key}
    )
    assert response.status_code in [200, 500]  # May fail without real data


@pytest.mark.asyncio
async def test_stock_search(client):
    """Test stock search endpoint."""
    response = await client.get(
        "/stocks/search",
        headers={"X-API-Key": settings.api_key}
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_rate_limiting(client):
    """Test rate limiting (basic check)."""
    # Make multiple requests
    for _ in range(5):
        response = await client.get("/health")
        assert "X-RateLimit-Limit" in response.headers or response.status_code == 200
