"""
TradeMind Configuration Module

Environment-based configuration using Pydantic Settings.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Application
    app_name: str = "TradeMind"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    secret_key: str = "change-me-in-production"
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/trademind"
    database_sync_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/trademind"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 3600  # 1 hour
    
    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/1"
    
    # API Security
    api_key: str = "change-me-in-production"
    rate_limit_per_minute: int = 60
    
    # Data Ingestion Schedule (IST)
    data_ingestion_hour: int = 18  # 6:30 PM IST
    data_ingestion_minute: int = 30
    nifty500_symbols_file: str = "data/nifty500_list.csv"
    
    # ML Model
    model_path: str = "models/"
    current_model_version: str = "v1.0.0"
    signal_confidence_threshold: float = 0.6
    
    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
