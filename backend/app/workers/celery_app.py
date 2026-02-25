"""
Celery Application

Background task queue configuration.
"""

from celery import Celery
from celery.schedules import crontab

from app.config import settings

# Create Celery app
celery_app = Celery(
    "trademind",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Timezone
    timezone="Asia/Kolkata",
    enable_utc=True,
    
    # Task settings
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3300,  # Soft limit at 55 mins
    
    # Result settings
    result_expires=86400,  # 24 hours
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
    
    # Scheduled tasks (Beat)
    beat_schedule={
        # Daily data ingestion at 6:30 PM IST (after market close)
        "ingest-daily-data": {
            "task": "app.workers.tasks.ingest_daily_data",
            "schedule": crontab(
                hour=settings.data_ingestion_hour,
                minute=settings.data_ingestion_minute,
            ),
        },
        # Compute indicators at 7:00 PM IST
        "compute-daily-indicators": {
            "task": "app.workers.tasks.compute_daily_indicators",
            "schedule": crontab(hour=19, minute=0),
        },
        # Run inference at 7:30 PM IST
        "run-daily-inference": {
            "task": "app.workers.tasks.run_daily_inference",
            "schedule": crontab(hour=19, minute=30),
        },
        # Generate signals at 8:00 PM IST
        "generate-daily-signals": {
            "task": "app.workers.tasks.generate_daily_signals",
            "schedule": crontab(hour=20, minute=0),
        },
        # Weekly model retraining on Sunday at 10:00 AM IST
        "weekly-model-training": {
            "task": "app.workers.tasks.train_model",
            "schedule": crontab(day_of_week=0, hour=10, minute=0),
        },
    },
)
