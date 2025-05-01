"""
Celery application factory for background job processing.
"""
from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()
# import celery_worker.tasks as _ # noqa: E402 

# Default configuration (can be overridden by environment variables)
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "twitter_space_clipper",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["celery_worker.tasks"],  # <-- This line is key!
)

# Configure logging
import logging
logger = logging.getLogger('celery')
logger.setLevel(logging.INFO)

# Optional: Additional configuration
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=60 * 60 * 2,  # 2 hours max per task
    task_routes={
        'celery_worker.tasks.download_media': {'queue': 'download'},
        'celery_worker.tasks.transcribe_media': {'queue': 'transcribe'},
        'celery_worker.tasks.generate_quotes_task': {'queue': 'generate'},
        'celery_worker.tasks.generate_summary_task': {'queue': 'generate'},
    },
    task_publish_retry=True,
    task_publish_retry_policy={
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.5,
    },
)
