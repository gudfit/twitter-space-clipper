import os

# Broker settings
broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Task settings
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
enable_utc = True

# Task execution settings
task_track_started = True
task_time_limit = 18000  # 5 hours max
task_soft_time_limit = 14400  # 4 hours soft limit

# Task routing
task_routes = {
    'celery_worker.tasks.download_media': {'queue': 'download'},
    'celery_worker.tasks.transcribe_media': {'queue': 'transcribe'},
    'celery_worker.tasks.generate_quotes_task': {'queue': 'generate'},
    'celery_worker.tasks.generate_summary_task': {'queue': 'generate'},
}

# Concurrency settings
worker_concurrency = 2  # Limit concurrent tasks due to GPU/memory usage

# State persistence
worker_state_db = 'celery_worker/worker_state.db'

# Logging
worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
worker_task_log_format = '[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s' 