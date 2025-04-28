import os
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL")

if REDIS_URL is None:
    print("❌ No REDIS_URL or CELERY_BROKER_URL found in environment.")
    exit(1)

# Convert redis:// URL to redis-py format if needed (strip trailing /0 for db=0)
if REDIS_URL.endswith("/0"):
    REDIS_URL = REDIS_URL[:-2]

try:
    r = redis.from_url(REDIS_URL)
    test_key = "test:connection"
    test_value = "success"
    r.set(test_key, test_value, ex=10)  # Set with 10s expiry
    value = r.get(test_key)
    if value and value.decode() == test_value:
        print("✅ Redis connection test succeeded.")
    else:
        print("❌ Redis connection test failed: Value mismatch.")
except Exception as e:
    print(f"❌ Redis connection test failed: {e}") 