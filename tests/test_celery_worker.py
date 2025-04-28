from celery_worker.tasks import add

if __name__ == "__main__":
    print("Submitting test Celery task (add 2 + 3)...")
    result = add.delay(2, 3)
    try:
        value = result.get(timeout=10)
        if value == 5:
            print("✅ Celery worker test succeeded: 2 + 3 = 5")
        else:
            print(f"❌ Celery worker test failed: Unexpected result {value}")
    except Exception as e:
        print(f"❌ Celery worker test failed: {e}")