# 002-celery-tasks: Celery Integration Best Practices

## Rule Description

When adding or modifying Celery tasks in this codebase, follow these best practices to ensure reliable task registration, avoid circular imports, and guarantee smooth development and deployment.

---

## Requirements

- **Task Registration:**
  - Do **not** import the tasks module directly in your Celery app file to avoid circular imports.
  - Use the `include` argument in the `Celery` app constructor to register all task modules, e.g.:
    ```python
    celery_app = Celery(
        "my_app",
        broker=..., backend=...,
        include=["celery_worker.tasks"]
    )
    ```
- **Task Definition:**
  - Define all Celery tasks in a dedicated module (e.g., `celery_worker/tasks.py`).
  - Import the Celery app instance in the tasks module:
    ```python
    from celery_worker.celery_app import celery_app
    @celery_app.task
    def my_task(...): ...
    ```
- **Worker Startup:**
  - Always start the worker with the correct app path, e.g.:
    ```bash
    celery -A celery_worker.celery_app worker --loglevel=info
    ```
- **Testing Tasks:**
  - When testing, import tasks from the main tasks module, not from test scripts.
  - Run test scripts from the project root, or set `PYTHONPATH=.` if needed.
- **Project Structure:**
  - Ensure all relevant directories (`celery_worker/`, `tests/`) contain an `__init__.py` file to make them Python packages.

---

## Common Pitfalls & Solutions

- **Circular Import Error:**
  - Solution: Use `include` in the Celery app, do not import tasks at the top level of the app file.
- **Unregistered Task Error:**
  - Solution: Ensure the worker is started with the correct app and that tasks are included via the `include` argument.
- **ModuleNotFoundError:**
  - Solution: Run scripts from the project root and ensure all packages have `__init__.py` files.

---

## Example

```python
# celery_worker/celery_app.py
celery_app = Celery(
    "twitter_space_clipper",
    broker=..., backend=...,
    include=["celery_worker.tasks"],
)

# celery_worker/tasks.py
from celery_worker.celery_app import celery_app
@celery_app.task
def add(x, y):
    return x + y
```

---

> 💡 **Tip:** If you see errors about unregistered tasks or circular imports, check your app's `include` argument and your import paths first. 