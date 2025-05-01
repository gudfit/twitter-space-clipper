---
description: Redis Key Namespacing Best Practices
globs: **/*.py
alwaysApply: false
---
# Redis Key Namespacing Best Practices

## Context
- When using Redis across multiple environments (dev, prod, testing)
- When multiple instances share the same Redis server
- When managing Celery task states and metadata

## Requirements
- ALWAYS namespace Redis keys with the hostname when sharing Redis instances
- ALWAYS use consistent key format: `{hostname}:{key_type}:{key_name}`
- ALWAYS use the same namespacing method across all Redis operations
- ALWAYS document Redis key patterns in comments
- ALWAYS clean up only keys belonging to the current host

## Examples

### Good: Namespaced Celery Task Keys
```python
import socket

hostname = socket.gethostname()
task_id = f"{hostname}:task:{uuid.uuid4()}"
redis_key = f"celery-task-meta-{hostname}:{task_id}"
```

### Good: Cleaning Up Host-Specific Keys
```python
# Only clean keys for current host
pattern = f"{hostname}:*"
keys = redis.keys(pattern)
if keys:
    redis.delete(*keys)
```

### Bad: Non-Namespaced Keys
```python
# DON'T do this - no hostname namespace
task_id = str(uuid.uuid4())
redis_key = f"celery-task-meta-{task_id}"
```

## Explanation
When multiple environments or instances share the same Redis server, key namespacing prevents:
- Key collisions between environments
- Accidental deletion of other environment's data
- Confusion about key ownership
- Issues with concurrent testing and production use 