import io
from datetime import datetime


class TimestampedIO(io.StringIO):
    """StringIO that prefixes each non-blank write with a timestamp."""

    TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"

    def write(self, s: str) -> int:  # noqa: D401
        if s and s.strip():
            stamp = datetime.now().strftime(self.TIMESTAMP_FMT)
            s = f"[{stamp}] {s}"
        return super().write(s)
