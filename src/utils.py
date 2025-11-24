import time
from typing import Any


def track_time(f) -> tuple[float, Any]:
    start = time.time()
    result = f()
    end = time.time()
    diff = end - start
    return diff, result
