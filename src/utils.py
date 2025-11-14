import time
from collections.abc import Callable
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any

from distributed.client import as_completed


def track_time(f) -> tuple[float, Any]:
    start = time.time()
    result = f()
    end = time.time()
    diff = end - start
    return diff, result


def pool(funcs: list[Callable], workers) -> list[Any]:
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(f) for f in funcs]
        for future in as_completed(futures):
            results.append(future.result())

    return results
