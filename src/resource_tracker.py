import os
import threading
import time

import numpy as np
import psutil
import pynvml

pynvml.nvmlInit()

_proc = psutil.Process(os.getpid())


def get_cpu_memory():
    cpu = psutil.cpu_percent(interval=None)
    mem = _proc.memory_info().rss / 1024**2
    return cpu, mem


def get_gpu():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

    return util.gpu, mem.used / 1024**2


class ResourceTracker:
    def __init__(self, sample_interval: float = 0.25):
        self.sample_interval = sample_interval

        self.samples = []

    def __enter__(self):
        self.running = True
        self.start = time.perf_counter()

        self._sample_loop()

        return self

    def _sample_loop(self):
        def run():
            while self.running:
                cpu, mem = get_cpu_memory()
                gpu_util, gpu_mem = get_gpu()

                self.samples.append(
                    {
                        "cpu": cpu,
                        "ram": mem,
                        "gpu_util": gpu_util,
                        "gpu_mem": gpu_mem,
                    }
                )

                time.sleep(self.sample_interval)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.thread.join()

        self.end = time.perf_counter()

    def results(self):
        cpu = [s["cpu"] for s in self.samples]
        ram = [s["ram"] for s in self.samples]
        gpu_util = [s["gpu_util"] for s in self.samples if s["gpu_util"] is not None]
        gpu_mem = [s["gpu_mem"] for s in self.samples if s["gpu_mem"] is not None]

        stats = {
            "time": self.end - self.start,
            "cpu_mean": np.mean(cpu) if cpu else None,
            "cpu_peak": np.max(cpu) if cpu else None,
            "ram_mean": np.mean(ram) if ram else None,
            "ram_peak": np.max(ram) if ram else None,
            "gpu_util_mean": np.mean(gpu_util),
            "gpu_util_peak": np.max(gpu_util),
            "gpu_mem_mean": np.mean(gpu_mem),
            "gpu_mem_peak": np.max(gpu_mem),
        }

        return stats
