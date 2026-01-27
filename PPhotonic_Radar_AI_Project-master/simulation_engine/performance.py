"""
Performance and Latency Monitor
==============================

Tracks execution metrics for real-time validation.
Monitors:
- Latency per phase (ms).
- Total frame time.
- Effective FPS.

Author: Simulation Engineer
"""

import time
from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.latencies = {} # {phase: deque}
        self.start_times = {}

    def start_phase(self, phase_name: str):
        self.start_times[phase_name] = time.perf_counter()

    def end_phase(self, phase_name: str):
        if phase_name not in self.start_times:
            return
            
        latency = (time.perf_counter() - self.start_times[phase_name]) * 1000 # to ms
        if phase_name not in self.latencies:
            self.latencies[phase_name] = deque(maxlen=self.window_size)
            
        self.latencies[phase_name].append(latency)

    def get_metrics(self) -> dict:
        """
        Returns average metrics over the window.
        """
        metrics = {}
        for phase, values in self.latencies.items():
            metrics[f"{phase}_ms"] = round(sum(values) / len(values), 2)
            
        # Calculate FPS based on total_ms
        if "total_ms" in metrics and metrics["total_ms"] > 0:
            metrics["effective_fps"] = round(1000 / metrics["total_ms"], 1)
            
        return metrics
