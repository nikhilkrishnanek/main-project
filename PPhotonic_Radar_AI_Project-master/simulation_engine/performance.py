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
        Returns average metrics over the window and history for trends.
        """
        metrics = {}
        history = {}
        
        for phase, values in self.latencies.items():
            avg_lat = round(sum(values) / len(values), 2)
            metrics[f"{phase}_ms"] = avg_lat
            history[phase] = list(values)
            
        # Calculate FPS based on total_ms
        if "total_ms" in metrics and metrics["total_ms"] > 0:
            metrics["effective_fps"] = round(1000 / metrics["total_ms"], 1)
            
        return {
            "averages": metrics,
            "history": history
        }

    def check_bottlenecks(self, threshold_ms: float = 80.0) -> list[str]:
        """
        Identify phases exceeding the real-time budget.
        """
        bottlenecks = []
        metrics = self.get_metrics()["averages"]
        for phase, lat in metrics.items():
            if phase != "total_ms" and lat > threshold_ms:
                bottlenecks.append(f"CRITICAL: {phase} delayed ({lat}ms)")
        return bottlenecks
