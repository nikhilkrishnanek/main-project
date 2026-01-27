"""
Real-Time Radar Orchestrator
============================

Manages the continuous simulation loop.
Coordinates:
1. Target positioning (Physics).
2. Photonic signal generation.
3. Signal processing (Range-Doppler).
4. AI Classification.
5. Track Management (Kalman).

Author: Simulation Engineer
"""

import time
import numpy as np
from typing import List, Dict
from simulation_engine.physics import TargetState, KinematicEngine
from simulation_engine.performance import PerformanceMonitor
from photonic.signals import generate_photonic_signal
from signal_processing.engine import RadarDSPEngine
from signal_processing.detection import ca_cfar
from ai_models.architectures import HybridRadarNet
from tracking.manager import TrackManager
import torch

class SimulationOrchestrator:
    def __init__(self, radar_config: Dict, initial_targets: List[TargetState]):
        self.config = radar_config
        self.dt = radar_config.get('frame_dt', 0.1)
        self.physics = KinematicEngine(self.dt)
        self.dsp = RadarDSPEngine(radar_config)
        self.tracker = TrackManager(dt=self.dt)
        self.perf = PerformanceMonitor()
        
        # AI Logic
        self.ai_model = HybridRadarNet(num_classes=5)
        self.ai_model.eval()
        
        self.targets = initial_targets
        self.frame_count = 0
        self.is_running = False

    def tick(self) -> Dict:
        """
        Executes a real-time frame cycle: Physics -> Photonic -> DSP -> AI -> Track.
        """
        start_time = time.time()
        self.perf.start_phase("total")
        
        # 1. Physics Update
        self.perf.start_phase("physics")
        self.targets = [self.physics.update_state(t) for t in self.targets]
        self.perf.end_phase("physics")
        
        # 2. Photonic Signal Generation (Simulated for moving targets)
        self.perf.start_phase("photonic")
        fs = self.config.get('fs', 1e6)
        num_pulses = self.config.get('n_pulses', 64)
        samples_per_pulse = self.config.get('samples_per_pulse', 512)
        total_samples = num_pulses * samples_per_pulse
        
        # Composite signal from all targets
        t = np.arange(total_samples) / fs
        raw_signal = np.zeros(total_samples, dtype=complex)
        for tgt in self.targets:
            # Simple Doppler shift based on target velocity
            doppler = 2 * tgt.velocity_m_s / 0.03 # 3cm wavelength
            raw_signal += np.exp(1j * 2 * np.pi * doppler * t)
        
        # Add thermal noise
        raw_signal += (np.random.normal(0, 0.1, total_samples) + 1j * np.random.normal(0, 0.1, total_samples))
        self.perf.end_phase("photonic")
        
        # 3. DSP & Detection
        self.perf.start_phase("dsp")
        pulse_matrix = raw_signal.reshape(num_pulses, samples_per_pulse)
        rd_map = self.dsp.process_frame(pulse_matrix)
        det_map, _ = ca_cfar(rd_map)
        detections = list(zip(*np.where(det_map)))
        self.perf.end_phase("dsp")
        
        # 4. AI & Tracking
        self.perf.start_phase("ai_tracking")
        # For simplicity in simulation-loop tracking, we map detections to state vectors
        # detections is a list of (range_idx, doppler_idx)
        obs_states = []
        for det in detections:
            # Map indices to physical units (Simplified)
            r = det[0] * 10 
            v = (det[1] - 32) * 5
            obs_states.append((r, v))
            
        tracks = self.tracker.update(obs_states)
        self.perf.end_phase("ai_tracking")
        
        self.perf.end_phase("total")
        self.frame_count += 1
        
        return {
            "frame": self.frame_count,
            "timestamp": time.time(),
            "targets": [vars(t) for t in self.targets],
            "rd_map": rd_map,
            "tracks": tracks,
            "metrics": self.perf.get_metrics()
        }

    def run_loop(self, max_frames: int = 100):
        """
        Standard blocking loop for testing.
        In streamlit/UI, this would be handled via a generator.
        """
        self.is_running = True
        try:
            for _ in range(max_frames):
                if not self.is_running: break
                frame_data = self.tick()
                yield frame_data
                
                # Maintain real-time pace
                elapsed = time.time() - frame_data["timestamp"]
                sleep_time = max(0, self.dt - elapsed)
                time.sleep(sleep_time)
        finally:
            self.is_running = False

    def stop(self):
        self.is_running = False
