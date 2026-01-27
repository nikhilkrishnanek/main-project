"""
Benchmarking Engine
===================

Provides systematic performance analysis for Photonic Radar research.
Generates metrics and curves to evaluate system robustness and AI reliability.

Key Metrics:
1. Pd vs SNR: Sensitivity analysis.
2. Pfa vs Threshold: Operational reliability.
3. AI Accuracy vs Noise: Robustness of Deep Learning layer.
4. Runtime vs System Load: Scalability and latency benchmarks.

Author: Principal Radar Scientist
"""

import time
import numpy as np
from typing import List, Dict, Tuple
from evaluation.metrics import calculate_pd_swerling1, calculate_far
from photonic.environment import Target, ChannelConfig
from photonic.noise import NoiseConfig
from photonic.signals import PhotonicConfig

def get_pd_curve(snr_range_db: np.ndarray, pfa: float = 1e-6) -> np.ndarray:
    """Calculates Pd over an SNR sweep."""
    return np.array([calculate_pd_swerling1(snr, pfa) for snr in snr_range_db])

def get_pfa_curve(threshold_range_db: np.ndarray, noise_floor_db: float = -50.0) -> np.ndarray:
    """Calculates Pfa over a threshold sweep relative to noise floor."""
    # Convert threshold relative to noise (Power Ratio)
    # Pfa = exp(-ThresholdLinear / NoiseLinear)
    curves = []
    for t in threshold_range_db:
        # threshold_level in calculate_far seems to expect a voltage/power ratio if it uses np.exp(-t^2/2)
        # Let's align with src.dsp.performance_metrics.calculate_far logic
        # which is: np.exp(-(threshold_level**2) / (2 * noise_variance))
        
        # We'll treat threshold_range_db as dB above noise floor
        ratio_linear = 10**(t / 10.0)
        # For a Rayleigh envelope, P_detected_power follows exp(-T/NoisePower)
        # calculate_far uses np.exp(-t^2 / 2s^2). If t is amplitude and s is std dev.
        # Let's use a simpler power-based Pfa for consistency if needed, 
        # but let's stick to the module's implementation.
        # Assuming threshold_level in calculate_far is amplitude-like.
        
        pfa = calculate_far(threshold_level=np.sqrt(2 * ratio_linear), noise_variance=1.0)
        curves.append(pfa)
    return np.array(curves)

def get_ai_accuracy_benchmark(pipeline, snr_range_db: np.ndarray, n_trials: int = 5) -> np.ndarray:
    """
    Evaluates AI classification accuracy under varying noise conditions.
    Simulates frames and checks if the prediction matches the dominant target.
    """
    from photonic.environment import Target
    
    accuracies = []
    # Fixed target for consistency
    test_target = Target(200.0, 50.0, 10.0, "Drone")
    p_cfg = PhotonicConfig(fs=10e9, f_start=10e9, bandwidth=2e9, duration=10e-6)
    n_cfg = NoiseConfig(rin_db_hz=-150)
    
    for snr in snr_range_db:
        correct = 0
        # Adjust channel noise level to achieve desired SNR approximately
        # SNR ~ TargetRCS - NoiseLevel? (highly simplified)
        c_cfg = ChannelConfig(carrier_freq=10e9, noise_level_db=10.0 - snr) 
        
        for _ in range(n_trials):
            # Run one-off simulation frame
            frame = pipeline.run(p_cfg, c_cfg, n_cfg, [test_target])
            if frame.prediction.predicted_class == "Drone":
                correct += 1
        
        accuracies.append(correct / n_trials)
        
    return np.array(accuracies)

def get_latency_benchmark(pipeline, complexity_factors: List[int]) -> List[float]:
    """
    Measures processing time (ms) as a function of 'Complexity'.
    Complexity here mapped to number of chirps or signal length.
    """
    latencies = []
    p_cfg = PhotonicConfig(fs=10e9, f_start=10e9, bandwidth=2e9, duration=10e-6)
    c_cfg = ChannelConfig()
    n_cfg = NoiseConfig()
    targets = [Target(100.0, 10.0, 10.0, "Drone")]
    
    for factor in complexity_factors:
        # Fake complexity by increasing bandwidth or duration (simulating data volume)
        p_cfg.duration = factor * 10e-6
        
        start = time.perf_counter()
        pipeline.run(p_cfg, c_cfg, n_cfg, targets)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000) # ms
        
    return latencies
