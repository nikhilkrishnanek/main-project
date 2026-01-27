"""
Performance Metrics Module
=========================

Core functions for Photonic Radar performance analysis.
Provides physically theoretical benchmarks and statistical estimators.

Metrics:
1. Signal-to-Noise Ratio (SNR)
2. Probability of Detection (Pd)
3. False Alarm Rate (FAR)
4. Theoretical Resolutions (Range/Velocity)
5. AI Model Confidence

Author: Principal Radar Scientist
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union

@dataclass
class ResolutionMetrics:
    range_resolution: float      # in meters
    velocity_resolution: float   # in m/s
    max_range: float            # in meters
    max_velocity: float         # in m/s

@dataclass
class PerformanceStats:
    """Container for estimated performance statistics."""
    snr_db: float
    probability_detection: float
    false_alarm_rate: float
    model_confidence: float
    model_entropy: float

def calculate_resolutions(
    bandwidth: float,
    chirp_duration: float,
    fc: float,
    n_chirps: int,
    fs: float
) -> ResolutionMetrics:
    """
    Calculates theoretical radar resolution limits.
    
    Delta R = c / (2 * B)
    Delta v = lambda / (2 * T_integration)
    """
    c = 3e8
    wavelength = c / fc
    t_int = n_chirps * chirp_duration
    
    delta_r = c / (2 * bandwidth)
    delta_v = wavelength / (2 * t_int) if t_int > 0 else 0.0
    
    # Nyquist Limits
    slope = bandwidth / chirp_duration
    max_r = fs * c / (2 * slope)
    max_v = wavelength / (4 * chirp_duration)
    
    return ResolutionMetrics(delta_r, delta_v, max_r, max_v)

def calculate_snr_db(signal_power: float, noise_power: float) -> float:
    """
    Calculates SNR in Decibels.
    """
    if noise_power <= 0:
        return 999.0
    return 10 * np.log10(signal_power / noise_power)

def calculate_pd_swerling1(snr_db: float, pfa: float = 1e-6) -> float:
    """
    Probability of Detection for Swerling I Target (Slow Fluctuation).
    Albersheim's Approximation logic (Simplified).
    
    Pd = Pfa ^ (1 / (1 + SNR_lin))
    """
    snr_lin = 10**(snr_db / 10.0)
    return float(pfa ** (1.0 / (1.0 + snr_lin)))

def calculate_far(threshold_level: float, noise_variance: float) -> float:
    """
    Estimates False Alarm Rate (Probability) for linear detector.
    Pfa = exp( - Threshold^2 / (2 * NoiseVar) )
    Assumes Rayleigh noise distribution.
    """
    if noise_variance <= 0:
        return 0.0
    
    # Argument of exp
    arg = -(threshold_level**2) / (2 * noise_variance)
    return float(np.exp(arg))

def evaluate_model_confidence(
    probs: Union[List[float], np.ndarray]
) -> Dict[str, float]:
    """
    Evaluates the certainty of the AI model.
    Returns Mean, Max, and Entropy of the probability vector.
    """
    p = np.array(probs)
    if p.sum() == 0:
        return {"confidence": 0.0, "entropy": 0.0}
        
    confidence = float(np.max(p))
    
    # Entropy = -Sum(p * log(p))
    # Add epsilon
    entropy = -np.sum(p * np.log(p + 1e-9))
    
    return {
        "confidence": confidence,
        "entropy": float(entropy)
    }
