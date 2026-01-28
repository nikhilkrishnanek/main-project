"""
Radar Performance Metrics and Theoretical Benchmarking
======================================================

This module provides the analytical tools and statistical estimators required 
to evaluate the radar system's operational effectiveness. It covers 
resolution limits, detection statistics, and AI model certainty.

Core Metrics:
-------------
1. Range/Velocity Resolution: Dictated by the waveform bandwidth and 
   coherent integration time (CPI).
2. Signal-to-Noise Ratio (SNR): Fundamental measure of signal quality.
3. Probability of Detection (Pd): Modeled using Swerling I fluctuation profiles.
4. Probability of False Alarm (Pfa): Derived from the CFAR threshold level.

Author: Lead Radar Systems Architect
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class ResolutionMetrics:
    """Container for theoretical radar system performance limits."""
    range_resolution_m: float
    velocity_resolution_ms: float
    maximum_unambiguous_range_m: float
    maximum_unambiguous_velocity_ms: float


@dataclass
class PerformanceStats:
    """Container for estimated real-time performance statistics."""
    snr_db: float
    probability_detection: float
    false_alarm_rate: float
    model_confidence: float
    model_entropy: float


def calculate_theoretical_resolutions(
    sweep_bandwidth_hz: float,
    chirp_duration_s: float,
    carrier_frequency_hz: float,
    num_pulses: int,
    sampling_rate_hz: float
) -> ResolutionMetrics:
    """
    Computes the theoretical resolution and ambiguity limits of the radar system.
    
    Equations:
    - Range Resolution (ΔR): c / (2 * B)
    - Velocity Resolution (Δv): λ / (2 * T_int), where T_int is the CPI.
    """
    speed_of_light = 3e8
    wavelength_m = speed_of_light / carrier_frequency_hz
    coherent_integration_time_s = num_pulses * chirp_duration_s
    
    delta_r = speed_of_light / (2 * sweep_bandwidth_hz)
    delta_v = wavelength_m / (2 * coherent_integration_time_s) if coherent_integration_time_s > 0 else 0.0
    
    # Ambiguity and Nyquist Limits
    sweep_slope_hz_s = sweep_bandwidth_hz / chirp_duration_s
    max_range_m = (sampling_rate_hz * speed_of_light) / (2 * sweep_slope_hz_s)
    # Max Velocity (Pulse-to-Pulse phase overlap limit)
    max_velocity_ms = wavelength_m / (4 * chirp_duration_s)
    
    return ResolutionMetrics(
        range_resolution_m=delta_r,
        velocity_resolution_ms=delta_v,
        maximum_unambiguous_range_m=max_range_m,
        maximum_unambiguous_velocity_ms=max_velocity_ms
    )


def calculate_snr_decibels(signal_power_linear: float, noise_power_linear: float) -> float:
    """
    Computes the Signal-to-Noise Ratio in decibels.
    """
    if noise_power_linear <= 0:
        return 100.0 # Upper bound proxy for near-perfect signals
    return 10 * np.log10(signal_power_linear / noise_power_linear)


def estimate_pd_swerling1(snr_db: float, pfa_target: float = 1e-6) -> float:
    """
    Estimates the Probability of Detection (Pd) for a Swerling I target.
    
    Swerling I models targets with many scattering centers of similar size, 
    typical for large aircraft or complex drones.
    Pd = Pfa ^ (1 / (1 + SNR_linear))
    """
    snr_linear = 10**(snr_db / 10.0)
    return float(pfa_target ** (1.0 / (1.0 + snr_linear)))


def estimate_false_alarm_rate(threshold_voltage: float, noise_variance: float) -> float:
    """
    Calculates the Probability of False Alarm (Pfa) for a Square-Law detector.
    
    Assumes a Rayleigh noise distribution in the quadrature samples.
    """
    if noise_variance <= 0:
        return 0.0
    
    exponent_arg = -(threshold_voltage**2) / (2 * noise_variance)
    return float(np.exp(exponent_arg))


def evaluate_ai_inference_confidence(class_probabilities: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    Analyzes the softmax output of the AI model to quantify classification certainty.
    
    Metrics:
    - Confidence: Maximum probability assigned to a single class.
    - Information Entropy: Measure of uncertainty across all possible classes.
    """
    probs = np.array(class_probabilities)
    if probs.sum() == 0:
        return {"confidence": 0.0, "entropy": 0.0}
        
    max_confidence = float(np.max(probs))
    
    # Shannon Entropy with stability epsilon
    shannon_entropy = -np.sum(probs * np.log(probs + 1e-12))
    
    return {
        "confidence": max_confidence,
        "entropy": float(shannon_entropy)
    }
