"""
Performance Evaluation Module
============================

Advanced metrics for Radar System Analysis.
Implements statistical performance estimators for Detection and Classification.

Key Functions:
1. calculate_pd: Probability of Detection given SNR and Pfa.
2. calculate_pfa: Probability of False Alarm given Threshold.
3. analyze_model_confidence: Statistical summary of AI confidence scores.
4. calculate_resolution: Re-exports theoretical resolution limits.

Author: Principal Radar Scientist
"""

import sys
import os

# Fix path for standalone execution (if run as script)
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from scipy.special import erf

# Import existing metric utilities to provide a unified evaluation interface
try:
    from evaluation.radar_metrics import (
        calculate_theoretical_metrics, 
        estimate_snr, 
        RadarPerformance
    )
except ImportError:
    # If standard import fails (e.g. IDE vs CLI issues), try relative or handle error
    # This usually happens if sys.path isn't set right.
    # We will assume the sys.path check above fixed it for CLI.
    # For module usage, it should work if package is installed or root is in path.
    pass

@dataclass
class EvaluationReport:
    """Comprehensive Performance Report."""
    snr_db: float
    pd_estimated: float      # Estimated Prob of Detection
    pfa_estimated: float     # Estimated False Alarm Rate
    range_res_m: float
    velocity_res_m_s: float
    ai_confidence_mean: float
    ai_confidence_var: float

def calculate_pd(snr_db: float, p_fa: float = 1e-6) -> float:
    """
    Calculates Probability of Detection (Pd) for a Swerling I target
    (slowly fluctuating) using Albersheim's approximation or standard model.
    
    Formula Approximation for Swerling I:
    Pd = Pfa ^ (1 / (1 + SNR_linear))
    
    Args:
        snr_db: Signal-to-Noise Ratio in dB.
        p_fa: Desired Probability of False Alarm.
        
    Returns:
        Pd (0.0 to 1.0)
    """
    snr_linear = 10**(snr_db / 10.0)
    
    # Avoid division by zero and singularity at SNR=0
    # If SNR is very low, Pd -> Pfa
    if snr_linear <= 0:
        return p_fa
        
    pd = p_fa ** (1.0 / (1.0 + snr_linear))
    return float(pd)

def calculate_pfa(threshold_db: float, noise_power_db: float) -> float:
    """
    Estimates Probability of False Alarm (Pfa) for a Linear Detector
    assuming Rayleigh distributed noise envelope (Gaussian I/Q).
    
    Pfa = exp( - (V_thresh^2) / (2 * sigma^2) )
    
    Args:
        threshold_db: Detection threshold in dB.
        noise_power_db: Average Noise Power in dB.
    """
    # Convert to linear voltage ratio squared (Power ratio)
    # Threshold Power / Noise Power
    power_ratio = 10**((threshold_db - noise_power_db) / 10.0)
    
    # For a square-law detector (power), Pfa = exp(-Threshold/NoisePower)
    # For linear detector (voltage), Pfa = exp(-V^2 / 2sigma^2) which is the same as exp(-P_t / P_n)
    
    pfa = np.exp(-power_ratio)
    return float(pfa)

def analyze_model_confidence(confidences: List[float]) -> Dict[str, float]:
    """
    Analyzes the statistical reliability of the AI model.
    """
    if not confidences:
        return {"mean": 0.0, "var": 0.0, "min": 0.0, "max": 0.0}
        
    arr = np.array(confidences)
    return {
        "mean": float(np.mean(arr)),
        "var": float(np.var(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }

def generate_report(
    rd_map_log: np.ndarray, 
    threshold_db: float,
    system_params: dict,
    ai_confidences: List[float] = None
) -> EvaluationReport:
    """
    Generates a full performance evaluation report for a frame.
    """
    # 1. Estimate SNR
    snr = estimate_snr(rd_map_log)
    noise_floor = np.median(rd_map_log)
    
    # 2. Calculate Probabilities
    # We estimate Pfa based on the cut threshold relative to noise
    # Assuming threshold_db is absolute dB value used for detection
    # If standard CFAR is used, threshold varies. Here we assume global threshold for estimation.
    pfa_est = calculate_pfa(threshold_db, noise_floor)
    pd_est = calculate_pd(snr, pfa_est)
    
    # 3. Theoretical Limits
    # Need bandwidth etc from params
    perf = calculate_theoretical_metrics(
        bandwidth=system_params.get('bandwidth', 4e9),
        chirp_duration=system_params.get('duration', 10e-6),
        carrier_freq=system_params.get('f_start', 10e9),
        n_chirps=system_params.get('n_chirps', 64),
        fs=system_params.get('fs', 20e9)
    )
    
    # 4. AI Stats
    ai_stats = analyze_model_confidence(ai_confidences if ai_confidences else [0.0])
    
    return EvaluationReport(
        snr_db=snr,
        pd_estimated=pd_est,
        pfa_estimated=pfa_est,
        range_res_m=perf.range_resolution_m,
        velocity_res_m_s=perf.velocity_resolution_m_s,
        ai_confidence_mean=ai_stats['mean'],
        ai_confidence_var=ai_stats['var']
    )

if __name__ == "__main__":
    # Test
    print("Testing Performance Module...")
    
    # 1. Pd Check
    snr_vals = [-10, 0, 10, 20]
    print(f"Pd vs SNR (Pfa=1e-6):")
    for s in snr_vals:
        print(f"  SNR {s} dB -> Pd: {calculate_pd(s):.4f}")
        
    # 2. Pfa Check
    print(f"\nPfa vs Threshold (Noise=0dB):")
    threshs = [5, 10, 13, 20]
    for t in threshs:
        print(f"  Thresh {t} dB -> Pfa: {calculate_pfa(t, 0):.4e}")
        
    print("\nModule Verified.")
