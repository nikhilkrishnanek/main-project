"""
System Pipeline Orchestrator (Metrics Integrated)
================================================

Connects the Simulation, DSP, and AI layers into a coherent pipeline.
Now includes advanced performance evaluation.

Author: Principal Systems Architect
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from photonic.signals import generate_photonic_signal, PhotonicConfig
from photonic.environment import simulate_target_response, Target, ChannelConfig
from photonic.noise import add_rin_noise, apply_fiber_dispersion, add_thermal_noise, NoiseConfig
from signal.transforms import (
    dechirp_signal, 
    compute_range_doppler_map, 
    compute_spectrogram, 
    extract_features
)
from evaluation.metrics import (
    calculate_resolutions, 
    calculate_snr_db, 
    calculate_pd_swerling1, 
    calculate_far, 
    evaluate_model_confidence,
    ResolutionMetrics,
    PerformanceStats
)
from ai.model import ClassifierPipeline, Prediction

@dataclass
class RadarFrame:
    """Container for a single radar processing frame."""
    time_axis: np.ndarray
    rx_signal: np.ndarray
    rd_map: np.ndarray
    spectrogram: np.ndarray
    prediction: Prediction
    metrics: dict
    performance: ResolutionMetrics
    stats: PerformanceStats

class RadarPipeline:
    def __init__(self):
        self.ai = ClassifierPipeline() # Persistent AI model
        
    def run(self, 
            p_cfg: PhotonicConfig, 
            c_cfg: ChannelConfig, 
            n_cfg: NoiseConfig,
            targets: List[Target]
            ) -> RadarFrame:
        """
        Executes one full frame of the radar loop.
        """
        # --- 1. Simulation Layer ---
        t, tx_sig = generate_photonic_signal(p_cfg)
        
        # Apply Channel (Echoes + Noise)
        rx_sig = simulate_target_response(tx_sig, p_cfg.fs, targets, c_cfg)
        
        # Apply Advanced Noise (Optional Layer)
        rin = add_rin_noise(10**(p_cfg.optical_power_dbm/10 - 3), n_cfg, len(tx_sig), p_cfg.fs)
        rx_sig += rin 
        rx_sig = apply_fiber_dispersion(rx_sig, n_cfg, p_cfg.fs)
        rx_sig = add_thermal_noise(rx_sig, n_cfg, p_cfg.fs)
        
        # --- Future Hardware Hook ---
        # if st.session_state.get('use_hardware'):
        #     from interfaces.hardware import SDRInterface
        #     rx_sig = sdr_interface.stream_iq(len(tx_sig))
        
        # --- 2. DSP Layer ---
        if_sig = dechirp_signal(rx_sig, tx_sig)
        
        samples_per_chirp = 64
        n_chirps = 64
        
        rd_map = compute_range_doppler_map(
            if_sig, n_chirps=n_chirps, samples_per_chirp=samples_per_chirp
        )
        spec = compute_spectrogram(if_sig, p_cfg.fs)
        
        # --- Metrics Calculation ---
        # 1. Resolutions
        perf = calculate_resolutions(
            p_cfg.bandwidth, p_cfg.duration, c_cfg.carrier_freq, n_chirps, p_cfg.fs
        )
        
        # 2. SNR
        # Estimate signal peak and noise floor from RD Map
        peak_pow = np.max(rd_map)
        noise_pow = np.median(rd_map)
        snr_est = peak_pow - noise_pow  # Log scale subtraction
        
        # 3. Detection Stats (Pd, FAR)
        # Using a fixed reference Pfa or threshold for estimation
        pd_val = calculate_pd_swerling1(snr_est, pfa=1e-6)
        
        # Estimate FAR for a nominal threshold (e.g. 13dB above noise)
        # This is strictly theoretical based on current noise floor
        far_val = calculate_far(threshold_level=13.0, noise_variance=1.0) # Normalized
        
        # --- 3. AI Layer ---
        pred = self.ai.predict(rd_map, spec)
        
        # 4. Confidence Stats
        conf_stats = evaluate_model_confidence(list(pred.probabilities.values()))
        
        # Feature Extraction
        feats = extract_features(if_sig)
        
        # Compile Performance Stats
        stats = PerformanceStats(
            snr_db=snr_est,
            probability_detection=pd_val,
            false_alarm_rate=far_val,
            model_confidence=conf_stats["confidence"],
            model_entropy=conf_stats["entropy"]
        )
        
        # Generic DSP features
        metrics_dict = {
            "snr_db": snr_est,
            **feats
        }
        
        # --- 4. Package ---
        return RadarFrame(
            time_axis=t,
            rx_signal=rx_sig,
            rd_map=rd_map,
            spectrogram=spec,
            prediction=pred,
            metrics=metrics_dict,
            performance=perf,
            stats=stats
        )
