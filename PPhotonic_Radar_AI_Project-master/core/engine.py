"""
Strategic Radar Orchestration and Cognitive Feedback Loop
==========================================================

This module implements the primary execution pipeline for the photonic 
radar system. It orchestrates the flow of data across the six architectural 
layers: Simulation, Photonic, DSP, AI, Tracking, and Cognitive Feedback.

The pipeline is "cognitive" in that it utilizes AI-driven assessments 
to adaptively reconfigure hardware parameters (bandwidth, power, thresholds) 
for the subsequent processing frame.

Key Orchestration Stages:
-------------------------
1. Adaptive Reconfiguration: Applies feedback from frame N-1 to frame N.
2. Signal Synthesis: Generates high-fidelity heterodyne photonic RF signals.
3. DSP Pipeline: Executes de-chirping, spectral mapping, and CA-CFAR detection.
4. Intelligence & Tracking: Performs GNN-based tracking and multimodal AI classification.
5. Metrics & QA: Benchmarks resolution, SNR, and model confidence.

Author: Lead Systems Architect (Advanced Radar Systems)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from photonic.signals import generate_synthetic_photonic_signal, PhotonicConfig
from photonic.environment import simulate_target_response, Target, ChannelConfig
from photonic.noise import add_rin_noise, apply_fiber_dispersion, add_thermal_noise, NoiseConfig
from signal_processing.transforms import (
    dechirp_signal, 
    compute_range_doppler_map, 
    compute_spectrogram, 
    extract_statistical_features
)
from evaluation.metrics import (
    calculate_theoretical_resolutions, 
    calculate_snr_decibels, 
    estimate_pd_swerling1, 
    estimate_false_alarm_rate, 
    evaluate_ai_inference_confidence,
    ResolutionMetrics,
    PerformanceStats
)
from ai_models.model import IntelligencePipeline, IntelligenceOutput
from cognitive.engine import CognitiveRadarEngine, create_track_dict_for_cognitive
from tracking.manager import TacticalTrackManager
from signal_processing.detection import execute_detection_pipeline


@dataclass
class TacticalIntelligenceFrame:
    """
    Standardized telemetry container for a single processing cycle.
    """
    frame_id: int
    time_velocity_axis: np.ndarray
    raw_rx_signal: np.ndarray
    range_doppler_map: np.ndarray
    micro_doppler_spectrogram: np.ndarray
    intelligence_output: IntelligenceOutput
    stat_metrics: dict
    resolution_benchmarks: ResolutionMetrics
    probabilistic_stats: PerformanceStats
    cognitive_narrative: str = ""


class CognitiveRadarPipeline:
    """
    High-level orchestration engine for the Cognitive Photonic Radar.
    """
    def __init__(self, sampling_period_s: float = 1e-3):
        """
        Initializes the strategic pipeline components.
        """
        self.intelligence_unit = IntelligencePipeline() 
        self.cognitive_engine = CognitiveRadarEngine()
        self.track_manager = TacticalTrackManager(sampling_period_s=sampling_period_s)
        self.frame_index = 0
        self.active_adaptation = None
        
    def execute_tactical_processing_frame(self, 
                                        photonic_cfg: PhotonicConfig, 
                                        channel_cfg: ChannelConfig, 
                                        noise_cfg: NoiseConfig,
                                        active_targets: List[Target]
                                        ) -> TacticalIntelligenceFrame:
        """
        Executes a complete closed-loop radar processing cycle.
        """
        self.frame_index += 1
        
        # --- 1. Cognitive Parameter Adaptation (Feedback Loop) ---
        # Apply scaling factors computed from the previous frame's assessment
        if self.active_adaptation:
            photonic_cfg.bandwidth_scaling_factor = self.active_adaptation.bandwidth_scaling
            photonic_cfg.transmit_power_scaling_factor = self.active_adaptation.tx_power_scaling
            
        # --- 2. Photonic Signal Generation & Environment Simulation ---
        time_vector, tx_pulse = generate_synthetic_photonic_signal(photonic_cfg)
        
        # Simulate physical environment (Reflection, Delay, Doppler)
        rx_echoes = simulate_target_response(tx_pulse, photonic_cfg.sampling_rate_hz, 
                                           active_targets, channel_cfg)
        
        # Inject hardware-level noise (RIN, Dispersion, Thermal)
        optical_pwr_watts = 10**(photonic_cfg.optical_power_dbm/10 - 3)
        rx_signal_noised = rx_echoes + add_rin_noise(optical_pwr_watts, noise_cfg, 
                                                   len(tx_pulse), photonic_cfg.sampling_rate_hz)
        rx_signal_noised = apply_fiber_dispersion(rx_signal_noised, noise_cfg, photonic_cfg.sampling_rate_hz)
        rx_signal_noised = add_thermal_noise(rx_signal_noised, noise_cfg, photonic_cfg.sampling_rate_hz)
        
        # --- 3. Digital Signal Processing (DSP) & Detection ---
        intermediate_freq_signal = dechirp_signal(rx_signal_noised, tx_pulse)
        
        # Configure tactical CPI (Coherent Processing Interval)
        pulses_per_cpi = 64
        samples_per_pulse = 64
        
        # Detect targets using adaptive CFAR logic
        cfar_alpha_scale = self.active_adaptation.cfar_alpha_scale if self.active_adaptation else 1.0
        dsp_results = execute_detection_pipeline(
            intermediate_freq_signal, 
            num_pulses=pulses_per_cpi, 
            samples_per_pulse=samples_per_pulse,
            sampling_rate_hz=photonic_cfg.sampling_rate_hz, 
            n_fft_range=128, 
            n_fft_doppler=128,
            cognitive_alpha_scale=cfar_alpha_scale
        )
        
        rd_intensity_map = dsp_results["rd_map"]
        target_centroids = [(d[0], d[1]) for d in dsp_results["detections"]]
        
        # Extract temporal frequency signatures (Micro-Doppler)
        micro_doppler_spec = compute_spectrogram(intermediate_freq_signal, photonic_cfg.sampling_rate_hz)
        
        # --- 4. Tactical Tracking & Multimodal AI Intelligence ---
        # Map spectral indices to physical world coordinates (Simplified for demo)
        observed_vectors = [(c[0] * 5.0, c[1] * 0.5) for c in target_centroids] 
        track_summaries = self.track_manager.update_pipeline(observed_vectors)
        
        # Multimodal classification of the primary target cluster
        intel_output = self.intelligence_unit.infer_tactical_intelligence(rd_intensity_map, micro_doppler_spec)
        
        # --- 5. Situation Assessment & Cognitive Decisioning ---
        # Prepare track data for cognitive assessment
        track_dictionaries = [create_track_dict_for_cognitive(t) for t in self.track_manager.active_tracks]
        
        # Per-target intelligence mapping
        target_classifications = []
        for track in track_summaries:
            target_classifications.append({
                "class": intel_output.tactical_class,
                "confidence": intel_output.inference_confidence,
                "class_probabilities": list(intel_output.class_probabilities.values())
            })
            
        situation_assessment = self.cognitive_engine.assess_situation(
            frame_id=self.frame_index,
            timestamp=float(self.frame_index * 0.1),
            detections=observed_vectors,
            tracks=track_dictionaries,
            ai_predictions=target_classifications,
            rd_map=rd_intensity_map
        )
        
        # Compute adaptation commands for the next frame (N+1)
        self.active_adaptation = self.cognitive_engine.decide_adaptation(situation_assessment)
        xai_narrative = self.cognitive_engine.generate_xai_narrative(situation_assessment, self.active_adaptation)
        
        # --- 6. Performance Benchmarking & QA ---
        resolution_benchmarks = calculate_theoretical_resolutions(
            photonic_cfg.sweep_bandwidth_hz * photonic_cfg.bandwidth_scaling_factor, 
            photonic_cfg.chirp_duration_s, 
            channel_cfg.carrier_freq, 
            pulses_per_cpi, 
            photonic_cfg.sampling_rate_hz
        )
        
        # SNR Estimation from residue-to-peak ratio
        peak_pwr = np.max(rd_intensity_map)
        noise_floor = np.median(rd_intensity_map)
        snr_est_db = float(peak_pwr - noise_floor)
        
        # Statistical confidence validation
        pd_estimate = estimate_pd_swerling1(snr_est_db, pfa_target=1e-6)
        far_estimate = estimate_false_alarm_rate(threshold_voltage=13.0, noise_variance=1.0)
        
        # Extract metadata from AI probabilities
        inference_stats = evaluate_ai_inference_confidence(list(intel_output.class_probabilities.values()))
        feature_vector = extract_statistical_features(intermediate_freq_signal)
        
        probabilistic_stats = PerformanceStats(
            snr_db=snr_est_db,
            probability_detection=pd_estimate,
            false_alarm_rate=far_estimate,
            model_confidence=inference_stats["confidence"],
            model_entropy=inference_stats["entropy"]
        )
        
        # --- 7. Final Data Packaging ---
        return TacticalIntelligenceFrame(
            frame_id=self.frame_index,
            time_velocity_axis=time_vector,
            raw_rx_signal=rx_signal_noised,
            range_doppler_map=rd_intensity_map,
            micro_doppler_spectrogram=micro_doppler_spec,
            intelligence_output=intel_output,
            stat_metrics={**feature_vector, "snr_db": snr_est_db},
            resolution_benchmarks=resolution_benchmarks,
            probabilistic_stats=probabilistic_stats,
            cognitive_narrative=xai_narrative
        )
