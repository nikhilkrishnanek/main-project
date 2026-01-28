"""
Multimodal Synthetic Radar Dataset Generator
============================================

This module orchestrates the generation of high-fidelity synthetic radar data 
for training tactical intelligence models. It utilizes underlying physics 
models for micro-Doppler, JEM, and stochastic clutter to create 
multi-modal samples including:
1. Range-Doppler Intensity Maps (Spatial-Spectral)
2. Micro-Doppler Spectrograms (Temporal-Frequency)
3. Raw Pulse/Kinematic Streams (Temporal IQ)

Tactical Classes:
-----------------
- Drone: Hexacopter/Quadcopter models with multiple rotor modulation sidebands.
- Aircraft: Large RCS fixed-wing platforms with Jet Engine Modulation (JEM).
- Missile: High-hypersonic profiles with significant Doppler drift and acceleration.
- Bird: Low-RCS biological targets with erratic flapping amplitude modulation.
- Noise: Stochastic background (AWGN) and heavy-tailed radar clutter.

Author: Senior AI Research Scientist (Radar Physics)
"""

import numpy as np
import os
import torch
from typing import Tuple, List, Dict
from photonic.signals import generate_synthetic_photonic_signal
from signal_processing.transforms import compute_range_doppler_map, compute_spectrogram
from signal_processing.noise import inject_thermal_awgn, generate_stochastic_clutter
from ai_models.model import get_tactical_classes


class RadarDatasetGenerator:
    """
    Orchestration engine for synthetic tactical radar data synthesis.
    """
    def __init__(self, simulation_config: Dict):
        self.config = simulation_config
        self.target_labels = get_tactical_classes()
        
    def synthesize_multimodal_sample(self, tactical_class_name: str) -> Dict[str, np.ndarray]:
        """
        Synthesizes a single high-fidelity tactical radar sample.
        """
        duration = self.config.get('chirp_duration_s', 0.1)
        fs = self.config.get('sampling_rate_hz', 5e5)
        num_samples = int(fs * duration)
        time_vector = np.arange(num_samples) / fs
        
        target_class = tactical_class_name.lower()
        kinematic_params = {}
        
        # 1. Physics-Driven Signal Synthesis
        if target_class == "noise":
            # Tactical Clutter Modeling (K-distribution for non-Rayleigh sea/urban clutter)
            signal = generate_stochastic_clutter(num_samples, distribution_type='k', 
                                              shape_parameter=1.5, scale_parameter=2.0)
        
        elif target_class == "bird":
            # Biological modulation: 3-10Hz erratic wing flapping AM
            velocity = np.random.uniform(5, 15)
            flapping_freq = np.random.uniform(3, 10)
            carrier_phase = np.exp(1j * 2 * np.pi * velocity * time_vector)
            amplitude_modulation = 0.5 * (1 + 0.6 * np.sin(2 * np.pi * flapping_freq * time_vector))
            signal = carrier_phase * amplitude_modulation
            
        else:
            # Platform-specific kinematic and modulation parameters
            platform_registry = {
                "drone": {"v": 15, "rcs": -15, "rotors": 4, "rpm": 15000, "blade_len": 0.15},
                "aircraft": {"v": 250, "rcs": 25, "jem_blades": 32, "jem_rpm": 8000},
                "missile": {"v_start": 400, "accel": 150, "rcs": 5}
            }
            kinematic_params = platform_registry[target_class]
            
            if target_class == "drone":
                # Multi-rotor simulation with blade-tip phase modulation
                actual_velocity = kinematic_params["v"] + np.random.normal(0, 3)
                carrier = np.exp(1j * 2 * np.pi * actual_velocity * time_vector)
                
                micro_doppler_modulation = np.zeros(num_samples, dtype=complex)
                for _ in range(kinematic_params["rotors"]):
                    rotor_rpm = kinematic_params["rpm"] + np.random.normal(0, 1000)
                    fm = rotor_rpm / 60.0
                    # Phase modulation depth (beta) based on blade length and wavelength
                    beta = (2 * np.pi * kinematic_params["blade_len"]) / 0.03
                    micro_doppler_modulation += np.exp(1j * beta * np.sin(2 * np.pi * fm * time_vector))
                
                signal = carrier * (micro_doppler_modulation / kinematic_params["rotors"])
                
            elif target_class == "aircraft":
                # Jet Engine Modulation (JEM) sideband simulation
                actual_velocity = kinematic_params["v"] + np.random.normal(0, 5)
                carrier = np.exp(1j * 2 * np.pi * actual_velocity * time_vector)
                
                jem_frequency = kinematic_params["jem_blades"] * kinematic_params["jem_rpm"] / 60.0
                jem_sideband_modulation = 1 + 0.2 * np.cos(2 * np.pi * jem_frequency * time_vector)
                signal = carrier * jem_sideband_modulation
                
            elif target_class == "missile":
                # High-speed kinematic profile with linear radial acceleration (Doppler drift)
                v0 = kinematic_params["v_start"] + np.random.normal(0, 50)
                acceleration = kinematic_params["accel"] + np.random.normal(0, 20)
                wavelength = 0.03 # 10 GHz
                
                # Integrated phase for acceleration: phi(t) = 2pi * (v0*t + 0.5*a*t^2) / lambda
                phase_path = (2 * np.pi / wavelength) * (v0 * time_vector + 0.5 * acceleration * time_vector**2)
                signal = np.exp(1j * phase_path)

        # 2. Adaptive Noise Injection (Signal-to-Noise Floor Calibration)
        target_snr = np.random.uniform(5, 30)
        signal = inject_thermal_awgn(signal, target_snr_db=target_snr)
        
        # 3. Multimodal Tactical Feature Extraction
        # Spatial-Spectral Mapping (Range-Doppler)
        # Using 64-pulse coherent processing interval (CPI)
        spectral_map = compute_range_doppler_map(signal, n_chirps=64, samples_per_chirp=num_samples//64)
        
        # Temporal-Frequency Mapping (Spectrogram)
        temporal_spectrogram = compute_spectrogram(signal, sampling_rate_hz=fs, nperseg=256, noverlap=128)
        
        # Raw Kinematic Kinematics (I/Q time-series)
        # Sub-sampled to standard intelligence sequence length
        sub_samples = 512
        raw_iq_series = np.stack([np.real(signal[:sub_samples]), np.imag(signal[:sub_samples])])
        
        return {
            "spectral_map": spectral_map,
            "spectrogram": temporal_spectrogram,
            "kinematic_series": raw_iq_series,
            "label_index": self.target_labels.index(tactical_class_name),
            "metadata": {
                "class": tactical_class_name,
                "snr_db": target_snr,
                "kinematic_params": kinematic_params if kinematic_params else {},
            }
        }

    def generate_batch(self, samples_per_class: int = 50) -> Dict[str, torch.Tensor]:
        """
        Orchestrates the synthesis of a balanced multimodal training batch.
        """
        accumulation_rd = []
        accumulation_spec = []
        accumulation_ts = []
        accumulation_y = []
        
        for tactical_class in self.target_labels:
            for _ in range(samples_per_class):
                sample = self.synthesize_multimodal_sample(tactical_class)
                accumulation_rd.append(sample["spectral_map"])
                accumulation_spec.append(sample["spectrogram"])
                accumulation_ts.append(sample["kinematic_series"])
                accumulation_y.append(sample["label_index"])
                
        return {
            "rd_maps": torch.tensor(np.array(accumulation_rd), dtype=torch.float32).unsqueeze(1),
            "spectrograms": torch.tensor(np.array(accumulation_spec), dtype=torch.float32).unsqueeze(1),
            "time_series": torch.tensor(np.array(accumulation_ts), dtype=torch.float32),
            "labels": torch.tensor(np.array(accumulation_y), dtype=torch.long)
        }
