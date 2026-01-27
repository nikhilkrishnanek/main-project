"""
Synthetic Radar Dataset Generator
=================================

This module generates large-scale synthetic datasets for training 
Radar AI models. It leverages the photonic and signal_processing layers 
to create physically realistic samples.

Classes:
- Drone: High micro-Doppler variance.
- Aircraft: Large RCS, steady trajectory.
- Missile: High velocity, low micro-Doppler.
- Noise: Thermal/Clutter background.

Author: Radar AI Engineer
"""

import numpy as np
import os
import torch
from typing import Tuple, List, Dict
from photonic.signals import generate_photonic_signal
from signal_processing.transforms import compute_range_doppler_map, compute_spectrogram
from signal_processing.noise import add_awgn, generate_clutter

class RadarDatasetGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.classes = ["drone", "aircraft", "missile", "bird", "noise"]
        
    def generate_sample(self, target_class: str) -> Dict[str, np.ndarray]:
        """
        Generates a physically realistic multi-modal sample.
        Includes complex modulation for Micro-Doppler and tactical noise.
        """
        duration = self.config.get('duration', 0.1)
        fs = self.config.get('fs', 5e5)
        n_samples = int(fs * duration)
        t = np.arange(n_samples) / fs
        
        # 1. Physics-based Signal Generation
        if target_class == "noise":
            # Tactical Clutter (Sea/Urban mix)
            signal = generate_clutter(n_samples, distribution='k', shape=1.5, scale=2.0)
        
        elif target_class == "bird":
            # Biological modulation: Low frequency, erratic amplitude
            v = np.random.uniform(2, 12)
            flapping_freq = np.random.uniform(2, 8)
            carrier = np.exp(1j * 2 * np.pi * v * t)
            # Amplitude modulation for flapping
            am = 0.5 * (1 + 0.4 * np.sin(2 * np.pi * flapping_freq * t))
            signal = carrier * am
            
        else:
            params = {
                "drone": {"v": 15, "rcs": -15, "rotors": 4, "rpm": 12000},
                "aircraft": {"v": 240, "rcs": 25, "rotors": 0},
                "missile": {"v": 950, "rcs": 2, "rotors": 0}
            }[target_class]
            
            # Base Doppler
            v_actual = params["v"] + np.random.normal(0, 2)
            carrier = np.exp(1j * 2 * np.pi * v_actual * t)
            
            # Micro-Doppler (Rotor Physics)
            if params["rotors"] > 0:
                md_signal = np.zeros(n_samples, dtype=complex)
                for _ in range(params["rotors"]):
                    l_blade = np.random.uniform(0.1, 0.2) # meters
                    rpm = params["rpm"] + np.random.normal(0, 500)
                    fm = rpm / 60.0
                    # Phase modulation from rotatory motion
                    beta = (2 * np.pi * l_blade) / 0.03 # 3cm wavelength approximation
                    md_signal += np.exp(1j * beta * np.sin(2 * np.pi * fm * t))
                carrier *= (md_signal / params["rotors"])
            
            signal = carrier

        # 2. Photonic Noise Integration (WDM/MDM Crosstalk simulation)
        # We simulate the effects of optical nonlinearities and crosstalk
        snr = np.random.uniform(5, 30)
        signal = add_awgn(signal, snr_db=snr)
        
        # Add a subtle "channel hum" representing MDM crosstalk
        hum_freq = 50.0 # Hz
        signal += 0.01 * np.exp(1j * 2 * np.pi * hum_freq * t)

        # 3. Feature Extraction
        # Range-Doppler Tensor (Spatial)
        rd_map = compute_range_doppler_map(signal, n_chirps=64, samples_per_chirp=n_samples//64)
        
        # Spectrogram (Temporal Frequency)
        spec = compute_spectrogram(signal, fs=fs, nperseg=256, noverlap=128)
        
        # Raw IQ Sequence (sub-sampled for time-series branch)
        ts_points = 1000
        time_series = np.concatenate([np.real(signal[:ts_points]), np.imag(signal[:ts_points])])
        
        return {
            "rd_map": rd_map,
            "spectrogram": spec,
            "time_series": time_series,
            "label": self.classes.index(target_class)
        }

    def generate_batch(self, samples_per_class: int = 50) -> Dict[str, torch.Tensor]:
        """
        Generates a full batch ready for training.
        """
        all_rd, all_spec, all_ts, all_y = [], [], [], []
        
        for cls in self.classes:
            for _ in range(samples_per_class):
                sample = self.generate_sample(cls)
                all_rd.append(sample["rd_map"])
                all_spec.append(sample["spectrogram"])
                all_ts.append(sample["time_series"])
                all_y.append(sample["label"])
                
        return {
            "rd_maps": torch.tensor(np.array(all_rd), dtype=torch.float32).unsqueeze(1),
            "spectrograms": torch.tensor(np.array(all_spec), dtype=torch.float32).unsqueeze(1),
            "time_series": torch.tensor(np.array(all_ts), dtype=torch.float32),
            "labels": torch.tensor(np.array(all_y), dtype=torch.long)
        }

if __name__ == "__main__":
    cfg = {"duration": 0.05, "fs": 1e5}
    gen = RadarDatasetGenerator(cfg)
    batch = gen.generate_batch(5)
    print(f"Generated batch | RD shape: {batch['rd_maps'].shape} | Labels: {batch['labels']}")
