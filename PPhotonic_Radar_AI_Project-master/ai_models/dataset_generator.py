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
        self.classes = ["drone", "aircraft", "missile", "noise"]
        
    def generate_sample(self, target_class: str) -> Dict[str, np.ndarray]:
        """
        Generates a single multi-modal sample (RD-Map, Spectrogram, Time-Series).
        """
        duration = self.config.get('duration', 0.1)
        fs = self.config.get('fs', 1e6)
        
        # 1. Generate core signal based on class physics
        if target_class == "noise":
            signal = generate_clutter(int(fs*duration), distribution='weibull')
        else:
            # Simplified physics mapping
            params = {
                "drone": {"v": 15, "rcs": -10, "md": True},
                "aircraft": {"v": 200, "rcs": 20, "md": False},
                "missile": {"v": 800, "rcs": 5, "md": False}
            }[target_class]
            
            # Use top-level photonic signal generator
            # Note: We simulate micro-Doppler for drones by modulating the phase
            t = np.arange(int(fs*duration)) / fs
            carrier = np.exp(1j * 2 * np.pi * params["v"] * t) # Doppler
            
            if params["md"]:
                # Micro-Doppler (Rotor)
                carrier *= np.exp(1j * 2 * np.pi * 5 * np.sin(2 * np.pi * 50 * t))
                
            signal = add_awgn(carrier, snr_db=np.random.uniform(5, 25))
            
        # 2. Extract Features (Inputs for AI)
        rd_map = compute_range_doppler_map(signal, n_chirps=64, samples_per_chirp=len(signal)//64)
        spec = compute_spectrogram(signal, fs=fs)
        
        # Doppler Time-Series (Sub-sampled for LSTM)
        time_series = np.abs(signal[:1000]) # Take first 1000 points
        
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
