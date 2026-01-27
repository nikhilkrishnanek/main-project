"""
Radar Comparison Module: Photonic vs Electronic
==============================================

This module quantitatively compares Microwave Photonic Radar against 
Conventional Electronic Radar systems.

Key Differentiators Modeled:
1. Bandwidth: Photonics allows multi-octave GHz bandwidths (better resolution).
2. Transmission Loss: Fiber (0.2 dB/km) vs Coax (100+ dB/km at 20GHz).
3. EMI Immunity: Photonic links are immune to electromagnetic interference.
4. Frequency Stability: Optical generation offers superior phase noise at high freqs.

Author: Principal Radar Scientist
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class RadarSystemSpecs:
    name: str
    carrier_freq_ghz: float
    max_bandwidth_ghz: float
    transmission_loss_db_per_km: float # Coax vs Fiber
    emi_susceptibility_db: float      # Susceptibility to interference
    phase_noise_10khz_dbc: float      # Phase noise at 10kHz offset
    
    @property
    def range_resolution_cm(self) -> float:
        """Delta R = c / 2B"""
        c = 3e8
        # Avoid div zero
        bw = max(self.max_bandwidth_ghz * 1e9, 1.0)
        return (c / (2 * bw)) * 100.0

def get_photonic_specs() -> RadarSystemSpecs:
    """Returns typical specs for a modern Photonic Radar."""
    return RadarSystemSpecs(
        name="Microwave Photonic Radar",
        carrier_freq_ghz=20.0,
        max_bandwidth_ghz=8.0,      # Huge bandwidth
        transmission_loss_db_per_km=0.2, # Fiber is ultra-low loss
        emi_susceptibility_db=-100.0,   # Immune
        phase_noise_10khz_dbc=-120.0    # Extremely clean
    )

def get_electronic_specs() -> RadarSystemSpecs:
    """Returns typical specs for a conventional K-band Electronic Radar."""
    return RadarSystemSpecs(
        name="Conventional Electronic Radar",
        carrier_freq_ghz=20.0,
        max_bandwidth_ghz=0.5,      # Bandwidth limited by electronics/mixers
        transmission_loss_db_per_km=50.0, # Coax loss is huge at 20GHz
        emi_susceptibility_db=-40.0,    # Susceptible to crosstalk/jamming
        phase_noise_10khz_dbc=-90.0     # Noisier oscillators
    )

def compare_systems(dist_km: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Generates a comparative analysis report.
    Args:
        dist_km: Distance for transmission loss calculation (e.g. remoting antennas).
    """
    pho = get_photonic_specs()
    ele = get_electronic_specs()
    
    # 1. Resolution Comparison
    # Ratio > 1 means Photonic is better (finer resolution = smaller number)
    # We invert logic: How many times better?
    res_improvement = ele.range_resolution_cm / pho.range_resolution_cm
    
    # 2. Transmission Efficiency
    # Loss at distance
    loss_pho = pho.transmission_loss_db_per_km * dist_km
    loss_ele = ele.transmission_loss_db_per_km * dist_km
    loss_advantage = loss_ele - loss_pho # Positive = Photonic saves X dB
    
    # 3. EMI Resistance
    # Direct comparison in dB
    emi_advantage = ele.emi_susceptibility_db - pho.emi_susceptibility_db
    
    return {
        "resolution": {
            "photonic_cm": pho.range_resolution_cm,
            "electronic_cm": ele.range_resolution_cm,
            "improvement_factor": res_improvement
        },
        "transmission_loss": {
            "distance_km": dist_km,
            "photonic_loss_db": loss_pho,
            "electronic_loss_db": loss_ele,
            "advantage_db": loss_advantage
        },
        "phase_noise": {
            "photonic_dbc": pho.phase_noise_10khz_dbc,
            "electronic_dbc": ele.phase_noise_10khz_dbc
        }
    }

if __name__ == "__main__":
    report = compare_systems(dist_km=1.0)
    print("\n--- Radar Comparison Report ---")
    
    print(f"\n1. RANGE RESOLUTION (Lower is Better)")
    res = report["resolution"]
    print(f"   Photonic  : {res['photonic_cm']:.2f} cm")
    print(f"   Electronic: {res['electronic_cm']:.2f} cm")
    print(f"   Advantage : {res['improvement_factor']:.1f}x finer detail")
    
    print(f"\n2. TRANSMISSION LOSS (1 km Link)")
    tl = report["transmission_loss"]
    print(f"   Photonic  : {tl['photonic_loss_db']:.2f} dB (Fiber)")
    print(f"   Electronic: {tl['electronic_loss_db']:.2f} dB (Coax)")
    print(f"   Saved Power: {tl['advantage_db']:.1f} dB")
    
    print(f"\n3. EMI IMMUNITY")
    print(f"   Photonic systems are inherently resistant to EMP and RFI due to optical nature.")
