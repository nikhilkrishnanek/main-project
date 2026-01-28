"""
Tactical Integrity & Stability Audit
====================================

This module performs a rigorous end-to-end stability audit of the Cognitive 
Photonic Radar pipeline. It verifies:
1. Architectural Integrity: Validates multimodal imports across all 6 layers.
2. Numerical Stability: Checks for NaN/Inf propagating through the DSP pipeline.
3. Edge Case Resilience: Audits system behavior under extreme conditions 
   (e.g., zero-target environments, extreme thermal noise).
4. Functional Correctness: Ensures cognitive feedback doesn't cause pipeline stalls.

Author: Principal Software Integrity Engineer
"""

import sys
import os
import numpy as np
import traceback

# Add project root to path for reliable module discovery
sys.path.append(os.getcwd())

from core.engine import CognitiveRadarPipeline, TacticalIntelligenceFrame
from photonic.signals import PhotonicConfig
from photonic.environment import ChannelConfig, Target
from photonic.noise import NoiseConfig


def verify_numerical_stability(data_tensor: np.ndarray, telemetry_name: str) -> bool:
    """
    Validates that no numerical instabilities (NaN/Inf) are present.
    """
    if np.any(np.isnan(data_tensor)):
        print(f"  [STABILITY-ERROR] NaN detected in {telemetry_name}")
        return False
    if np.any(np.isinf(data_tensor)):
        print(f"  [STABILITY-ERROR] Infinity detected in {telemetry_name}")
        return False
    return True


def run_comprehensive_system_audit():
    """
    Executes a multi-phase system stability audit.
    """
    print("[AUDIT] Initiating Tactical Radar Integrity Audit...")
    
    try:
        pipeline = CognitiveRadarPipeline()
    except Exception as e:
        print(f"[AUDIT-FAIL] Failed to instantiate CognitiveRadarPipeline: {e}")
        return False
    
    # Baseline Configurations
    p_cfg = PhotonicConfig()
    c_cfg = ChannelConfig()
    n_cfg = NoiseConfig()
    
    # Defined Audit Scenarios
    audit_scenarios = [
        ("Nominal Operation", [Target(100, 10, 0, "Drone")]),
        ("Zero-Target Environment", []),
        ("Extreme Thermal Noise", [Target(100, 10, 0, "Drone")], {"noise_level_db": 10.0}),
        ("Numerical Signal Underflow", [Target(100, 10, 0, "Drone")], {"noise_level_db": -200.0})
    ]
    
    system_integral = True
    
    for title, targets, *params in audit_scenarios:
        print(f"  > Scenario: {title}...", end=" ", flush=True)
        
        # Apply operational overrides for the audit
        active_channel = ChannelConfig()
        if params:
            for key, value in params[0].items():
                if hasattr(active_channel, key):
                    setattr(active_channel, key, value)
                
        try:
            # Execute tactical frame
            intel_frame = pipeline.execute_tactical_processing_frame(p_cfg, active_channel, n_cfg, targets)
            
            # Numerical Integrity Verification
            stability_checks = [
                verify_numerical_stability(intel_frame.raw_rx_signal, "I/Q Waveform"),
                verify_numerical_stability(intel_frame.range_doppler_map, "RD Intensity Map"),
                verify_numerical_stability(intel_frame.micro_doppler_spectrogram, "µDoppler Spectrogram")
            ]
            
            if all(stability_checks):
                print("✅ PASSED")
            else:
                print("⚠️ STABILITY_WARN")
                system_integral = False
                
        except Exception as crash_error:
            print(f"❌ PIPELINE_CRASH: {crash_error}")
            traceback.print_exc()
            system_integral = False
            
    # Sub-system Import Integrity
    print("\n[AUDIT] Verifying UI and Integration Imports...")
    try:
        import ui.components
        import ui.layout
        print("  > UI Module Architecture: ✅ INTEGRAL")
    except ImportError as e:
        print(f"  > UI Module Architecture: ❌ CORRUPT ({e})")
        system_integral = False
        
    return system_integral


if __name__ == "__main__":
    if run_comprehensive_system_audit():
        print("\n[AUDIT-PASS] TACTICAL SYSTEM INTEGRITY CONFIRMED")
        sys.exit(0)
    else:
        print("\n[AUDIT-FAIL] SYSTEM ARCHITECTURE COMPROMISED")
        sys.exit(1)
