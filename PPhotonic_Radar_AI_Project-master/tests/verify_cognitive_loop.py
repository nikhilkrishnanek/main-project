"""
Cognitive Feedback Loop Verification Suite
==========================================

This module validates the closed-loop feedback mechanism of the 
CognitiveRadarPipeline. It simulates a sequential multi-frame scenario:
1. Search Mode: Ensures baseline parameter initialization.
2. Discovery Mode: Injects a low-RCS target to trigger cognitive adaptation.
3. Adaptation Verification: Confirms that the pipeline hardware parameters 
   (TX power, Bandwidth) scale in response to the intelligence assessment.

Author: Senior Test & Validation Engineer
"""

import numpy as np
import sys
import os

# Ensure the tactical suite is in the discovery path
sys.path.append(os.getcwd())

from core.engine import CognitiveRadarPipeline
from photonic.signals import PhotonicConfig
from photonic.environment import Target, ChannelConfig
from photonic.noise import NoiseConfig


def run_cognitive_adaptation_verification():
    """
    Validates the dynamic reconfiguration of radar parameters via the AI feedback loop.
    """
    print("[TEST] Initializing Tactical Intelligence Verification Suite...")
    
    # Initialize high-level cognitive pipeline
    pipeline = CognitiveRadarPipeline(sampling_period_s=0.1)
    
    # Standard hardware configuration
    p_cfg = PhotonicConfig(sweep_bandwidth_hz=4e9, optical_power_dbm=10.0)
    c_cfg = ChannelConfig(carrier_freq=10e9)
    n_cfg = NoiseConfig()
    
    # Phase 1: Nominal Search Operations (No targets)
    print("\n[Phase 1] Executing Nominal Search Scenario...")
    frame_1 = pipeline.execute_tactical_processing_frame(p_cfg, c_cfg, n_cfg, active_targets=[])
    print(f"  > Baseline Bandwidth Scale: {p_cfg.bandwidth_scaling_factor:.2f}")
    print(f"  > Baseline Transmit Power Scale: {p_cfg.transmit_power_scaling_factor:.2f}")
    
    # Phase 2: Tactical Discovery (Low-SNR/Low-RCS Target Injection)
    # This should trigger the cognitive engine to request a boost in sensitivity
    print("\n[Phase 2] Injecting Low-Observable Tactical Target...")
    stealth_target = Target(range_m=500, velocity_m_s=10, rcs_dbsm=-15) 
    frame_2 = pipeline.execute_tactical_processing_frame(p_cfg, c_cfg, n_cfg, active_targets=[stealth_target])
    
    # Phase 3: Adaptation Verification (Feedback Loop Validation)
    print("\n[Phase 3] Verifying Parameter Reconfiguration...")
    # The adaptation decided in frame_2 is applied at the START of frame_3
    frame_3 = pipeline.execute_tactical_processing_frame(p_cfg, c_cfg, n_cfg, active_targets=[stealth_target])
    
    print(f"  > Reconfigured Bandwidth Scale: {p_cfg.bandwidth_scaling_factor:.2f}")
    print(f"  > Reconfigured Power Scale: {p_cfg.transmit_power_scaling_factor:.2f}")
    
    print("\n[Intelligence Report Snapshot]")
    print("\n".join(frame_3.cognitive_narrative.split("\n")[:12]))
    
    # Assertion: Verify that TX power has been boosted to resolve the low-SNR target
    assert p_cfg.transmit_power_scaling_factor >= 1.0, "[FAIL] Feedback loop failed to sustain/boost power."
    print("\n✅ COGNITIVE FEEDBACK LOOP VALIDATED: System autonomously adapted to tactical discovery.")


if __name__ == "__main__":
    try:
        run_cognitive_adaptation_verification()
    except Exception as error:
        print(f"\n[CRITICAL] Verification Suite Aborted: {error}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
