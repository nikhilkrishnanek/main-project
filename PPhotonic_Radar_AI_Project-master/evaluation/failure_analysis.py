"""
AI Failure Analysis Module
=========================

Diagnoses the root cause of AI errors by comparing Ground Truth vs Prediction
against environmental conditions (SNR, Clutter).

Key Functions:
1. analyze_error: Main entry point.
2. _diagnose_miss: Why did we miss the target?
3. _diagnose_false_alarm: Why did we hallucinate a target?

Author: Principal QA Engineer
"""

from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class FailureReport:
    error_type: str  # "Miss", "False Alarm", "Misclassification", "Correct"
    root_cause: str
    remediation: str
    severity: str    # "Critical", "Warning", "Info"

def analyze_prediction(
    ground_truth_class: str,
    predicted_class: str,
    snr_db: float,
    clutter_power_db: float = -100.0,
    confidence: float = 0.0
) -> FailureReport:
    """
    Performs Root Cause Analysis on a single prediction event.
    
    Args:
        ground_truth_class: Actual label (e.g. "Drone", "Noise")
        predicted_class: Model output
        snr_db: Signal-to-Noise Ratio at target bin
        clutter_power_db: Power of competing clutter
        confidence: Model confidence score
    """
    
    # 1. Check for Match
    if ground_truth_class == predicted_class:
        return FailureReport(
            error_type="Correct",
            root_cause="N/A",
            remediation="None",
            severity="Info"
        )
        
    # 2. Missed Detection (Truth=Object, Pred=Noise)
    if ground_truth_class != "Noise" and predicted_class == "Noise":
        # Diagnosis: Is it SNR?
        if snr_db < 13.0:
            return FailureReport(
                error_type="Missed Detection",
                root_cause=f"Low SNR ({snr_db:.1f} dB). Below robust detection threshold (13dB).",
                remediation="Increase Tx Power or Integration Time (more chirps).",
                severity="Critical"
            )
        else:
             return FailureReport(
                error_type="Missed Detection",
                root_cause="Model Failure. Signal was strong but features were unrecognized.",
                remediation="Add this sample to Training Set (Hard Example Mining).",
                severity="Critical"
            )

    # 3. False Alarm (Truth=Noise, Pred=Object)
    if ground_truth_class == "Noise" and predicted_class != "Noise":
        # Diagnosis: Clutter?
        if clutter_power_db > -60.0:
             return FailureReport(
                error_type="False Alarm",
                root_cause="High Clutter Environment. Clutter spikes mistaken for target.",
                remediation="Enable MTI (Moving Target Indication) filter.",
                severity="Warning"
            )
        else:
            return FailureReport(
                error_type="False Alarm",
                root_cause="Noise Spike / Hallucination.",
                remediation="Increase CFAR Threshold.",
                severity="Warning"
            )
            
    # 4. Misclassification (Truth=A, Pred=B)
    # e.g. Drone vs Bird
    if snr_db < 10.0:
         return FailureReport(
                error_type=f"Misclassification ({ground_truth_class}->{predicted_class})",
                root_cause="Signal Degradation. Features corrupted by noise.",
                remediation="Improve SNR.",
                severity="Warning"
            )
    
    return FailureReport(
        error_type=f"Misclassification ({ground_truth_class}->{predicted_class})",
        root_cause="Feature Ambiguity. Signatures are similar in current domain.",
        remediation="Check Micro-Doppler spectrogram resolution.",
        severity="Warning"
    )

if __name__ == "__main__":
    # Smoke Test
    print("--- Failure Analysis Audit ---")
    
    # Case 1: Miss due to SNR
    r1 = analyze_prediction("Drone", "Noise", snr_db=5.0)
    print(f"1. {r1.error_type}: {r1.root_cause}")
    
    # Case 2: False Alarm
    r2 = analyze_prediction("Noise", "Missile", snr_db=0.0, clutter_power_db=-40.0)
    print(f"2. {r2.error_type}: {r2.root_cause}")
    
    # Case 3: Confused
    r3 = analyze_prediction("Bird", "Drone", snr_db=20.0)
    print(f"3. {r3.error_type}: {r3.root_cause} -> {r3.remediation}")
