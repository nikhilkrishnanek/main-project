"""
Cognitive Radar Decision Engine
================================

Implements AI-driven feedback loop for adaptive radar waveform and threshold control.
Provides closed-loop adaptation based on scene characterization and detection confidence.

Key Functions:
1. Situation Assessment: Computes scene metrics from detections/tracks/inference
2. Decision Logic: Rule-based adaptive decision tree
3. Parameter Optimization: Generates bounded waveform adaptations
4. XAI Feedback: Provides textual explanation for all decisions

Author: Cognitive Radar Systems Team
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

# Setup logging
logger = logging.getLogger(__name__)


class SceneType(Enum):
    """Scene classification enum."""
    SEARCH = "Search"           # No confirmed tracks
    SPARSE = "Sparse"           # Few targets, low clutter
    TRACKING = "Tracking"       # Multiple confirmed tracks
    DENSE = "Dense"             # High target density (swarm)
    CLUTTERED = "Cluttered"     # High false alarm rate


@dataclass
class SituationAssessment:
    """
    Quantitative assessment of current radar scene.
    Computed from detections, tracking, and AI inference.
    """
    frame_id: int
    timestamp: float
    
    # Track metrics
    num_confirmed_tracks: int = 0
    num_provisional_tracks: int = 0
    num_coasting_tracks: int = 0
    
    # Detection metrics
    num_detections: int = 0
    num_false_alarms: int = 0
    
    # Confidence metrics
    mean_classification_confidence: float = 0.5
    mean_class_entropy: float = 0.0  # 0=certain, 1=uniform
    
    # Track quality metrics
    mean_track_stability: float = 0.5
    mean_track_age: float = 0.0
    
    # Scene characterization
    clutter_ratio: float = 0.0  # num_false_alarms / num_detections
    mean_velocity_spread: float = 0.0  # std of track velocities
    scene_type: SceneType = SceneType.SEARCH
    
    # SNR estimation (from RD-map peak vs noise floor)
    estimated_snr_db: float = 10.0
    
    # Power/efficiency
    mean_signal_power: float = 0.0
    mean_noise_power: float = 0.0


@dataclass
class AdaptationCommand:
    """
    Cognitive engine output: Recommended parameter adaptations for next frame.
    """
    frame_id: int
    timestamp: float
    
    # Parameter scaling factors (multiplicative)
    bandwidth_scaling: float = 1.0
    prf_scale: float = 1.0
    tx_power_scaling: float = 1.0
    cfar_alpha_scale: float = 1.0
    dwell_time_scale: float = 1.0
    
    # Explanations
    reasoning: Dict[str, str] = field(default_factory=dict)
    decision_confidence: float = 0.7  # Confidence in this adaptation
    
    # Impact predictions
    predicted_snr_improvement_db: float = 0.0
    predicted_pfa_change: float = 0.0  # Fractional change
    predicted_range_resolution_m: float = 0.0


@dataclass
class CognitiveRadarState:
    """
    Persistent state of cognitive engine across frames.
    """
    last_adaptation: Optional[AdaptationCommand] = None
    adaptation_history: List[AdaptationCommand] = field(default_factory=list)
    parameter_history: List[Dict] = field(default_factory=list)
    scene_history: List[SceneType] = field(default_factory=list)
    
    # Metrics tracking
    mean_confidence_trend: float = 0.5
    track_stability_trend: float = 0.5
    clutter_trend: float = 0.0
    
    # Adaptation hysteresis (prevent oscillation)
    last_major_adaptation_frame: int = -1000


class CognitiveRadarEngine:
    """
    Main cognitive decision engine.
    Closes the loop between AI inference and DSP adaptation.
    """
    
    # Parameter bounds (constraints on scaling)
    ADAPTATION_BOUNDS = {
        'bandwidth_scaling': (0.8, 1.5),      # ±50% max expansion
        'prf_scale': (0.7, 1.3),              # ±30% max change
        'tx_power_scaling': (0.7, 2.0),       # 70% to 200% (realistic)
        'cfar_alpha_scale': (0.85, 1.3),      # ±15-30% threshold change
        'dwell_time_scale': (0.9, 2.0),       # 0.9 to 2.0× coherent time
    }
    
    # Scene thresholds
    CLUTTER_THRESHOLD = 0.2              # Ratio above which scene is "cluttered"
    CONFIDENCE_THRESHOLD_LOW = 0.60      # Below this: low confidence
    CONFIDENCE_THRESHOLD_HIGH = 0.85     # Above this: high confidence
    TRACK_STABILITY_THRESHOLD = 0.5      # Below this: unstable tracks
    VELOCITY_SPREAD_THRESHOLD = 100.0    # m/s, above: high velocity variance
    
    def __init__(self):
        """Initialize cognitive engine."""
        self.state = CognitiveRadarState()
        self.logger = logging.getLogger(__name__)
        
    def assess_situation(self, 
                        frame_id: int,
                        timestamp: float,
                        detections: List[Tuple[float, float]],
                        tracks: List[Dict],
                        ai_predictions: List[Dict],
                        rd_map: Optional[np.ndarray] = None) -> SituationAssessment:
        """
        Compute comprehensive situation assessment from radar observables.
        
        Args:
            frame_id: Current frame index
            timestamp: Frame timestamp (seconds)
            detections: List of (range, doppler) tuples
            tracks: List of track dicts with metadata (state, stability, velocity)
            ai_predictions: List of {class, confidence, entropy}
            rd_map: Optional Range-Doppler map for SNR estimation
            
        Returns:
            SituationAssessment object
        """
        assessment = SituationAssessment(
            frame_id=frame_id,
            timestamp=timestamp
        )
        
        # --- Track Statistics ---
        assessment.num_confirmed_tracks = sum(1 for t in tracks if t.get('state') == 'CONFIRMED')
        assessment.num_provisional_tracks = sum(1 for t in tracks if t.get('state') == 'PROVISIONAL')
        assessment.num_coasting_tracks = sum(1 for t in tracks if t.get('state') == 'COASTING')
        
        # --- Detection Statistics ---
        assessment.num_detections = len(detections)
        
        # Estimate false alarms: detections not matched to tracks
        # (heuristic: if more detections than tracks, extras are likely clutter)
        num_active_tracks = assessment.num_confirmed_tracks + assessment.num_provisional_tracks
        assessment.num_false_alarms = max(0, assessment.num_detections - num_active_tracks)
        
        # --- Clutter Ratio ---
        if assessment.num_detections > 0:
            assessment.clutter_ratio = assessment.num_false_alarms / assessment.num_detections
        else:
            assessment.clutter_ratio = 0.0
        
        # --- Confidence Metrics ---
        if ai_predictions:
            confidences = [p.get('confidence', 0.5) for p in ai_predictions]
            assessment.mean_classification_confidence = np.mean(confidences) if confidences else 0.5
            
            # Compute class entropy (as proxy for uncertainty)
            entropies = []
            for pred in ai_predictions:
                class_probs = pred.get('class_probabilities', [1.0])
                if class_probs:
                    entropy = -np.sum([p * np.log(np.clip(p, 1e-10, 1.0)) 
                                       for p in class_probs])
                    entropies.append(entropy)
            assessment.mean_class_entropy = np.mean(entropies) if entropies else 0.0
        
        # --- Track Quality Metrics ---
        if tracks:
            stabilities = [t.get('stability_score', 0.5) for t in tracks]
            ages = [t.get('age', 1) for t in tracks]
            velocities = [t.get('velocity', 0.0) for t in tracks]
            
            assessment.mean_track_stability = np.mean(stabilities) if stabilities else 0.5
            assessment.mean_track_age = np.mean(ages) if ages else 1.0
            assessment.mean_velocity_spread = np.std(velocities) if len(velocities) > 1 else 0.0
        
        # --- SNR Estimation from RD-Map ---
        if rd_map is not None and rd_map.size > 0:
            peak_power = np.max(rd_map)
            # Noise floor = low percentile
            noise_floor = np.percentile(rd_map, 25)
            if noise_floor > 0:
                assessment.estimated_snr_db = 10 * np.log10(peak_power / noise_floor)
            
            assessment.mean_signal_power = float(np.mean(rd_map[rd_map > np.median(rd_map)]))
            assessment.mean_noise_power = float(np.mean(rd_map[rd_map <= np.median(rd_map)]))
        
        # --- Scene Classification ---
        assessment.scene_type = self._classify_scene(assessment)
        
        # --- Update state trends ---
        self.state.mean_confidence_trend = 0.8 * self.state.mean_confidence_trend + \
                                           0.2 * assessment.mean_classification_confidence
        self.state.track_stability_trend = 0.8 * self.state.track_stability_trend + \
                                           0.2 * assessment.mean_track_stability
        self.state.clutter_trend = 0.8 * self.state.clutter_trend + \
                                   0.2 * assessment.clutter_ratio
        
        return assessment
    
    def _classify_scene(self, assessment: SituationAssessment) -> SceneType:
        """
        Classify scene type based on metrics.
        """
        num_confirmed = assessment.num_confirmed_tracks
        clutter_ratio = assessment.clutter_ratio
        
        if num_confirmed == 0:
            return SceneType.SEARCH
        elif clutter_ratio > self.CLUTTER_THRESHOLD:
            return SceneType.CLUTTERED
        elif num_confirmed > 5:
            return SceneType.DENSE
        elif num_confirmed > 0:
            return SceneType.TRACKING
        else:
            return SceneType.SPARSE
    
    def decide_adaptation(self, assessment: SituationAssessment) -> AdaptationCommand:
        """
        Execute cognitive decision logic to determine parameter adaptations.
        Returns bounded, physically-motivated parameter scalings.
        
        Args:
            assessment: SituationAssessment from assess_situation()
            
        Returns:
            AdaptationCommand with all parameter scalings and reasoning
        """
        cmd = AdaptationCommand(
            frame_id=assessment.frame_id,
            timestamp=assessment.timestamp
        )
        
        # Extract convenient metrics
        avg_conf = assessment.mean_classification_confidence
        track_stability = assessment.mean_track_stability
        clutter_ratio = assessment.clutter_ratio
        scene = assessment.scene_type
        snr_db = assessment.estimated_snr_db
        
        # ========== DECISION 1: TRANSMIT POWER ==========
        if avg_conf < self.CONFIDENCE_THRESHOLD_LOW:
            cmd.tx_power_scaling = 1.5  # Increase 50%
            cmd.reasoning['tx_power_scaling'] = \
                f"Low classification confidence ({avg_conf:.1%}): boost TX power for weak target SNR"
        elif track_stability > 0.9 and snr_db > 20:
            cmd.tx_power_scaling = 0.8  # Reduce 20%
            cmd.reasoning['tx_power_scaling'] = \
                f"Stable tracks ({track_stability:.2f}) + high SNR ({snr_db:.1f} dB): reduce power for efficiency"
        else:
            cmd.tx_power_scaling = 1.0
            cmd.reasoning['tx_power_scaling'] = "Nominal operating point"
        
        # ========== DECISION 2: CHIRP BANDWIDTH ==========
        if scene == SceneType.CLUTTERED:
            cmd.bandwidth_scaling = 1.3  # Expand 30%
            cmd.reasoning['bandwidth_scaling'] = \
                f"Cluttered scene (clutter ratio {clutter_ratio:.1%}): expand BW for range resolution"
        elif scene == SceneType.DENSE:
            cmd.bandwidth_scaling = 1.2  # Expand 20%
            cmd.reasoning['bandwidth_scaling'] = \
                "Dense targets (swarm): expand BW to separate closely-spaced targets"
        else:
            cmd.bandwidth_scaling = 1.0
            cmd.reasoning['bandwidth_scaling'] = "Standard range resolution"
        
        # ========== DECISION 3: CFAR THRESHOLD SCALING ==========
        if avg_conf > self.CONFIDENCE_THRESHOLD_HIGH:
            cmd.cfar_alpha_scale = 0.9  # Tighten by 10%
            cmd.reasoning['cfar_alpha_scale'] = \
                f"High classification confidence ({avg_conf:.1%}): aggressive detection enabled"
        elif scene == SceneType.CLUTTERED:
            cmd.cfar_alpha_scale = 1.2  # Relax by 20%
            cmd.reasoning['cfar_alpha_scale'] = \
                "Clutter-rich environment: conservative threshold to avoid false alarms"
        else:
            cmd.cfar_alpha_scale = 1.0
            cmd.reasoning['cfar_alpha_scale'] = "Standard CFAR threshold"
        
        # ========== DECISION 4: COHERENT DWELL TIME ==========
        if track_stability < self.TRACK_STABILITY_THRESHOLD:
            cmd.dwell_time_scale = 1.5  # Extend 50%
            cmd.reasoning['dwell_time_scale'] = \
                f"Unstable tracks ({track_stability:.2f}): extend coherent integration time for SNR"
        elif track_stability > 0.85:
            cmd.dwell_time_scale = 0.95  # Slight reduction for efficiency
            cmd.reasoning['dwell_time_scale'] = "Stable tracks: nominal dwell time"
        else:
            cmd.dwell_time_scale = 1.0
            cmd.reasoning['dwell_time_scale'] = "Standard integration time"
        
        # ========== DECISION 5: PRF ADJUSTMENT ==========
        if assessment.mean_velocity_spread > self.VELOCITY_SPREAD_THRESHOLD:
            cmd.prf_scale = 0.9  # Reduce PRF by 10%
            cmd.reasoning['prf_scale'] = \
                f"High velocity spread ({assessment.mean_velocity_spread:.1f} m/s): " \
                "reduce PRF to widen Doppler unambiguous range"
        else:
            cmd.prf_scale = 1.0
            cmd.reasoning['prf_scale'] = "Standard PRF"
        
        # ========== Apply Bounded Scaling ==========
        cmd = self._apply_bounds(cmd)
        
        # ========== Hysteresis (prevent rapid oscillation) ==========
        if (assessment.frame_id - self.state.last_major_adaptation_frame) < 5:
            # Recent major adaptation, dampen changes
            cmd.bandwidth_scaling = 0.7 * cmd.bandwidth_scaling + 0.3 * 1.0
            cmd.tx_power_scaling = 0.7 * cmd.tx_power_scaling + 0.3 * 1.0
        
        # ========== Compute Expected Impact ==========
        cmd.predicted_snr_improvement_db = 10 * np.log10(cmd.tx_power_scaling) + \
                                           5 * np.log10(cmd.dwell_time_scale)
        
        # Tighter CFAR → lower Pfa (fractional change)
        cmd.predicted_pfa_change = -0.15 * (1.0 - cmd.cfar_alpha_scale)
        
        # Estimate new range resolution
        # Δr = c / (2 * B), so Δr ∝ 1/B
        cmd.predicted_range_resolution_m = 75.0 / cmd.bandwidth_scaling  # Nominal 75m
        
        cmd.decision_confidence = 0.8
        
        # Update state
        self.state.last_adaptation = cmd
        self.state.adaptation_history.append(cmd)
        if len(self.state.adaptation_history) > 1000:
            self.state.adaptation_history.pop(0)
        
        self.state.scene_history.append(scene)
        if len(self.state.scene_history) > 1000:
            self.state.scene_history.pop(0)
        
        return cmd
    
    def _apply_bounds(self, cmd: AdaptationCommand) -> AdaptationCommand:
        """
        Enforce hard bounds on all parameter scalings.
        Prevents unphysical or destabilizing adaptations.
        """
        for param_name, (min_b, max_b) in self.ADAPTATION_BOUNDS.items():
            if hasattr(cmd, param_name):
                current_val = getattr(cmd, param_name)
                bounded_val = np.clip(current_val, min_b, max_b)
                setattr(cmd, param_name, bounded_val)
        
        return cmd
    
    def generate_xai_narrative(self, 
                              assessment: SituationAssessment,
                              cmd: AdaptationCommand) -> str:
        """
        Generate human-readable explanation of cognitive decisions.
        
        Args:
            assessment: Situation assessment
            cmd: Adaptation command
            
        Returns:
            Formatted narrative string
        """
        narrative = f"""
╔════════════════════════════════════════════════════════════════════╗
║           COGNITIVE RADAR ADAPTATION REPORT (Frame {assessment.frame_id})           ║
╚════════════════════════════════════════════════════════════════════╝

SITUATION ASSESSMENT:
  • Scene Type: {assessment.scene_type.value}
  • Active Confirmed Tracks: {assessment.num_confirmed_tracks}
  • Active Provisional Tracks: {assessment.num_provisional_tracks}
  • Total Detections: {assessment.num_detections}
  • Estimated False Alarms: {assessment.num_false_alarms}
  • Clutter Ratio: {assessment.clutter_ratio:.1%}
  
  • Mean Classification Confidence: {assessment.mean_classification_confidence:.1%}
  • Mean Class Entropy: {assessment.mean_class_entropy:.2f} (0=certain, 1=uniform)
  • Mean Track Stability: {assessment.mean_track_stability:.2f}
  • Mean Target Velocity Spread: {assessment.mean_velocity_spread:.1f} m/s
  
  • Estimated SNR: {assessment.estimated_snr_db:.1f} dB
  • Mean Signal Power: {assessment.mean_signal_power:.2e}
  • Mean Noise Power: {assessment.mean_noise_power:.2e}

COGNITIVE DECISIONS & REASONING:
"""
        
        for param, reasoning in cmd.reasoning.items():
            val = getattr(cmd, param)
            narrative += f"\n  [{param.upper().replace('_', ' ')}]: {val:.2f}×"
            narrative += f"\n     → {reasoning}\n"
        
        narrative += f"""
EXPECTED IMPACT (Next Frame):
  • SNR Improvement: +{cmd.predicted_snr_improvement_db:.1f} dB
  • Pfa Change: {cmd.predicted_pfa_change:+.1%}
  • Predicted Range Resolution: ±{cmd.predicted_range_resolution_m:.1f} m
  • Decision Confidence: {cmd.decision_confidence:.1%}

╚════════════════════════════════════════════════════════════════════╝
"""
        return narrative
    
    def get_state_summary(self) -> Dict:
        """
        Return summary of cognitive engine state for monitoring/logging.
        """
        return {
            'mean_confidence_trend': self.state.mean_confidence_trend,
            'track_stability_trend': self.state.track_stability_trend,
            'clutter_trend': self.state.clutter_trend,
            'num_adaptations': len(self.state.adaptation_history),
            'last_adaptation': self.state.last_adaptation,
            'recent_scenes': list(s.value for s in self.state.scene_history[-10:]),
        }


# ============================================================================
# Utility Functions for Integration
# ============================================================================

def extract_track_metrics(tracks: List[Dict]) -> Tuple[float, float, List[float]]:
    """
    Extract track stability and age metrics from track list.
    
    Returns:
        (mean_stability, mean_age, velocities)
    """
    if not tracks:
        return 0.5, 1.0, []
    
    stabilities = []
    ages = []
    velocities = []
    
    for track in tracks:
        # Expect track dict with keys: state, age, hits, consecutive_misses
        if 'stability_score' in track:
            stabilities.append(track['stability_score'])
        else:
            # Compute from hits/age
            hits = track.get('hits', 1)
            age = track.get('age', 1)
            stability = min(1.0, hits / max(age, 1))
            stabilities.append(stability)
        
        ages.append(track.get('age', 1))
        velocities.append(track.get('velocity', 0.0))
    
    mean_stability = np.mean(stabilities) if stabilities else 0.5
    mean_age = np.mean(ages) if ages else 1.0
    
    return mean_stability, mean_age, velocities


def create_track_dict_for_cognitive(track) -> Dict:
    """
    Convert tracking.manager.KinematicTrack object to dict for cognitive engine.
    
    Args:
        track: KinematicTrack object
        
    Returns:
        Dict with required cognitive engine fields
    """
    # Stability = hits / age
    stability = min(1.0, track.total_detections_count / max(track.track_age_frames, 1))
    
    return {
        'track_id': track.track_id,
        'state': track.current_state.name,
        'age': track.track_age_frames,
        'hits': track.total_detections_count,
        'consecutive_misses': track.coasting_frame_count,
        'stability_score': stability,
        'velocity': float(track.state_estimator.state_vector[1]) if hasattr(track, 'state_estimator') else 0.0,
        'range': float(track.state_estimator.state_vector[0]) if hasattr(track, 'state_estimator') else 0.0,
    }
