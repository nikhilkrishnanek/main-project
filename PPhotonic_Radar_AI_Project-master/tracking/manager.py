"""
Tactical Track Management and Data Association
===============================================

This module orchestrates the lifecycle of multiple target tracks using 
a multi-state logic controller and Global Nearest Neighbor (GNN) association.

Track State Transition Logic:
-----------------------------
1. PROVISIONAL: Newly initiated track from a single detection. Requires 
   M-of-N confirmation (e.g., 3 consecutive hits) to become CONFIRMED.
2. CONFIRMED: High-confidence track with active measurement updates.
3. COASTING: Temporary loss of detection; state is predicted using the 
   last known kinematics until either re-acquired or pruned.
4. DELETED: Track removed from the tactical inventory due to persistent loss.

Association Strategy:
---------------------
GNN association is performed using the Mahalanobis distance, which accounts 
for both the spatial residual and the filter's current estimation uncertainty.

Author: Senior Tracking & Fusion Architect
"""

import numpy as np
from enum import Enum
from typing import List, Dict, Optional, Tuple
from tracking.kalman import KinematicKalmanFilter


class TacticalTrackState(Enum):
    """Enumeration of valid states in the track lifecycle."""
    PROVISIONAL = 1
    CONFIRMED   = 2
    COASTING    = 3
    DELETED     = 4


class KinematicTrack:
    """
    Represents a single tactical entity being tracked by the system.
    """
    def __init__(self, 
                 track_id: int, 
                 initial_measurement: np.ndarray, 
                 sampling_period_s: float, 
                 max_coasting_frames: int = 20):
        self.track_id = track_id
        self.max_coasting_frames = max_coasting_frames
        
        # Core State Estimator
        self.state_estimator = KinematicKalmanFilter(sampling_period_s=sampling_period_s)
        
        # Seed the filter with initial position and velocity
        # [Range, Velocity, Acceleration]
        self.state_estimator.state_vector = np.array([initial_measurement[0], initial_measurement[1], 0.0])
        
        # Lifecycle Management
        self.current_state = TacticalTrackState.PROVISIONAL
        self.track_age_frames = 1
        self.total_detections_count = 1
        self.coasting_frame_count = 0
        self.consecutive_detection_count = 1
        
    def predict_next_state(self):
        """Advances the track kinematics using the internal motion model."""
        self.track_age_frames += 1
        return self.state_estimator.predict()
        
    def synchronize_with_measurement(self, observation_vector: np.ndarray):
        """Updates the track state with a confirmed tactical measurement."""
        self.total_detections_count += 1
        self.consecutive_detection_count += 1
        self.coasting_frame_count = 0
        
        # Promotion: Provisional -> Confirmed
        if self.current_state == TacticalTrackState.PROVISIONAL and self.consecutive_detection_count >= 3:
            self.current_state = TacticalTrackState.CONFIRMED
        
        # Re-acquisition: Coasting -> Confirmed
        if self.current_state == TacticalTrackState.COASTING:
            self.current_state = TacticalTrackState.CONFIRMED

        return self.state_estimator.update(observation_vector)
        
    def handle_missed_detection(self):
        """Processes a frame where no measurement was associated with this track."""
        self.coasting_frame_count += 1
        self.consecutive_detection_count = 0
        
        # Enter Coasting mode if a confirmed track is lost
        if self.current_state == TacticalTrackState.CONFIRMED:
            self.current_state = TacticalTrackState.COASTING
            
        # Deletion logic (Pruning)
        if self.current_state == TacticalTrackState.PROVISIONAL and self.coasting_frame_count >= 2:
            self.current_state = TacticalTrackState.DELETED
        elif self.current_state == TacticalTrackState.COASTING and self.coasting_frame_count >= self.max_coasting_frames:
            self.current_state = TacticalTrackState.DELETED

    @property
    def is_active(self) -> bool:
        """Returns True if the track is still part of the tactical inventory."""
        return self.current_state != TacticalTrackState.DELETED


class TacticalTrackManager:
    """
    Controller for multi-target tracking and data association.
    """
    def __init__(self, 
                 sampling_period_s: float, 
                 association_gate_sigma: float = 3.5, 
                 max_coast_interval: int = 20):
        """
        Args:
            sampling_period_s: Radar frame interval.
            association_gate_sigma: Mahalanobis distance threshold (gate).
            max_coast_interval: Maximum frames to coast a lost track.
        """
        self.dt = sampling_period_s
        self.active_tracks: List[KinematicTrack] = []
        self.global_track_id_counter = 1
        self.association_gate_sigma = association_gate_sigma 
        self.max_coast_interval = max_coast_interval

    def update_pipeline(self, current_detections: List[Tuple[float, float]]) -> List[Dict]:
        """
        Executes the cyclical tracking chain: Prediction, Association, Update, and Management.
        """
        # 1. State Prediction (A-priori)
        for track in self.active_tracks:
            track.predict_next_state()
            
        # 2. Data Association (Nearest Neighbor GNN)
        unassigned_det_indices = list(range(len(current_detections)))
        unassigned_track_indices = list(range(len(self.active_tracks)))
        track_to_det_assignments = []
        
        if self.active_tracks and current_detections:
            # Construct Cost Matrix using Mahalanobis Distance
            cost_matrix = np.zeros((len(self.active_tracks), len(current_detections)))
            for i, track in enumerate(self.active_tracks):
                for j, det in enumerate(current_detections):
                    cost_matrix[i, j] = track.state_estimator.calculate_mahalanobis_distance(np.array(det))
            
            # Greedy Association (Efficient for real-time dense environments)
            while unassigned_track_indices and unassigned_det_indices:
                min_flat_idx = np.argmin(cost_matrix)
                i_min, j_min = np.unravel_index(min_flat_idx, cost_matrix.shape)
                cost_value = cost_matrix[i_min, j_min]
                
                if cost_value < self.association_gate_sigma:
                    track_to_det_assignments.append((i_min, j_min))
                    # Mask the assigned row and column
                    cost_matrix[i_min, :] = np.inf
                    cost_matrix[:, j_min] = np.inf
                    unassigned_track_indices.remove(i_min)
                    unassigned_det_indices.remove(j_min)
                else:
                    break # No more detections within the gated region
                    
        # 3. Correct Assigned Tracks with New Measurements
        for track_idx, det_idx in track_to_det_assignments:
            self.active_tracks[track_idx].synchronize_with_measurement(np.array(current_detections[det_idx]))
            
        # 4. Process Missed Tracks (Coasting)
        for track_idx in unassigned_track_indices:
            self.active_tracks[track_idx].handle_missed_detection()
            
        # 5. Initialize New Tracks (Provisional Initiation)
        for det_idx in unassigned_det_indices:
            new_entity = KinematicTrack(self.global_track_id_counter, 
                                       np.array(current_detections[det_idx]), 
                                       self.dt, 
                                       self.max_coast_interval)
            self.active_tracks.append(new_entity)
            self.global_track_id_counter += 1
            
        # 6. Inventory Pruning
        self.active_tracks = [t for t in self.active_tracks if t.is_active]
        
        # Scalability constraint: Keep at most 50 highest-confidence tracks
        if len(self.active_tracks) > 50:
            self.active_tracks = sorted(self.active_tracks, key=lambda x: x.total_detections_count, reverse=True)[:50]
        
        # 7. Generate Output Telemetry
        return [
                {
                    "id": t.track_id,
                    "tactical_state": t.current_state.name,
                    "estimated_range_m": float(t.state_estimator.state_vector[0]),
                    "estimated_velocity_ms": float(t.state_estimator.state_vector[1]),
                    "estimated_acceleration_ms2": float(t.state_estimator.state_vector[2]),
                    "track_confidence_score": self._estimate_track_confidence(t),
                    "track_life_frames": t.track_age_frames
                }
                for t in self.active_tracks if t.current_state in [TacticalTrackState.CONFIRMED, TacticalTrackState.COASTING]
            ]

    def _estimate_track_confidence(self, track: KinematicTrack) -> float:
        """Heuristic calculation of track viability."""
        if track.current_state == TacticalTrackState.CONFIRMED:
            base_score = 0.9
        elif track.current_state == TacticalTrackState.COASTING:
            base_score = 0.5 - (track.coasting_frame_count * 0.05)
        else:
            base_score = 0.3
            
        # Longevity bonus (Credit for persistent tracks)
        longevity_bonus = min(0.1, track.total_detections_count * 0.01)
        return float(np.clip(base_score + longevity_bonus, 0.1, 0.99))
