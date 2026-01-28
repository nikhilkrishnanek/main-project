"""
Multi-Target Track Manager
==========================

Handles the lifecycle of multiple radar tracks using a multi-state machine
and Mahalanobis-based Global Nearest Neighbor (GNN) association.

Track States:
- PROVISIONAL: Initial detection, needs N consecutive hits to confirm.
- CONFIRMED: High-confidence track.
- COASTING: Target lost, predicting state based on motion model (Re-acquisition).

Author: Principal Radar Tracking Scientist
"""

import numpy as np
from enum import Enum
from typing import List, Dict, Optional, Tuple
from tracking.kalman import KalmanFilter

class TrackState(Enum):
    PROVISIONAL = 1
    CONFIRMED = 2
    COASTING = 3
    DELETED = 4

class RadarTrack:
    def __init__(self, track_id: int, initial_meas: np.ndarray, dt: float, max_coast: int = 20):
        self.track_id = track_id
        self.max_coast = max_coast
        self.kf = KalmanFilter(dt=dt)
        # Initialize state: [range, velocity, 0]
        self.kf.x = np.array([initial_meas[0], initial_meas[1], 0.0])
        
        self.state = TrackState.PROVISIONAL
        self.age = 1
        self.hits = 1
        self.misses_total = 0
        self.consecutive_misses = 0
        self.consecutive_hits = 1
        
    def predict(self):
        self.age += 1
        return self.kf.predict()
        
    def update(self, z: np.ndarray):
        """Successful association update."""
        self.hits += 1
        self.consecutive_hits += 1
        self.consecutive_misses = 0
        
        # State Transition: Provisional -> Confirmed
        if self.state == TrackState.PROVISIONAL and self.consecutive_hits >= 3:
            self.state = TrackState.CONFIRMED
        
        # State Transition: Coasting -> Confirmed
        if self.state == TrackState.COASTING:
            self.state = TrackState.CONFIRMED

        return self.kf.update(z)
        
    def mark_miss(self):
        """Handle missed detection (Coasting)."""
        self.consecutive_misses += 1
        self.misses_total += 1
        self.consecutive_hits = 0
        
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.COASTING
            
        # Deletion logic
        if self.state == TrackState.PROVISIONAL and self.consecutive_misses >= 2:
            self.state = TrackState.DELETED
        elif self.state == TrackState.COASTING and self.consecutive_misses >= self.max_coast:
            self.state = TrackState.DELETED

    @property
    def active(self) -> bool:
        return self.state != TrackState.DELETED

class TrackManager:
    def __init__(self, dt: float, association_gate: float = 3.5, max_coast_frames: int = 20):
        """
        Args:
            dt: Sampling time.
            association_gate: Mahalanobis distance threshold (Standard deviations).
            max_coast_frames: Number of frames to hold track without detection.
        """
        self.dt = dt
        self.tracks: List[RadarTrack] = []
        self.next_id = 1
        self.association_gate = association_gate 
        self.max_coast_frames = max_coast_frames

    def update(self, detections: List[Tuple[float, float]]) -> List[Dict]:
        """
        Orchestrates the tracking cycle: Predict -> Associate -> Update -> Manage.
        """
        # 1. Prediction for all existing tracks
        for track in self.tracks:
            track.predict()
            
        # 2. Data Association (GNN using Mahalanobis Distance)
        unassigned_detections = list(range(len(detections)))
        unassigned_tracks = list(range(len(self.tracks)))
        assignments = []
        
        if self.tracks and detections:
            # Build cost matrix (Mahalanobis Distance)
            costs = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    costs[i, j] = track.kf.mahalanobis_distance(np.array(det))
            
            # GNN optimal assignment (Greedy approach for real-time performance)
            while unassigned_tracks and unassigned_detections:
                i_min, j_min = np.unravel_index(np.argmin(costs, axis=None), costs.shape)
                min_cost = costs[i_min, j_min]
                
                if min_cost < self.association_gate:
                    assignments.append((i_min, j_min))
                    costs[i_min, :] = np.inf
                    costs[:, j_min] = np.inf
                    unassigned_tracks.remove(i_min)
                    unassigned_detections.remove(j_min)
                else:
                    break # Gate exceeded
                    
        # 3. Update assigned tracks
        for track_idx, det_idx in assignments:
            self.tracks[track_idx].update(np.array(detections[det_idx]))
            
        # 4. Handle missed detections (Coasting/Deletion)
        for track_idx in unassigned_tracks:
            self.tracks[track_idx].mark_miss()
            
        # 5. Handle new detections (Provisional initiation)
        for det_idx in unassigned_detections:
            new_track = RadarTrack(self.next_id, np.array(detections[det_idx]), self.dt, self.max_coast_frames)
            self.tracks.append(new_track)
            self.next_id += 1
            
        # 6. Cleanup
        self.tracks = [t for t in self.tracks if t.active]
        if len(self.tracks) > 50: # HARD LIMIT to prevent lag in extreme scenarios
            self.tracks = sorted(self.tracks, key=lambda x: x.hits, reverse=True)[:50]
        
        # 7. Format Output Summary
        return [
                {
                    "id": t.track_id,
                    "state": t.state.name,
                    "range_m": float(t.kf.x[0]),
                    "velocity_m_s": float(t.kf.x[1]),
                    "acceleration_m_s2": float(t.kf.x[2]),
                    "confidence": self._calculate_confidence(t),
                    "age": t.age
                }
                for t in self.tracks if t.state in [TrackState.CONFIRMED, TrackState.COASTING]
            ]

    def _calculate_confidence(self, track: RadarTrack) -> float:
        """Heuristic confidence based on hits and state."""
        if track.state == TrackState.CONFIRMED:
            base = 0.9
        elif track.state == TrackState.COASTING:
            base = 0.5 - (track.consecutive_misses * 0.05)
        else:
            base = 0.3
            
        # Boost based on longevity
        boost = min(0.1, track.hits * 0.01)
        return float(np.clip(base + boost, 0.1, 0.99))
