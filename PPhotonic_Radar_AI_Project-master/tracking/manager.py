"""
Multi-Target Track Manager
==========================

Handles the lifecycle of multiple radar tracks.
Features:
- Global Nearest Neighbor (GNN) data association.
- Persistent ID assignment.
- Track coasting (maintenance through missed detections).
- Automated track deletion (management of stale targets).

Author: Radar Tracking Expert
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from tracking.kalman import KalmanFilter

class RadarTrack:
    def __init__(self, track_id: int, initial_state: np.ndarray, dt: float):
        self.track_id = track_id
        self.kf = KalmanFilter(dt=dt)
        self.kf.x = initial_state # [range, velocity]
        
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.active = True
        
    def predict(self):
        self.age += 1
        return self.kf.predict()
        
    def update(self, z: np.ndarray):
        self.hits += 1
        self.misses = 0
        return self.kf.update(z)
        
    def mark_miss(self):
        self.misses += 1
        if self.misses > 5: # Threshold for deletion
            self.active = False

class TrackManager:
    def __init__(self, dt: float, association_threshold: float = 10.0):
        self.dt = dt
        self.tracks: List[RadarTrack] = []
        self.next_id = 1
        self.association_threshold = association_threshold # Euclidean distance in RD space

    def update(self, detections: List[Tuple[float, float]]) -> List[Dict]:
        """
        Updates trackers with a list of detections [(range, velocity), ...].
        Returns list of active track summaries.
        """
        # 1. Prediction step for all trackers
        for track in self.tracks:
            track.predict()
            
        # 2. Data Association (Nearest Neighbor)
        unassigned_detections = list(range(len(detections)))
        unassigned_tracks = list(range(len(self.tracks)))
        
        assignments = []
        
        if self.tracks and detections:
            # Build cost matrix (Euclidean distance)
            # Row: tracks, Col: detections
            costs = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    dist = np.linalg.norm(track.kf.x - np.array(det))
                    costs[i, j] = dist
            
            # Simplified GNN: Find minimums
            while unassigned_tracks and unassigned_detections:
                # Find the overall minimum distance
                i, j = np.unravel_index(np.argmin(costs, axis=None), costs.shape)
                min_dist = costs[i, j]
                
                if min_dist < self.association_threshold:
                    assignments.append((i, j))
                    # Prevent re-assignment
                    costs[i, :] = np.inf
                    costs[:, j] = np.inf
                    unassigned_tracks.remove(i)
                    unassigned_detections.remove(j)
                else:
                    break # No more close matches
                    
        # 3. Update assigned tracks
        for track_idx, det_idx in assignments:
            self.tracks[track_idx].update(np.array(detections[det_idx]))
            
        # 4. Handle unassigned tracks (Missed detections)
        for track_idx in unassigned_tracks:
            self.tracks[track_idx].mark_miss()
            
        # 5. Handle unassigned detections (New tracks)
        for det_idx in unassigned_detections:
            new_track = RadarTrack(self.next_id, np.array(detections[det_idx]), self.dt)
            self.tracks.append(new_track)
            self.next_id += 1
            
        # 6. Cleanup inactive tracks
        self.tracks = [t for t in self.tracks if t.active]
        
        # 7. Prepare output
        return [
            {
                "id": t.track_id,
                "range_m": t.kf.x[0],
                "velocity_m_s": t.kf.x[1],
                "covariance": t.kf.P.tolist(),
                "age": t.age,
                "confidence": max(0.2, 1.0 - (t.misses * 0.2)) # Heuristic
            }
            for t in self.tracks
        ]
