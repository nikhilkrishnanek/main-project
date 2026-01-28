import numpy as np
from tracking.kalman import KalmanFilter
from tracking.manager import TrackManager, TrackState

def test_kalman_ca_prediction():
    """Test if CA model predicts motion correctly."""
    dt = 1.0
    kf = KalmanFilter(dt=dt, process_noise=0.1)
    
    # Initial state: r=100, v=10, a=2
    kf.x = np.array([100.0, 10.0, 2.0])
    kf.predict()
    
    # Expected: r = 100 + 10*1 + 0.5*2*1^2 = 111
    # Expected: v = 10 + 2*1 = 12
    # Expected: a = 2
    assert np.isclose(kf.x[0], 111.0)
    assert np.isclose(kf.x[1], 12.0)
    assert np.isclose(kf.x[2], 2.0)

def test_mahalanobis_gate():
    """Test if Mahalanobis distance correctly gates measurements."""
    dt = 0.1
    kf = KalmanFilter(dt=dt)
    kf.x = np.array([100.0, 10.0, 0.0])
    kf.predict()
    
    # Measurement close to prediction
    z_good = np.array([101.0, 10.0])
    d_good = kf.mahalanobis_distance(z_good)
    
    # Measurement far from prediction
    z_bad = np.array([150.0, 50.0])
    d_bad = kf.mahalanobis_distance(z_bad)
    
    assert d_good < d_bad
    assert d_good < 5.0 # Reasonable gate

def test_track_lifecycle():
    """Test state transitions: Provisional -> Confirmed -> Coasting -> Deleted."""
    dt = 0.1
    manager = TrackManager(dt=dt)
    
    # 1. Initiation
    manager.update([(100.0, 10.0)])
    assert len(manager.tracks) == 1
    assert manager.tracks[0].state == TrackState.PROVISIONAL
    
    # 2. Confirmation (needs 3 consecutive hits)
    manager.update([(100.1, 10.0)]) # Hit 2
    manager.update([(100.2, 10.0)]) # Hit 3
    assert manager.tracks[0].state == TrackState.CONFIRMED
    
    # 3. Coasting (Missed detection)
    manager.update([]) # Miss 1
    assert manager.tracks[0].state == TrackState.COASTING
    
    # 4. Re-acquisition
    manager.update([(100.4, 10.0)]) # Hit back
    assert manager.tracks[0].state == TrackState.CONFIRMED

def test_track_deletion():
    """Test if tracks are deleted after too many misses."""
    dt = 0.1
    manager = TrackManager(dt=dt, max_coast_frames=10)
    
    # Provisional deletion (2 misses)
    manager.update([(500.0, 0.0)])
    manager.update([])
    manager.update([])
    assert len(manager.tracks) == 0
    
    # Confirmed/Coasting deletion (10 misses)
    manager.update([(100.0, 0.0)])
    manager.update([(100.1, 0.0)])
    manager.update([(100.2, 0.0)]) # Now confirmed
    
    for _ in range(10):
        manager.update([])
    
    assert len(manager.tracks) == 0

if __name__ == "__main__":
    print("Running Tracking V2 Unit Tests...")
    try:
        test_kalman_ca_prediction()
        print("[PASS] test_kalman_ca_prediction")
        test_mahalanobis_gate()
        print("[PASS] test_mahalanobis_gate")
        test_track_lifecycle()
        print("[PASS] test_track_lifecycle")
        test_track_deletion()
        print("[PASS] test_track_deletion")
        print("\nAll Tracking V2 tests PASSED!")
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
