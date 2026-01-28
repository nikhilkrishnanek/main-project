import numpy as np
from tracking.manager import TrackManager, TrackState

def test_track_output_filtering():
    """Verify that only CONFIRMED or COASTING tracks are returned by update()."""
    dt = 0.1
    manager = TrackManager(dt=dt)
    
    # 1. First detection - should be PROVISIONAL and NOT in output
    tracks = manager.update([(100.0, 10.0)])
    assert len(manager.tracks) == 1
    assert manager.tracks[0].state == TrackState.PROVISIONAL
    assert len(tracks) == 0, "Provisional tracks should not be in output"
    
    # 2. Second detection - still PROVISIONAL
    tracks = manager.update([(100.1, 10.0)])
    assert len(manager.tracks) == 1
    assert manager.tracks[0].state == TrackState.PROVISIONAL
    assert len(tracks) == 0, "Provisional tracks should not be in output"
    
    # 3. Third detection - becomes CONFIRMED
    tracks = manager.update([(100.2, 10.0)])
    assert len(manager.tracks) == 1
    assert manager.tracks[0].state == TrackState.CONFIRMED
    assert len(tracks) == 1, "Confirmed tracks should be in output"
    assert tracks[0]['id'] == 1
    
    # 4. Missed detection - becomes COASTING
    tracks = manager.update([])
    assert len(manager.tracks) == 1
    assert manager.tracks[0].state == TrackState.COASTING
    assert len(tracks) == 1, "Coasting tracks should still be in output"
    
    # 5. New detection - should be PROVISIONAL and NOT in output
    tracks = manager.update([(200.0, 5.0)])
    # We should have one CONFIRMED (the old one) and one PROVISIONAL (the new one)
    # Actually the old one is COASTING.
    assert len(manager.tracks) == 2
    # The first track (ID 1) is COASTING because it was missed in step 4 and then not associated in step 5?
    # Wait, in step 5 we provide a detection at 200m. The old track is at ~100.2m. 
    # Mahalanobis gate will likely block it. 
    # So old track stays COASTING, new detection creates PROVISIONAL.
    
    coasting_tracks = [t for t in tracks if t['id'] == 1]
    provisional_tracks = [t for t in tracks if t['id'] == 2]
    
    assert len(coasting_tracks) == 1, "Old coasting track should be in output"
    assert len(provisional_tracks) == 0, "New provisional track should NOT be in output"

if __name__ == "__main__":
    print("Running Track Filtering Tests...")
    try:
        test_track_output_filtering()
        print("[PASS] test_track_output_filtering")
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
