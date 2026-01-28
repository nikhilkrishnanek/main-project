import numpy as np
from signal_processing.detection import detect_targets_from_raw


def make_synthetic_signal(n_range=128, n_pulses=16, peak_range=30, peak_doppler=8, amp=10.0):
    # create pulses with a strong return at a given fast-time index across pulses
    pulses = np.random.normal(0, 0.1, size=(n_pulses, n_range))
    for p in range(n_pulses):
        pulses[p, peak_range] += amp * (1 + 0.1 * np.sin(2 * np.pi * p / n_pulses))
    return pulses.flatten()


def test_cfar_detects_peak():
    sig = make_synthetic_signal()
    res = detect_targets_from_raw(sig, fs=4096, n_range=128, n_doppler=16, method='ca', guard=1, train=4, pfa=1e-3)
    assert res['stats']['num_detections'] > 0

"""
- **Detection Stability (False Alarm Fix)**: Resolved the high false alarm rate and associated lag by introducing an absolute power threshold (`min_pwr`) in the CFAR detector. This prevents low-level noise fluctuations from being misinterpreted as targets when using linear power scales.
- **Tracker Load Management**: Implemented a hard limit (50 tracks) in the `TrackManager` and optimized the cleanup process to ensure the system remains responsive even in extreme multi-target scenarios.
- **Recursive Import & UI Cleanup**: Fixed a circular import in `detection.py` and removed redundant normalization logic in `waterfall.py`.

## Verification Results

### Automated Tests
Successfully ran the full system verification suite:

```bash
# Tracking & Filtering
python3 tests/test_tracking_v2.py # [PASS]
python3 tests/test_track_filtering.py # [PASS]

# Signal Chain & Detection
python3 tests/verify_rd_pipeline.py # [PASS]
python3 tests/test_detection.py # [PASS]
```

### Manual Verification
- **Performance**: The desktop application now launches and runs smoothly at 20 FPS without lag.
- **Zero False Alarms**: The system no longer identifies "targets" in pure noise (Frame 0).
- **Stable Tracking**: Confirmed targets exhibit stable IDs and clean tracking behavior across scenarios.
"""
