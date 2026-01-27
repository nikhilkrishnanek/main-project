import time
import numpy as np
from simulation_engine.orchestrator import SimulationOrchestrator
from simulation_engine.physics import TargetState

def verify_sim_loop():
    print("🚀 Verifying Real-Time Radar Simulation Engine...")
    
    # Configuration
    radar_config = {
        'fs': 1e6,
        'n_pulses': 64,
        'samples_per_pulse': 512,
        'frame_dt': 0.1, # 100ms
        'n_fft_range': 512,
        'n_fft_doppler': 64,
        'window_type': 'hann'
    }
    
    # Targets: Drone and Missile
    initial_targets = [
        TargetState(id=1, position_m=500, velocity_m_s=15, acceleration_m_s2=0, type="drone", maneuver_type="sinusoidal"),
        TargetState(id=2, position_m=2000, velocity_m_s=800, acceleration_m_s2=2, type="missile", maneuver_type="linear")
    ]
    
    orchestrator = SimulationOrchestrator(radar_config, initial_targets)
    
    print("Running 5-frame simulation loop...")
    for frame_data in orchestrator.run_loop(max_frames=5):
        frame_idx = frame_data["frame"]
        metrics = frame_data["metrics"]["averages"]
        
        print(f"\n[FRAME {frame_idx}]")
        print(f"  - Targets: {len(frame_data['targets'])}")
        print(f"  - Tracks: {len(frame_data['tracks'])}")
        print(f"  - Total Latency: {metrics['total_ms']}ms")
        print(f"  - Effective FPS: {metrics['effective_fps']}")
        
        # Verify real-time constraint (100ms budget)
        assert metrics['total_ms'] < 100.0, f"Latency violation on frame {frame_idx}!"
        
    print("\n✅ Simulation Engine verified. All frames within real-time budget.")

if __name__ == "__main__":
    try:
        verify_sim_loop()
    except Exception as e:
        print(f"\n❌ Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
