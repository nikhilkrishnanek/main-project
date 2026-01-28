"""
Real-Time Radar Orchestrator
============================

Manages the continuous simulation loop.
Coordinates:
1. Target positioning (Physics).
2. Photonic signal generation.
3. Signal processing (Range-Doppler).
4. AI Classification.
5. Track Management (Kalman).

Author: Simulation Engineer
"""

import time
import numpy as np
from typing import List, Dict
from simulation_engine.physics import TargetState, KinematicEngine
from simulation_engine.performance import PerformanceMonitor
from photonic.signals import generate_synthetic_photonic_signal
from signal_processing.engine import RadarDSPEngine
from signal_processing.detection import ca_cfar_detector, cluster_and_centroid_detections
from ai_models.architectures import TacticalHybridClassifier, initialize_tactical_model
from tracking.manager import TacticalTrackManager
from simulation_engine.evaluation import EvaluationManager
import torch
import torch.nn.functional as F

class SimulationOrchestrator:
    def __init__(self, radar_config: Dict, initial_targets: List[TargetState]):
        self.config = radar_config
        self.dt = radar_config.get('frame_dt', 0.1)
        self.physics = KinematicEngine(self.dt)
        self.dsp = RadarDSPEngine(radar_config)
        self.tracker = TacticalTrackManager(sampling_period_s=self.dt)
        self.perf = PerformanceMonitor()
        
        # AI Logic
        self.ai_model = initialize_tactical_model(num_target_classes=5)
        self.ai_model.eval()
        # Mock class labels
        self.class_labels = {0: "Noise", 1: "Drone", 2: "Bird", 3: "Aircraft", 4: "Missile"}
        
        self.targets = initial_targets
        self.frame_count = 0
        self.is_running = False
        
        # Evaluation
        self.eval_manager = EvaluationManager()

        # Scanning State
        self.scan_angle_deg = 0.0
        self.rpm = radar_config.get('rpm', 12.0) # 12 RPM default
        self.beamwidth_deg = radar_config.get('beamwidth_deg', 5.0)

    def _prepare_spectrogram(self, rd_map: np.ndarray) -> torch.Tensor:
        """Prepares RD map for CNN input (Reset to 128x128)."""
        # 1. Log modulus if not already
        if np.iscomplexobj(rd_map):
            rd_abs = np.abs(rd_map)
        else:
            rd_abs = rd_map
        
        # 2. Normalize
        rd_norm = (rd_abs - np.min(rd_abs)) / (np.max(rd_abs) + 1e-9)
        
        # 3. To Tensor (1, 1, H, W)
        tensor = torch.from_numpy(rd_norm).float().unsqueeze(0).unsqueeze(0)
        
        # 4. Interpolate to 128x128 (Model Requirement)
        return F.interpolate(tensor, size=(128, 128), mode='bilinear', align_corners=False)

    def _prepare_timeseries(self, velocity: float) -> torch.Tensor:
        """Synthesizes Doppler time-series input based on track velocity."""
        # Seq len 1000
        t = torch.linspace(0, 1, 1000)
        # Doppler shift proxy (just a frequency tone)
        freq = 10.0 + abs(velocity) # Base offset
        signal = torch.sin(2 * torch.pi * freq * t) + 0.1 * torch.randn(1000)
        return signal.unsqueeze(0) # (1, 1000)

    def tick(self) -> Dict:
        """
        Executes a real-time frame cycle: Physics -> Photonic -> DSP -> AI -> Track.
        """
        start_time = time.time()
        self.perf.start_phase("total")
        
        # 1. Physics Update (Advance all targets)
        self.perf.start_phase("physics")
        self.targets = [self.physics.update_state(t) for t in self.targets]
        self.perf.end_phase("physics")
        
        # 2. Scanning Update
        # Advance scan angle: omega = 360 * RPM / 60 = 6 * RPM (deg/s)
        # delta_deg = omega * dt
        scan_omega = 6.0 * self.rpm
        self.scan_angle_deg = (self.scan_angle_deg + scan_omega * self.dt) % 360.0
        
        # Determine Illuminated Targets
        illuminated_targets = []
        for t in self.targets:
            # Check angular difference
            az = t.azimuth_deg % 360.0
            beam_delta = abs(az - self.scan_angle_deg)
            # Handle wrap-around (e.g. 359 vs 1)
            if beam_delta > 180: beam_delta = 360 - beam_delta
            
            if beam_delta < (self.beamwidth_deg / 2.0):
                illuminated_targets.append(t)
        
        # 3. Photonic Signal Generation (Analytic De-chirped)
        self.perf.start_phase("photonic")
        sampling_rate_hz = self.config.get('sampling_rate_hz', 2e6)
        num_pulses = self.config.get('n_pulses', 64)
        samples_per_pulse = self.config.get('samples_per_pulse', 512)
        
        speed_of_light = 3e8
        carrier_freq_hz = self.config.get('start_frequency_hz', 77e9) 
        sweep_bandwidth_hz = self.config.get('sweep_bandwidth_hz', 150e6)
        chirp_duration_s = samples_per_pulse / sampling_rate_hz
        chirp_slope_hz_s = sweep_bandwidth_hz / chirp_duration_s
        
        total_samples = num_pulses * samples_per_pulse
        time_vector = np.arange(total_samples) / sampling_rate_hz
        beat_signal = np.zeros(total_samples, dtype=complex)
        
        # Helper: Pulse time vector for phase reset if needed
        t_pulse = np.arange(samples_per_pulse) / sampling_rate_hz
        t_matrix = np.tile(t_pulse, num_pulses)
        
        if illuminated_targets:
            for tgt in illuminated_targets:
                r = tgt.range_m
                v = tgt.radial_velocity
                
                # Physics:
                propagation_delay_tau = 2 * r / speed_of_light
                beat_frequency_hz = chirp_slope_hz_s * propagation_delay_tau
                doppler_frequency_hz = (2 * v * carrier_freq_hz) / speed_of_light
                
                # Phase: 2*pi*(fb + fd)*t + Phase_offset
                # Important: t must be local to pulse for Range term?
                # Actually for de-chirped:
                # Phase ~ 2*pi*( (S*tau + f_d)*t + fc*tau - 0.5*S*tau^2 )
                # The term 't' here resets every pulse for the S*tau part?
                # Yes, S*t is freq offset from current chirp start.
                
                total_phase = 2 * np.pi * carrier_freq_hz * propagation_delay_tau
                
                # Amplitude (Radar Equation scaling)
                amplitude = 1.0 / (r**2 + 1.0) * 1e5
                
                # Generate Tone
                tone = amplitude * np.exp(1j * 2 * np.pi * beat_frequency_hz * t_matrix) * \
                       np.exp(1j * 2 * np.pi * doppler_frequency_hz * time_vector) * \
                       np.exp(1j * total_phase)
                       
                beat_signal += tone
        
        # Add Noise (WDM/Thermal)
        noise_pwr = 10**(self.config.get('noise_level_db', -50)/10)
        beat_signal += np.random.normal(0, np.sqrt(noise_pwr), total_samples) + \
                       1j*np.random.normal(0, np.sqrt(noise_pwr), total_samples)
                       
        if self.frame_count % 10 == 0 and illuminated_targets:
            print(f"[DEBUG] Frame {self.frame_count}: {len(illuminated_targets)} targets illuminated.")
            print(f"        Beat Signal Max amp: {np.max(np.abs(beat_signal)):.4e}")
            
        self.perf.end_phase("photonic")
        
        # 3. DSP & Detection
        self.perf.start_phase("dsp")
        pulse_matrix = beat_signal.reshape(num_pulses, samples_per_pulse)
        rd_map, rd_power = self.dsp.process_frame(pulse_matrix) # Unpack dB and Linear maps
        det_map, _ = ca_cfar_detector(rd_power, power_floor=0.005) 
        
        detections = cluster_and_centroid_detections(det_map, rd_power) # Collapse redundant peaks
        
        if self.frame_count % 50 == 0:
            # Periodic debug only
            pass
            
        self.perf.end_phase("dsp")
        
        # 4. AI & Tracking
        self.perf.start_phase("ai_tracking")
        
        # Calculate Resolutions
        n_fft_r = self.dsp.n_fft_range
        r_scale = (speed_of_light * samples_per_pulse) / (2 * sweep_bandwidth_hz * n_fft_r)
        
        # Velocity Res
        n_fft_d = self.dsp.n_fft_doppler
        wavelength = speed_of_light / carrier_freq_hz
        v_scale = wavelength / (2 * num_pulses * chirp_duration_s * (n_fft_d / num_pulses))
        
        obs_states = []
        for v_idx, r_idx in detections:
            # Map indices to physical units
            # v_idx is Doppler (Dim 0), r_idx is Range (Dim 1)
            
            r = r_idx * r_scale
            v_bin_centered = v_idx - (n_fft_d // 2)
            # Check sign logic: if target approaching, f_dopp > 0.
            # In FFT shift, right side is positive freq.
            v = v_bin_centered * v_scale
            
            # Simple threshold to filter unrealistic ranges (close to 0 usually DC leakage)
            if r > 10: 
                obs_states.append((r, v))
            
        tracks = self.tracker.update_pipeline(obs_states)
        
        # DEBUG: tracing the data flow
        if self.frame_count % 10 == 0:
            print(f"[DEBUG] Frame {self.frame_count}:")
            print(f"        Illuminated: {len(illuminated_targets)}")
            print(f"        Detections:  {len(detections)} (Raw)")
            print(f"        Valid Obs:   {len(obs_states)} (After filtering)")
            print(f"        Tracks:      {len(tracks)}")
        
        # --- AI INJECTION START ---
        # Critical Fix: Run Inference on confirmed tracks
        if tracks:
            with torch.no_grad():
                # Prepare batch
                spec_batch = self._prepare_spectrogram(rd_map) # Shared scene context
                
                for tr in tracks:
                    # In a real system, we'd crop the ROI per target. 
                    # Here we pass the full scene context + target-specific kinematic time-series.
                    ts_input = self._prepare_timeseries(tr['estimated_velocity_ms'])
                    
                    logits, _ = self.ai_model(spec_batch, ts_input)
                    probs = F.softmax(logits, dim=1)
                    conf, class_idx = torch.max(probs, dim=1)
                    
                    # Attach to track object
                    tr['class_id'] = int(class_idx.item())
                    tr['class_label'] = self.class_labels.get(tr['class_id'], "Unknown")
                    tr['confidence'] = float(conf.item())
        # --- AI INJECTION END ---

        self.perf.end_phase("ai_tracking")
        
        # 5. Evaluation (Offline/Online Analysis)
        # Convert tracks to dict format for evaluator
        track_dicts = []
        for tr in tracks:
             # Basic transformation for eval matching
             track_dicts.append({
                 'id': tr['id'],
                 'range_m': tr['estimated_range_m'],
                 'velocity_m_s': tr['estimated_velocity_ms']
             })
             
        # Ground Truth (All targets, even those outside beam, for global tracking eval? 
        # Or only illuminated? Usually we want to know if we missed something existing)
        # Let's pass all active targets as Ground Truth
        gt_dicts = [vars(t) for t in self.targets]
        self.eval_manager.update(self.frame_count, gt_dicts, track_dicts)
        
        self.perf.end_phase("total")
        self.frame_count += 1
        
        return {
            "frame": self.frame_count,
            "timestamp": time.time(),
            "targets": [vars(t) for t in self.targets], # Send all targets for ground truth debug
            "scan_angle": self.scan_angle_deg,
            "illuminated_ids": [t.id for t in illuminated_targets],
            "rd_map": rd_map,
            "tracks": tracks,
            "metrics": self.perf.get_metrics(),
            "evaluation": self.eval_manager.get_summary()
        }

    def run_loop(self, max_frames: int = 100):
        """
        Standard blocking loop for testing.
        In streamlit/UI, this would be handled via a generator.
        """
        self.is_running = True
        try:
            for _ in range(max_frames):
                if not self.is_running: break
                frame_data = self.tick()
                yield frame_data
                
                # Maintain real-time pace
                elapsed = time.time() - frame_data["timestamp"]
                sleep_time = max(0, self.dt - elapsed)
                time.sleep(sleep_time)
        finally:
            self.is_running = False

    def stop(self):
        self.is_running = False
