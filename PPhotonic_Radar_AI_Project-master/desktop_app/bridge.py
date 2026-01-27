
import time
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot, QThread
from simulation_engine.orchestrator import SimulationOrchestrator
from photonic.scenarios import ScenarioGenerator
from simulation_engine.physics import TargetState

class RadarWorker(QObject):
    """
    Simulation Worker living in a separate QThread.
    Uses QTimer for accurate, non-blocking simulation ticking.
    """
    # Signals to GUI
    frame_ready = Signal(dict)
    stopped = Signal()
    error_occurred = Signal(str)

    def __init__(self, scenario_name="Drone Swarm"):
        super().__init__()
        self.scenario_name = scenario_name
        self.orchestrator = None
        self.timer = None
        self._target_dt = 0.05 # Default 20 FPS

    def initialize(self):
        """Called within the worker thread to setup resources."""
        self._load_orchestrator()
        
        # Setup Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._run_tick)
        # Check Config for Frame Rate
        if self.orchestrator:
             self._target_dt = self.orchestrator.config.get('frame_dt', 0.05)
        
    def _load_orchestrator(self):
        try:
            scenario = ScenarioGenerator.load(self.scenario_name)
        except Exception as e:
            print(f"Error loading scenario {self.scenario_name}: {e}")
            scenario = ScenarioGenerator.load("Drone Swarm")
            
        radar_config = {
            "fs": 2e6,
            "n_pulses": 64, 
            "samples_per_pulse": 512,
            "frame_dt": 0.05,
            "rpm": 12.0,
            "beamwidth_deg": 5.0,
            "noise_level_db": scenario.channel_config.noise_level_db
        }
        
        
        sim_targets = []
        for i, t in enumerate(scenario.targets):
            # Optimizing angle for Demo: 
            # Place targets in 10-50 degree sector so scanning beam hits them immediately.
            angle_rad = np.deg2rad(10 + (i * 10) % 40) 
            
            r = t.range_m
            px = r * np.cos(angle_rad)
            py = r * np.sin(angle_rad)
            vx = t.velocity_m_s * np.cos(angle_rad)
            vy = t.velocity_m_s * np.sin(angle_rad)
            sim_targets.append(TargetState(
                id=i+1, pos_x=px, pos_y=py, vel_x=vx, vel_y=vy, type=t.description.lower()
            ))

        self.orchestrator = SimulationOrchestrator(radar_config, sim_targets)

    @Slot()
    def start_sim(self):
        if self.timer and not self.timer.isActive():
            ms = int(self._target_dt * 1000)
            self.timer.start(ms)
            print(f"Simulation Worker started (Interval: {ms}ms)")

    @Slot()
    def stop_sim(self):
        if self.timer and self.timer.isActive():
            self.timer.stop()
            self.stopped.emit()

    @Slot()
    def _run_tick(self):
        if not self.orchestrator:
            return
            
        try:
            # Single Frame Update
            frame_data = self.orchestrator.tick()
            self.frame_ready.emit(frame_data)
        except Exception as e:
            print(f"Checking Error: {e}")
            self.error_occurred.emit(str(e))
            self.stop_sim()
            
    @Slot(str)
    def switch_scenario(self, name):
        was_running = self.timer.isActive()
        self.stop_sim()
        
        self.scenario_name = name
        self._load_orchestrator()
        
        if was_running:
            self.start_sim()
