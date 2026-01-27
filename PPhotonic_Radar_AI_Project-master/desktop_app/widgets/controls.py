
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QComboBox, 
                               QLabel, QGroupBox, QFormLayout)
from PySide6.QtCore import Signal, Slot

class ControlPanel(QWidget):
    scenario_changed = Signal(str)
    stop_resume = Signal(bool) # True = Stop, False = Resume

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Scenario Selection
        group_scenario = QGroupBox("Scenario Control")
        form_layout = QFormLayout()
        self.combo_scenarios = QComboBox()
        # Must match keys in ScenarioGenerator.list_scenarios()
        self.combo_scenarios.addItems([
            "Drone Swarm", 
            "Ground Combat", 
            "Sea Skimmer", 
            "Stealth Intruder", 
            "Electronic Warfare"
        ])
        self.combo_scenarios.currentTextChanged.connect(self._on_scenario_change)
        form_layout.addRow("Load Scenario:", self.combo_scenarios)
        group_scenario.setLayout(form_layout)
        layout.addWidget(group_scenario)
        
        # Simulation Control
        self.btn_pause = QPushButton("PAUSE SIMULATION")
        self.btn_pause.setCheckable(True)
        self.btn_pause.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 10px;")
        self.btn_pause.clicked.connect(self._on_pause_toggle)
        layout.addWidget(self.btn_pause)
        
        # Telemetry
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_targets = QLabel("Active Targets: --")
        self.lbl_status = QLabel("System Status: OPTIMAL")
        self.lbl_status.setStyleSheet("color: #00ff00;")
        
        telem_group = QGroupBox("Telemetry")
        telem_layout = QVBoxLayout()
        telem_layout.addWidget(self.lbl_fps)
        telem_layout.addWidget(self.lbl_targets)
        telem_layout.addWidget(self.lbl_status)
        telem_group.setLayout(telem_layout)
        layout.addWidget(telem_group)
        
        layout.addStretch()

    def _on_scenario_change(self, text):
        self.scenario_changed.emit(text)

    def _on_pause_toggle(self, checked):
        if checked:
            self.btn_pause.setText("RESUME SIMULATION")
            self.btn_pause.setStyleSheet("background-color: #dd0000; color: white; font-weight: bold; padding: 10px;")
        else:
            self.btn_pause.setText("PAUSE SIMULATION")
            self.btn_pause.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 10px;")
        self.stop_resume.emit(checked)
        
    @Slot(dict)
    def update_telemetry(self, frame_data):
        metrics = frame_data.get('metrics', {})
        fps = metrics.get('fps', 0.0)
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        
        targets = frame_data.get('targets', [])
        self.lbl_targets.setText(f"Active Targets: {len(targets)}")
