"""
Modular Cognitive Intelligence Bridge
======================================

This module provides a modular interface for integrating the cognitive 
decision engine into various radar pipelines. It manages the lifecycle of 
situation assessment and parameter adaptation in a standalone fashion.

Key Responsibilities:
---------------------
1. Telemetry Conversion: Maps tracking and AI outputs to cognitive observables.
2. State Retention: Maintains historical context for trend analysis.
3. Parameter Synthesis: Translates cognitive commands into hardware-level 
   waveform configurations.

Author: Senior Integration Engineer (Tactical Systems)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

from cognitive.engine import (
    CognitiveRadarEngine,
    SituationAssessment,
    AdaptationCommand,
    create_track_dict_for_cognitive,
)
from cognitive.parameters import (
    AdaptiveParameterManager,
    RadarWaveformParameters,
    convert_config_to_waveform_params,
)

logger = logging.getLogger(__name__)


class ModularCognitiveIntelligenceBridge:
    """
    Standalone bridge for decentralized cognitive radar orchestration.
    """
    
    def __init__(self, enable_autonomous_adaptation: bool = True):
        """
        Initializes the cognitive intelligence bridge.
        """
        self.autonomous_mode = enable_autonomous_adaptation
        self.intelligence_engine = CognitiveRadarEngine() if enable_autonomous_adaptation else None
        self.parameter_manager = AdaptiveParameterManager()
        
        # Operational State
        self.active_parameters = RadarWaveformParameters()
        self.last_assessment = None
        self.last_adaptation_command = None
        self.total_processed_frames = 0
        
        self.logger = logging.getLogger(__name__)
    
    def initialize_tactical_parameters(self, system_config: Dict) -> RadarWaveformParameters:
        """
        Bootstraps the waveform parameters from a system configuration.
        """
        self.active_parameters = convert_config_to_waveform_params(system_config)
        self.parameter_manager.update_cache(0, self.active_parameters)
        
        self.logger.info(
            f"Cognitive Bridge initialized (Autonomous Mode: {self.autonomous_mode})"
        )
        
        return self.active_parameters
    
    def execute_intelligence_cycle(self,
                                  current_detections: List[Tuple[float, float]],
                                  active_tracks: List,
                                  intelligence_reports: List[Dict],
                                  spectral_intensity_map: Optional[np.ndarray] = None,
                                  frame_timestamp: float = 0.0) \
            -> Tuple[RadarWaveformParameters, Optional[AdaptationCommand], str]:
        """
        Executes a complete cognitive loop for a single processing frame.
        """
        self.total_processed_frames += 1
        
        if not self.autonomous_mode:
            return self.active_parameters, None, "[STATIC MODE] Autonomous adaptation disabled."
        
        # 1. Observability Mapping
        # Convert internal KinematicTrack objects to cognitive-compatible telemetry
        tactical_telemetry_tracks = [create_track_dict_for_cognitive(track) for track in active_tracks]
        
        # 2. Situation Assessment
        assessment = self.intelligence_engine.assess_situation(
            frame_id=self.total_processed_frames,
            timestamp=frame_timestamp,
            detections=current_detections,
            tracks=tactical_telemetry_tracks,
            ai_predictions=intelligence_reports,
            rd_map=spectral_intensity_map
        )
        self.last_assessment = assessment
        
        # 3. Cognitive Decision Logic
        adaptation_cmd = self.intelligence_engine.decide_adaptation(assessment)
        self.last_adaptation_command = adaptation_cmd
        
        # 4. Parameter Synthesis
        synthesized_parameters = self.parameter_manager.apply_adaptation_command(
            adaptation_cmd,
            self.active_parameters
        )
        
        # 5. Operational Safety Constraints
        is_safe, constraint_violations = self.parameter_manager.validate_parameters(synthesized_parameters)
        if not is_safe:
            self.logger.warning(f"Cognitive adaptation rejected due to safety constraints: {constraint_violations}")
            synthesized_parameters = self.active_parameters
        
        # 6. Lifecycle Management
        self.parameter_manager.update_cache(self.total_processed_frames, synthesized_parameters)
        self.active_parameters = synthesized_parameters
        
        # 7. XAI Narrative Generation
        narrative = self.intelligence_engine.generate_xai_narrative(assessment, adaptation_cmd)
        
        return synthesized_parameters, adaptation_cmd, narrative
