"""
Radar Scenarios Module
=====================

Provides realistic operational presets for the Photonic Radar.
Generates complex multi-target environments and clutter/jamming conditions.

Scenarios:
1. Drone Swarm (Range Resolution test)
2. Highway Traffic (Doppler test)
3. Sea Skimmer (Clutter test)
4. Stealth Intruder (Sensitivity test)
5. Electronic Warfare (Jamming test)

Author: Principal Radar Scientist
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

from photonic.environment import Target, ChannelConfig
from photonic.noise import NoiseConfig

@dataclass
class Scenario:
    """Encapsulates a complete operational environment."""
    name: str
    description: str
    targets: List[Target]
    channel_config: ChannelConfig
    noise_config: NoiseConfig

class ScenarioGenerator:
    """Factory for radar scenarios."""
    
    @staticmethod
    def _generate_clutter(
        n_scatterers: int, 
        range_min: float, 
        range_max: float, 
        rcs_mean: float, 
        rcs_std: float,
        velocity_max: float = 0.5
    ) -> List[Target]:
        """Generates random clutter points (e.g., waves, trees)."""
        clutter = []
        rng = np.random.default_rng()
        
        for _ in range(n_scatterers):
            r = rng.uniform(range_min, range_max)
            v = rng.uniform(-velocity_max, velocity_max) # Slow movement
            rcs = rng.normal(rcs_mean, rcs_std)
            
            clutter.append(Target(
                range_m=r,
                velocity_m_s=v,
                rcs_dbsm=rcs,
                description="Clutter"
            ))
        return clutter

    @staticmethod
    def get_drone_swarm() -> Scenario:
        """Scenario 1: Close-range multi-target saturation."""
        targets = [
            Target(150.0, 15.0, -10.0, "Drone Alpha"),
            Target(155.0, 18.0, -10.0, "Drone Beta"),  # Close in range
            Target(160.0, 12.0, -10.0, "Drone Gamma"),
            Target(1000.0, 5.0, 10.0, "Mothership"),
        ]
        return Scenario(
            name="Drone Swarm",
            description="Multiple small UAVs maximizing range resolution challenge.",
            targets=targets,
            channel_config=ChannelConfig(noise_level_db=-60.0),
            noise_config=NoiseConfig(rin_db_hz=-155.0)
        )

    @staticmethod
    def get_ground_combat() -> Scenario:
        """Scenario 2: Ground Combat Support."""
        targets = [
            Target(500.0, 30.0, 10.0, "Light Tactical Vehicle (LTV)"),
            Target(450.0, -25.0, 15.0, "Armored Personnel Carrier"),
            Target(200.0, 1.5, -5.0, "Infantry Squad"),
            Target(300.0, 0.0, 20.0, "Forward Operating Base"),
        ]
        return Scenario(
            name="Ground Combat",
            description="Mixed tactical vehicle deployment with infantry.",
            targets=targets,
            channel_config=ChannelConfig(noise_level_db=-55.0),
            noise_config=NoiseConfig()
        )

    @staticmethod
    def get_sea_skimmer() -> Scenario:
        """Scenario 3: Missile amidst heavy sea clutter."""
        # 1. The Missile
        targets = [Target(3000.0, 250.0, 5.0, "Sea Skimmer Missile")]
        
        # 2. Add Clutter (Waves)
        # 50 random scatterers near the missile
        clutter = ScenarioGenerator._generate_clutter(
            n_scatterers=50, 
            range_min=2000, 
            range_max=4000, 
            rcs_mean=-20, 
            rcs_std=5
        )
        targets.extend(clutter)
        
        return Scenario(
            name="Sea Skimmer",
            description="High-speed target embedded in dense sea clutter.",
            targets=targets,
            channel_config=ChannelConfig(noise_level_db=-50.0), # Higher environmental noise
            noise_config=NoiseConfig()
        )

    @staticmethod
    def get_stealth_intruder() -> Scenario:
        """Scenario 4: Low RCS target detection."""
        targets = [
            Target(5000.0, 150.0, -25.0, "Stealth Jet") # Very low RCS
        ]
        return Scenario(
            name="Stealth Intruder",
            description="Long-range, extremely low RCS target.",
            targets=targets,
            channel_config=ChannelConfig(noise_level_db=-70.0), # Quiet night
            noise_config=NoiseConfig(rin_db_hz=-160.0) # High performance radar needed
        )

    @staticmethod
    def get_electronic_warfare() -> Scenario:
        """Scenario 5: Active Jamming."""
        targets = [
            Target(2000.0, 100.0, 10.0, "Hostile Fighter")
        ]
        # Jamming modeled as massively elevated noise floor
        return Scenario(
            name="Electronic Warfare",
            description="Target environment under active noise jamming (+30dB noise).",
            targets=targets,
            channel_config=ChannelConfig(noise_level_db=-20.0), # Jamming!
            noise_config=NoiseConfig()
        )

    @classmethod
    def list_scenarios(cls) -> Dict[str, str]:
        """Returns map of Scenario Name -> Description."""
        return {
            "Drone Swarm": "Close proximity micro-UAVs",
            "Ground Combat": "Tactical ground vehicle tracking",
            "Sea Skimmer": "Mach 0.8 missile in sea clutter",
            "Stealth Intruder": "Low-RCS detection limit test",
            "Electronic Warfare": "High-noise jamming environment"
        }

    @classmethod
    def load(cls, name: str) -> Scenario:
        """Factory method to load by name."""
        map_ = {
            "Drone Swarm": cls.get_drone_swarm,
            "Ground Combat": cls.get_ground_combat,
            "Sea Skimmer": cls.get_sea_skimmer,
            "Stealth Intruder": cls.get_stealth_intruder,
            "Electronic Warfare": cls.get_electronic_warfare
        }
        if name in map_:
            return map_[name]()
        raise ValueError(f"Unknown scenario: {name}")
