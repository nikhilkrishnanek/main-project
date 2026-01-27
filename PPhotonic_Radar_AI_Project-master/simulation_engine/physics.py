"""
Kinematic Physics Engine
========================

Simulates target motion using classical mechanics.
Supports:
- Constant Velocity (CV) model.
- Constant Acceleration (CA) model.

Mathematical Model:
p(t+dt) = p(t) + v(t)*dt + 0.5*a(t)*dt^2
v(t+dt) = v(t) + a(t)*dt

Author: Simulation Engineer
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class TargetState:
    position_m: float
    velocity_m_s: float
    acceleration_m_s2: float = 0.0

class KinematicEngine:
    def __init__(self, dt: float):
        self.dt = dt

    def update_state(self, state: TargetState) -> TargetState:
        """
        Calculates the next state based on current kinematics.
        """
        new_pos = state.position_m + (state.velocity_m_s * self.dt) + (0.5 * state.acceleration_m_s2 * (self.dt**2))
        new_vel = state.velocity_m_s + (state.acceleration_m_s2 * self.dt)
        
        return TargetState(
            position_m=new_pos,
            velocity_m_s=new_vel,
            acceleration_m_s2=state.acceleration_m_s2
        )

def simulate_trajectory(initial_state: TargetState, duration_s: float, dt: float) -> list[TargetState]:
    """
    Generates a full trajectory over a specified duration.
    """
    engine = KinematicEngine(dt)
    trajectory = [initial_state]
    current_state = initial_state
    
    n_steps = int(duration_s / dt)
    for _ in range(n_steps):
        current_state = engine.update_state(current_state)
        trajectory.append(current_state)
        
    return trajectory
