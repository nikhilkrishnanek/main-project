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
    id: int
    position_m: float
    velocity_m_s: float
    acceleration_m_s2: float = 0.0
    type: str = "civilian" # drone, aircraft, missile, bird
    maneuver_type: str = "linear" # linear, sinusoidal, evasive

class KinematicEngine:
    def __init__(self, dt: float):
        self.dt = dt
        self.time = 0.0

    def update_state(self, state: TargetState) -> TargetState:
        """
        Calculates the next state based on current kinematics and maneuver models.
        """
        self.time += self.dt
        
        accel = state.acceleration_m_s2
        
        # Maneuver Logic
        if state.maneuver_type == "sinusoidal":
            # Oscillatory movement (e.g., drone hovering/bobbing)
            accel = 2.0 * np.sin(2 * np.pi * 0.5 * self.time)
        elif state.maneuver_type == "evasive":
            # High-G turns or sudden acceleration changes
            if int(self.time) % 5 == 0:
                accel = np.random.uniform(-10, 10)
        
        new_pos = state.position_m + (state.velocity_m_s * self.dt) + (0.5 * accel * (self.dt**2))
        new_vel = state.velocity_m_s + (accel * self.dt)
        
        # Boundary check (simple wrap for range simulation)
        if new_pos > 5000: new_pos = 100
        
        return TargetState(
            id=state.id,
            position_m=new_pos,
            velocity_m_s=new_vel,
            acceleration_m_s2=accel,
            type=state.type,
            maneuver_type=state.maneuver_type
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
