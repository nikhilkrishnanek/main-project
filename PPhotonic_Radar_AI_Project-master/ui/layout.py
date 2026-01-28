"""
Strategic Radar Dashboard Layout
================================

Orchestrates the multi-column tactical workspace.
Features:
1. Real-time PPI Display.
2. Doppler Waterfall (Frequency-Time).
3. Target ID and Status Table.
4. High-priority Threat Alerts.

Author: Defense UI/UX Designer
"""

import streamlit as st
import numpy as np
import time
from typing import List, Dict

from ui.theme import apply_tactical_theme
from ui.components import render_sidebar, render_metrics, render_ppi, render_doppler_waterfall, render_threat_panel, render_target_table
from simulation_engine.orchestrator import SimulationOrchestrator, TargetState

def render_main_layout():
    # 1. Apply Tactical Aesthetics
    apply_tactical_theme(st)
    
    # 2. Sidebar Configuration
    p_cfg, c_cfg, n_cfg, ui_targets = render_sidebar()
    
    # Header with blinking effect simulation using CSS
    st.markdown("""
        <div style='text-align: center; padding: 10px; border-bottom: 2px solid #1a331a;'>
            <h1 style='color: #4dfa4d; margin: 0;'>📡 PHOENIX-RADAR STRATEGIC COMMAND</h1>
            <p style='color: #2b5c2b; font-size: 0.8em;'>INTERNAL DEFENSE NETWORK - CLASSIFIED LEVEL 4</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 3. Top-Level Metrics
    metrics = {
        "snr_db": 24.5,
        "range_res": 0.15,
        "vel_res": 0.5,
        "ai_conf": 0.98
    }
    render_metrics(metrics)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4. Tactical Grid
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        ppi_placeholder = st.empty()
        
    with col_right:
        waterfall_placeholder = st.empty()
        
    st.markdown("---")
    
    col_btm_left, col_btm_right = st.columns([2, 1])
    
    with col_btm_left:
        st.markdown("### 📋 TARGET IDENTIFICATION TABLE")
        target_table_placeholder = st.empty()
        
    with col_btm_right:
        threat_panel_placeholder = st.empty()

    # 5. Simulation Control
    if st.sidebar.button("INITIATE TACTICAL SWEEP", type="primary"):
        # Convert UI targets to Simulation States (2D Projection)
        initial_states = []
        for i, t in enumerate(ui_targets):
             # Assign random azimuth for 2D simulation to scatter targets on PPI
             angle_rad = np.random.uniform(0, 2 * np.pi)
             px = t.range_m * np.cos(angle_rad)
             py = t.range_m * np.sin(angle_rad)
             # Project radial velocity to vector
             vx = t.velocity_m_s * np.cos(angle_rad)
             vy = t.velocity_m_s * np.sin(angle_rad)
             
             initial_states.append(TargetState(
                 id=i+1, 
                 pos_x=px, pos_y=py, 
                 vel_x=vx, vel_y=vy, 
                 type=t.description.lower()
             ))
        
        # Add Scanning Config
        config_dict = p_cfg.as_dict() if hasattr(p_cfg, 'as_dict') else vars(p_cfg)
        config_dict['rpm'] = 12.0
        config_dict['beamwidth_deg'] = 15.0 # Wide beam for visibility
        
        sim = SimulationOrchestrator(config_dict, initial_states)
        
        # Reset buffers on new sweep
        if 'ppi_history' in st.session_state: del st.session_state.ppi_history
        if 'waterfall_buffer' in st.session_state: del st.session_state.waterfall_buffer

        # Continuous Execution
        for frame_data in sim.run_loop(max_frames=200):
            # Update Metrics (Dynamic)
            perf_metrics = frame_data["metrics"]["averages"]
            metrics["snr_db"] = 25.0 + np.random.uniform(-2, 2)
            
            # Update PPI
            # Update PPI
            with ppi_placeholder.container():
                ppi_targets = []
                for t in frame_data["targets"]:
                    # Compute Polar from Cartesian
                    px, py = t["pos_x"], t["pos_y"]
                    rng = np.sqrt(px**2 + py**2)
                    az = np.degrees(np.arctan2(py, px)) % 360
                    
                    ppi_targets.append({
                        "id": t["id"],
                        "estimated_range_m": rng,
                        "azimuth_deg": az,
                        "estimated_velocity_ms": t.get("radial_velocity", 0.0),
                        "class": t.get("type", "Unknown").capitalize()
                    })
                
                # We should also visualize the scan line
                scan_angle = frame_data.get("scan_angle", 0)
                render_ppi(ppi_targets, scan_angle=scan_angle)
                
            # Update Waterfall
            with waterfall_placeholder.container():
                render_doppler_waterfall(frame_data["rd_map"])
                
            # Update Target Table
            with target_table_placeholder.container():
                render_target_table(ppi_targets)
                
            # Update Threat Panel
            with threat_panel_placeholder.container():
                render_threat_panel(ppi_targets)
                
            time.sleep(0.01)
