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
from ui.components import render_sidebar, render_metrics, render_ppi, render_doppler_waterfall, render_threat_panel
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
        # Convert UI targets to Simulation States
        initial_states = [
            TargetState(position_m=float(t.range_m), velocity_m_s=float(t.velocity_m_s))
            for t in ui_targets
        ]
        
        sim = SimulationOrchestrator(vars(p_cfg), initial_states)
        
        # Continuous Execution
        for frame_data in sim.run_loop(max_frames=100):
            # Update PPI
            with ppi_placeholder.container():
                ppi_targets = []
                for i, t in enumerate(frame_data["targets"]):
                    ppi_targets.append({
                        "id": i+1,
                        "range_m": t["position_m"],
                        "velocity_m_s": t["velocity_m_s"],
                        "class": "Drone" # Placeholder until full pipeline integration
                    })
                render_ppi(ppi_targets)
                
            # Update Waterfall
            with waterfall_placeholder.container():
                rd_mock = np.random.randn(64, 128) # Mock data for UI demo
                render_doppler_waterfall(rd_mock)
                
            # Update Target Table
            with target_table_placeholder.container():
                import pandas as pd
                df = pd.DataFrame(frame_data["targets"])
                st.dataframe(df, use_container_width=True)
                
            # Update Threat Panel
            with threat_panel_placeholder.container():
                render_threat_panel(ppi_targets)
                
            time.sleep(0.05)
