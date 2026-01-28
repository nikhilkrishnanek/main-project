"""
UI Components Module
===================

Reusable Streamlit components for the Photonic Radar Dashboard.

Components:
1. render_sidebar: Handles Sim/Process configuration.
2. target_manager: UI to add/remove targets.
3. render_metrics: Displays top-bar KPIs.

Author: Principal Photonic Radar Scientist
"""

import numpy as np
import streamlit as st
from photonic.signals import PhotonicConfig
from photonic.environment import ChannelConfig, Target
from photonic.noise import NoiseConfig
from photonic.scenarios import ScenarioGenerator
from ai_models.model import get_tactical_classes
import plotly.graph_objects as go
import plotly.express as px

def render_metrics(metrics: dict):
    """Renders the top KPI metrics row with tactical styling."""
    c1, c2, c3, c4 = st.columns(4)
    
    val_snr = metrics.get('snr_db', -99)
    if not np.isfinite(val_snr): val_snr = -99.9
    
    c1.metric("SIGNAL QUALITY (SNR)", f"{val_snr:.1f} dB")
    c2.metric("RANGE RES", f"{metrics.get('range_res', 0):.2f} m")
    c3.metric("VELOCITY RES", f"{metrics.get('vel_res', 0):.2f} m/s")
    c4.metric("AI CONFIDENCE", f"{metrics.get('ai_conf', 0):.1%}")

def render_ppi(targets: List[Dict], history_size: int = 5, scan_angle: float = 0.0):
    """
    Renders a Plan Position Indicator (PPI) radar display with track history and scan beam.
    """
    st.markdown("### ⏺️ PPI RADAR DISPLAY")
    
    if 'ppi_history' not in st.session_state:
        st.session_state.ppi_history = {} # {id: [(r, theta)]}

    fig = go.Figure()
    
    # Draw Scan Beam (Line + Sector)
    fig.add_trace(go.Scatterpolar(
        r=[0, 3000],
        theta=[scan_angle, scan_angle],
        mode='lines',
        line=dict(color='rgba(77, 250, 77, 0.8)', width=2),
        name='Scan Beam',
        hoverinfo='skip',
        showlegend=False
    ))

    for t in targets:
        tid = t['id']
        r_now = t['estimated_range_m']
        # Use provided azimuth or default
        theta_now = t.get('azimuth_deg', (tid * 45) % 360) 
        
        if tid not in st.session_state.ppi_history:
            st.session_state.ppi_history[tid] = []
        st.session_state.ppi_history[tid].append((r_now, theta_now))
        
        # Keep only recent history
        if len(st.session_state.ppi_history[tid]) > history_size:
            st.session_state.ppi_history[tid].pop(0)
            
        # Draw history trail
        hist = st.session_state.ppi_history[tid]
        if len(hist) > 1:
            h_r, h_theta = zip(*hist)
            fig.add_trace(go.Scatterpolar(
                r=h_r, theta=h_theta,
                mode='lines',
                line=dict(color='#2b5c2b', width=1),
                hoverinfo='skip',
                showlegend=False
            ))

        # Current position
        fig.add_trace(go.Scatterpolar(
            r=[r_now], theta=[theta_now],
            mode='markers+text',
            text=[f"TGT {tid}"],
            marker=dict(size=14, color='#4dfa4d', symbol='cross', line=dict(color='white', width=1)),
            textposition="top center",
            name=f"Target {tid}"
        ))
    
    fig.update_layout(
        template="plotly_dark",
        polar=dict(
            bgcolor="#050805",
            radialaxis=dict(visible=True, range=[0, 3000], color="#1a331a", gridcolor="#1a331a"),
            angularaxis=dict(color="#1a331a", gridcolor="#1a331a", rotation=90, direction="clockwise")
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

def render_doppler_waterfall(rd_map: np.ndarray, max_history: int = 50):
    """
    Renders a scrolling Doppler Waterfall (Time vs Doppler Velocity).
    Shows the Peak Intensity slice across Doppler bins over time.
    """
    st.markdown("### 🌊 DOPPLER WATERFALL HISTORY")
    
    if 'waterfall_buffer' not in st.session_state:
        st.session_state.waterfall_buffer = np.zeros((max_history, rd_map.shape[0]))

    # Extract Doppler slice (Max intensity across ranges for each doppler bin)
    doppler_slice = np.max(rd_map, axis=1)
    
    # Update buffer (Scroll up)
    st.session_state.waterfall_buffer = np.roll(st.session_state.waterfall_buffer, -1, axis=0)
    st.session_state.waterfall_buffer[-1, :] = doppler_slice
    
    fig = px.imshow(
        st.session_state.waterfall_buffer,
        color_continuous_scale="Viridis",
        labels=dict(x="Doppler Bin (Velocity)", y="Time (Frames)"),
        aspect="auto"
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

def render_target_table(targets: List[Dict]):
    """Renders a tactical targets table with threat sorting."""
    st.markdown("### 📋 ACTIVE TARGET INVENTORY")
    
    if not targets:
        st.info("Searching for tactical signatures...")
        return

    import pandas as pd
    
    formatted_data = []
    for t in targets:
        # Threat assessment logic
        threat_score = 0
        if t['estimated_range_m'] < 500: threat_score += 50 # Proximity
        if abs(t['estimated_velocity_ms']) > 100: threat_score += 30 # High speed
        if t.get('class') == 'Missile': threat_score += 20 # Weaponry
        
        status = "CRITICAL" if threat_score >= 80 else "ELEVATED" if threat_score >= 40 else "NOMINAL"
        
        formatted_data.append({
            "ID": f"INV-{t['id']:03d}",
            "RANGE (m)": f"{t['estimated_range_m']:.1f}",
            "VELOCITY (m/s)": f"{t['estimated_velocity_ms']:.1f}",
            "CLASSIFICATION": t.get('class', 'Unknown').upper(),
            "STATUS": status,
            "THREAT": threat_score
        })
        
    df = pd.DataFrame(formatted_data).sort_values("THREAT", ascending=False)
    
    def color_status(val):
        color = '#ff3333' if val == "CRITICAL" else '#ffcc00' if val == "ELEVATED" else '#4dfa4d'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        df.style.map(color_status, subset=['STATUS']),
        use_container_width=True,
        hide_index=True
    )

def render_threat_panel(detections: List[Dict]):
    """
    Displays a tactical alert panel for high-risk targets.
    """
    st.markdown("### ⚠️ THREAT ALERT PANEL")
    for det in detections:
        risk = "HIGH" if det.get('estimated_range_m', 1000) < 300 else "MEDIUM"
        color = "#ff3333" if risk == "HIGH" else "#ffcc00"
        
        st.markdown(f"""
            <div style="border-left: 5px solid {color}; padding: 10px; background-color: #1a0a0a; margin-bottom: 5px;">
                <span style="color: {color}; font-weight: bold;">[{risk} RISK]</span> 
                Target ID {det['id']} at {det['estimated_range_m']:.1f}m | Velocity: {det['estimated_velocity_ms']:.1f}m/s
            </div>
        """, unsafe_allow_html=True)

def render_sidebar() -> tuple:
    """Renders the configuration sidebar with structured sections and tooltips."""
    st.sidebar.title("📡 System Control")
    st.sidebar.info("Configure the Photonic-Radar parameters below.")
    
    # --- 0. SCENARIO PRESETS ---
    with st.sidebar.expander("🚀 Operational Presets", expanded=False):
        scenarios = ["Custom"] + list(ScenarioGenerator.list_scenarios().keys())
        selected_scenario = st.sidebar.selectbox(
            "Quick Load Scenario", 
            scenarios,
            help="Select a pre-defined tactical scenario to benchmark the system."
        )
    
    # Default values
    p_defaults = {'fc': 10.0, 'bw': 4.0, 'dur': 10.0, 'lw': 100}
    c_defaults = {'noise': -50.0}
    n_defaults = {'rin': -155.0}
    
    if selected_scenario != "Custom":
        s = ScenarioGenerator.load(selected_scenario)
        if st.session_state.get("last_loaded_scenario") != selected_scenario:
            st.session_state.targets = list(s.targets) 
            st.session_state.last_loaded_scenario = selected_scenario
            st.toast(f"System Reconfigured: {selected_scenario}")
            
        c_defaults['noise'] = s.channel_config.noise_level_db
        if s.noise_config.rin_db_hz:
            n_defaults['rin'] = s.noise_config.rin_db_hz
            
    # --- 1. Physics ---
    st.sidebar.markdown("### 🔬 Photonic Core")
    with st.sidebar.container():
        fc = st.sidebar.number_input(
            "Carrier Frequency (GHz)", 8.0, 40.0, p_defaults['fc'], step=1.0,
            help="Center frequency of the radar signal. High frequency = better resolution but higher loss."
        ) * 1e9
        bw = st.sidebar.number_input(
            "Bandwidth (GHz)", 0.5, 12.0, p_defaults['bw'], step=0.5,
            help="Determines the Range Resolution (c/2B). Wider bandwidth = sharper distance sensing."
        ) * 1e9
        dur = st.sidebar.number_input(
            "Sweep Duration (µs)", 1.0, 1000.0, p_defaults['dur'], step=10.0,
            help="Time for one FMCW chirp sweep. Longer time = better velocity resolution."
        ) * 1e-6
        lw = st.sidebar.slider(
            "Laser Linewidth (kHz)", 1, 1000, p_defaults['lw'],
            help="Spectral purity of the optical source. Impacts system phase noise floor."
        ) * 1e3
    
    p_cfg = PhotonicConfig(
        sampling_rate_hz=25e9, 
        start_frequency_hz=fc, 
        sweep_bandwidth_hz=bw, 
        chirp_duration_s=dur, 
        laser_linewidth_hz=lw
    )
    
    # --- 2. Channel ---
    with st.sidebar.expander("🌍 Environmental Factors", expanded=True):
        noise = st.sidebar.slider(
            "Thermal Noise (dB)", -100.0, -10.0, float(c_defaults['noise']), step=1.0,
            help="Background electronic noise floor. Affects detection sensitivity."
        )
        rin = st.sidebar.slider(
            "Laser RIN (dB/Hz)", -170.0, -120.0, float(n_defaults['rin']), step=1.0,
            help="Relative Intensity Noise. Fundamental noise limit of the optical carrier."
        )
        c_cfg = ChannelConfig(carrier_freq=fc, noise_level_db=noise)
        n_cfg = NoiseConfig(rin_db_hz=rin, bandwidth=bw)
    
    # --- 3. Targets ---
    st.sidebar.markdown("### 🎯 Target Deployment")
    
    if 'targets' not in st.session_state:
        st.session_state.targets = [
            Target(100.0, 10.0, 10.0, "Drone"),
            Target(500.0, -50.0, 20.0, "Aircraft")
        ]
        
    # List targets with a cleaner display
    for i, t in enumerate(st.session_state.targets):
        with st.sidebar.container():
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{t.description}**  \n`{t.range_m}m | {t.velocity_m_s}m/s`", help=f"RCS: {t.rcs_dbsm} dBsm")
            if c2.button("🗑️", key=f"del_{i}", help="Remove target"):
                del st.session_state.targets[i]
                st.rerun()

    # Simple form to add target
    with st.sidebar.expander("➕ Add Tactical Target", expanded=False):
        with st.form("add_target", clear_on_submit=True):
            col1, col2 = st.columns(2)
            t_rng = col1.number_input("Range (m)", 10, 8000, 150)
            t_vel = col2.number_input("Vel (m/s)", -800, 800, 30)
            t_rcs = col1.number_input("RCS (dB)", -40, 60, 0)
            t_type = col2.selectbox("Classification", get_tactical_classes())
            if st.form_submit_button("DEPLOY"):
                st.session_state.targets.append(Target(float(t_rng), float(t_vel), float(t_rcs), t_type))
                st.rerun()
        
    return p_cfg, c_cfg, n_cfg, st.session_state.targets
