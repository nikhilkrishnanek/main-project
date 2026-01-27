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
from ai_models.model import list_classes
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

def render_ppi(targets: List[Dict]):
    """
    Renders a Plan Position Indicator (PPI) radar display.
    """
    st.markdown("### ⏺️ PPI RADAR DISPLAY")
    
    # Simulating bearing for visualization if not present
    # In a real system, bearing comes from the antenna position/beamforming
    theta = [np.random.uniform(0, 360) for _ in targets]
    r = [t['range_m'] for t in targets]
    labels = [f"ID {t['id']} [{t.get('class', 'Unknown')}]" for t in targets]
    
    fig = go.Figure()
    
    # Add target blips
    fig.add_trace(go.Scatterpolar(
        r=r, theta=theta,
        mode='markers+text',
        text=labels,
        marker=dict(size=12, color='#4dfa4d', symbol='cross'),
        textposition="top center",
        name="Tactical Targets"
    ))
    
    fig.update_layout(
        template="plotly_dark",
        polar=dict(
            bgcolor="#050805",
            radialaxis=dict(visible=True, range=[0, 1000], color="#1a331a"),
            angularaxis=dict(color="#1a331a")
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

def render_doppler_waterfall(rd_map: np.ndarray):
    """
    Renders a Doppler Waterfall (Time-Frequency Evolution).
    For simulation, we use the current RD-Map as a frame.
    """
    st.markdown("### 🌊 DOPPLER WATERFALL")
    fig = px.imshow(
        rd_map,
        color_continuous_scale="Viridis",
        labels=dict(x="Range Bin", y="Doppler Bin"),
        aspect="auto"
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

def render_threat_panel(detections: List[Dict]):
    """
    Displays a tactical alert panel for high-risk targets.
    """
    st.markdown("### ⚠️ THREAT ALERT PANEL")
    for det in detections:
        risk = "HIGH" if det.get('range_m', 1000) < 300 else "MEDIUM"
        color = "#ff3333" if risk == "HIGH" else "#ffcc00"
        
        st.markdown(f"""
            <div style="border-left: 5px solid {color}; padding: 10px; background-color: #1a0a0a; margin-bottom: 5px;">
                <span style="color: {color}; font-weight: bold;">[{risk} RISK]</span> 
                Target ID {det['id']} at {det['range_m']:.1f}m | Velocity: {det['velocity_m_s']:.1f}m/s
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
    n_defaults = {'rin': -155}
    
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
        fs=25e9, 
        f_start=fc, 
        bandwidth=bw, 
        duration=dur, 
        laser_linewidth=lw
    )
    
    # --- 2. Channel ---
    with st.sidebar.expander("🌍 Environmental Factors", expanded=True):
        noise = st.sidebar.slider(
            "Thermal Noise (dB)", -100.0, -10.0, c_defaults['noise'],
            help="Background electronic noise floor. Affects detection sensitivity."
        )
        rin = st.sidebar.slider(
            "Laser RIN (dB/Hz)", -170, -120, n_defaults['rin'],
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
            t_type = col2.selectbox("Classification", list_classes())
            if st.form_submit_button("DEPLOY"):
                st.session_state.targets.append(Target(float(t_rng), float(t_vel), float(t_rcs), t_type))
                st.rerun()
        
    return p_cfg, c_cfg, n_cfg, st.session_state.targets
