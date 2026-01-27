import streamlit as st
import plotly.graph_objects as go
import numpy as np

from ui.components import render_sidebar, render_metrics
from core.engine import RadarPipeline, RadarFrame

# --- Page Configuration ---
st.set_page_config(
    page_title="PHOENIX-RADAR",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Cyberpunk Styling ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    
    /* Custom Header Font */
    h1, h2, h3 { font-family: 'Orbitron', 'Inter', 'sans-serif'; color: #00FFCC; font-weight: 700; }
    
    /* Metric Boxes */
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #2D333B;
        padding: 15px;
        border-radius: 12px;
        transition: transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover { border-color: #00FFCC; transform: translateY(-2px); }
    div[data-testid="stMetricValue"] { font-size: 26px !important; color: #00FFCC !important; font-family: 'Courier New', monospace; }
    div[data-testid="stMetricLabel"] { font-size: 11px !important; color: #ABB2BF !important; text-transform: uppercase; letter-spacing: 1.2px; }

    /* Prediction Card */
    .ai-pred-box {
        padding: 20px;
        border: 2px solid #30363D;
        border-radius: 15px;
        text-align: center;
        background: linear-gradient(145deg, #1A1F26, #0D1117);
        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
        margin-bottom: 20px;
    }
    .ai-label-safe { font-size: 38px; font-weight: 800; color: #00FF99; text-shadow: 0 0 15px rgba(0,255,153,0.4); }
    .ai-label-danger { font-size: 38px; font-weight: 800; color: #FF3366; text-shadow: 0 0 15px rgba(255,51,102,0.4); }
    .ai-conf { font-size: 14px; color: #8B949E; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }

    /* Section Spacing */
    .section-header { 
        margin-top: 25px; 
        margin-bottom: 12px; 
        border-left: 4px solid #00FFCC; 
        padding-left: 10px; 
        font-size: 18px; 
        font-weight: 600;
        color: #E6EDF3;
    }
    
    /* Info/Warning Overrides */
    .stAlert { border-radius: 10px; border: none; background-color: #1A1F26; }
</style>
""", unsafe_allow_html=True)

def render_plots(frame: RadarFrame):
    """Renders the main visualization tabs with polished charts."""
    tab1, tab2, tab3 = st.tabs(["🎯 TACTICAL VIEW", "🔬 SIGNAL ANALYSIS", "📉 RESEARCH BENCHMARKS"])
    
    with tab1:
        c1, c2 = st.columns([1, 2])
        
        # --- AI Panel ---
        with c1:
            st.markdown('<div class="section-header">🧠 AI Intelligence Analysis</div>', unsafe_allow_html=True)
            pred = frame.prediction
            
            # XAI Integration
            from ai.xai import generate_explanation
            explanation = generate_explanation(pred.predicted_class, pred.confidence, frame.metrics)
            
            # Dynamic Style
            css_class = "ai-label-danger" if pred.predicted_class in ["Missile", "Drone"] else "ai-label-safe"
            
            st.markdown(f"""
            <div class="ai-pred-box">
                <div class="ai-conf">IDENTIFIED THREAT</div>
                <div class="{css_class}">{explanation.title}</div>
                <div class="ai-conf">Confidence Score: {pred.confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📝 Strategic Analysis & XAI", expanded=True):
                st.write(explanation.narrative)
                if explanation.warning:
                    st.warning(explanation.warning)
                
                st.markdown("**Evidence Attribution**")
                for feat, weight in explanation.feature_importance.items():
                    if weight > 0:
                        st.progress(weight, text=f"{feat}")

        # --- Range-Doppler Map ---
        with c2:
            st.markdown('<div class="section-header">🗺️ Range-Doppler Heatmap</div>', unsafe_allow_html=True)
            fig = go.Figure(data=go.Heatmap(
                z=frame.rd_map,
                colorscale='Magma',
                showscale=True,
                colorbar=dict(title="Amplitude", thickness=15, len=0.8)
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=30, b=40, l=40, r=20),
                xaxis=dict(title="Doppler Velocity (m/s)", showgrid=False, zeroline=False),
                yaxis=dict(title="Range Distance (m)", showgrid=False, zeroline=False),
                font=dict(color="#ABB2BF", size=10),
                height=420,
                hovermode='closest'
            )
            st.plotly_chart(fig, width="stretch")
            st.caption("Visualizes target distance vs. velocity using 2D FFT.")

        # --- Spectrogram ---
        st.markdown('<div class="section-header">🌊 Micro-Doppler Time-Frequency Signature</div>', unsafe_allow_html=True)
        fig_spec = go.Figure(data=go.Heatmap(
            z=frame.spectrogram,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Intensity", thickness=15, len=0.8)
        ))
        fig_spec.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, b=40, l=40, r=20),
            xaxis=dict(title="Observation Time (Sweep Index)", showgrid=False),
            yaxis=dict(title="Doppler Frequency Shift (Hz)", showgrid=False),
            font=dict(color="#ABB2BF", size=10),
            height=320
        )
        st.plotly_chart(fig_spec, width="stretch")
        st.caption("Reveals mechanical vibrations (propellers, turbines) over time.")

    with tab2:
        st.markdown('<div class="section-header">📉 Raw Photodetector Output (Time Domain)</div>', unsafe_allow_html=True)
        fig_raw = go.Figure()
        ds = 10 
        fig_raw.add_trace(go.Scatter(
            x=frame.time_axis[::ds] * 1e6, 
            y=np.real(frame.rx_signal)[::ds],
            mode='lines',
            line=dict(color='#00FFCC', width=1.5),
            name="Real Component"
        ))
        fig_raw.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,10,10,0.5)',
            margin=dict(t=30, b=40, l=50, r=20),
            xaxis=dict(title="Time (µs)", showgrid=True, gridcolor='#2D333B'),
            yaxis=dict(title="Amplitude (V)", showgrid=True, gridcolor='#2D333B'),
            font=dict(color="#ABB2BF", size=11),
            height=350,
            hovermode='x'
        )
        st.plotly_chart(fig_raw, width="stretch")
        st.caption("Signal directly from the high-speed photodiode after optical heterodyning.")
    
    with tab3:
        st.markdown('<div class="section-header">📊 Quantitative Research Benchmarks</div>', unsafe_allow_html=True)
        st.info("Performance characterization of the Photonic-Radar system across varying SNR and operational constraints.")
        
        from evaluation.benchmarking import get_pd_curve, get_pfa_curve, get_ai_accuracy_benchmark, get_latency_benchmark
        
        c1, c2 = st.columns(2)
        
        with c1:
            # 1. Pd vs SNR
            snr_sweep = np.linspace(-15, 25, 40)
            pd_vals = get_pd_curve(snr_sweep)
            fig_pd = go.Figure(go.Scatter(x=snr_sweep, y=pd_vals, mode='lines', line=dict(color='#00FFCC', width=3)))
            fig_pd.update_layout(title="Sensitivity: Pd vs SNR", xaxis_title="SNR (dB)", yaxis_title="Prob. Detection", height=300, 
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.2)', font=dict(color="#888"))
            st.plotly_chart(fig_pd, width="stretch")
            st.caption("Validates the radar's ability to detect weak signals. Ideal for Swerling-I fluctuating targets.")

            # 2. AI Accuracy vs Noise
            if st.button("RUN AI STRESS TEST 🧠"):
                with st.spinner("Analyzing AI robustness..."):
                    acc_vals = get_ai_accuracy_benchmark(st.session_state.pipeline, np.linspace(-10, 20, 10))
                    fig_acc = go.Figure(go.Scatter(x=np.linspace(-10, 20, 10), y=acc_vals, mode='lines+markers', line=dict(color='#FF3366')))
                    fig_acc.update_layout(title="Robustness: AI Accuracy vs SNR", xaxis_title="SNR (dB)", yaxis_title="Accuracy", height=300,
                                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.2)', font=dict(color="#888"))
                    st.plotly_chart(fig_acc, width="stretch")
            else:
                st.write("Click 'Run AI Stress Test' to evaluate neural network performance under noise.")

        with c2:
            # 3. Pfa vs Threshold
            thresh_sweep = np.linspace(5, 25, 40)
            pfa_vals = get_pfa_curve(thresh_sweep)
            fig_pfa = go.Figure(go.Scatter(x=thresh_sweep, y=pfa_vals, mode='lines', line=dict(color='#FFA500')))
            fig_pfa.update_layout(title="Reliability: Pfa vs Threshold", xaxis_title="Threshold (dB above noise)", yaxis_title="False Alarm Rate", 
                                yaxis_type="log", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.2)', font=dict(color="#888"))
            st.plotly_chart(fig_pfa, width="stretch")
            st.caption("Measures system susceptibility to false triggers. Essential for setting operational thresholds.")

            # 4. Latency scaling
            comp_factors = [1, 2, 4, 8, 16]
            lat_vals = get_latency_benchmark(st.session_state.pipeline, comp_factors)
            fig_lat = go.Figure(go.Scatter(x=comp_factors, y=lat_vals, mode='lines+markers', line=dict(color='#00CCFF')))
            fig_lat.update_layout(title="Scalability: Latency vs Data Volume", xaxis_title="Signal Complexity Factor", yaxis_title="Runtime (ms)", 
                                height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.2)', font=dict(color="#888"))
            st.plotly_chart(fig_lat, width="stretch")
            st.caption("Benchmarks the computational efficiency of the DSP + AI pipeline.")
        s = frame.stats 
        perf = frame.performance
        
        st.markdown("#### Detection Statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Detection Prob. (Pd)", f"{s.probability_detection:.4f}", help="Calculated using Shnidman/Albersheim models for the given SNR.")
        c2.metric("False Alarm Rate", f"{s.false_alarm_rate:.1e}", help="Probability of background noise being misidentified as a target.")
        c3.metric("AI Entropy", f"{s.model_entropy:.3f}", help="Measures classification uncertainty. Lower is more certain.")
        
        st.markdown("---")
        st.markdown("#### Physical Resolution Benchmarks")
        cols = st.columns(2)
        cols[0].info(f"**Range Resolution**: {perf.range_resolution:.3f} m")
        cols[1].info(f"**Velocity Resolution**: {perf.velocity_resolution:.2f} m/s")

def main():
    # 1. Pipeline Initialization (Singleton in Session State)
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = RadarPipeline()

    # 2. Sidebar Configuration
    p_cfg, c_cfg, n_cfg, targets = render_sidebar()

    # 3. Header & Controls
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("PHOENIX-RADAR")
        st.markdown("*Cognitive Microwave Photonics Research System | Advanced AI Analysis*")
    with c2:
        st.write("---")
        if st.button("RUN SIMULATION ⚡", type="primary", help="Trigger the photonic signal generation and AI processing pipeline."):
            with st.spinner("Processing Physics & AI Analysis..."):
                st.session_state.last_frame = st.session_state.pipeline.run(
                    p_cfg, c_cfg, n_cfg, targets
                )

    st.divider()

    # 4. Main View
    if 'last_frame' in st.session_state:
        frame = st.session_state.last_frame
        
        # Metrics Header
        metrics = frame.metrics
        metrics.update({
            'range_res': frame.performance.range_resolution,
            'vel_res': frame.performance.velocity_resolution,
            'ai_conf': frame.prediction.confidence
        })
        render_metrics(metrics)
        st.divider()
        
        # Visualization
        render_plots(frame)
    else:
        st.info("👋 Welcome to PHOENIX-RADAR. Configure parameters in the sidebar and click 'RUN SIMULATION'.")

if __name__ == "__main__":
    main()
