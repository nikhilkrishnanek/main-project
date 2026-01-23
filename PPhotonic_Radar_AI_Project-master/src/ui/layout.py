import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

from src.data.generator import RadarGenerator, Target
from src.signal.features import get_all_features
from src.ai.inference import InferenceEngine
from src.ui.components import draw_header, render_sidebar

# Initialize Singletons in Session State
if 'radar_gen' not in st.session_state:
    st.session_state.radar_gen = RadarGenerator()
if 'ai_engine' not in st.session_state:
    st.session_state.ai_engine = InferenceEngine("results/radar_model_pytorch.pt")

def render_main_layout():
    draw_header()
    
    # 1. Sidebar Controls
    mode, config = render_sidebar()
    
    # 2. Main Content
    # We create a placeholder for the live dashboard
    dashboard_container = st.container()
    
    with dashboard_container:
        st.markdown(f"## 👁️ LIVE MONITORING [{mode.upper()}]")
        
        # Grid Layout
        c1, c2 = st.columns([1, 1])
        
        rd_plot = c1.empty()
        spec_plot = c2.empty()
        
        c3, c4 = st.columns([2, 1])
        ai_table = c3.empty()
        log_feed = c4.empty()
        
    # 3. Execution Logic
    start_btn = st.sidebar.button("RUN SIMULATION", type="primary")
    stop_btn = st.sidebar.button("STOP")
    
    if 'running' not in st.session_state:
        st.session_state.running = False
        
    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False
        
    if st.session_state.running:
        run_loop(rd_plot, spec_plot, ai_table, log_feed, config)

def run_loop(rd_plot, spec_plot, ai_table, log_feed, config):
    gen = st.session_state.radar_gen
    ai = st.session_state.ai_engine
    
    # Convert config targets to Target objects
    targets = []
    for t in config.get('targets', []):
        targets.append(Target(
            range_m=t['range'],
            velocity_m_s=t['velocity'],
            rcs_db=t['rcs'],
            category=t['category']
        ))
        
    # Simulate Frame
    # In a real app, this would be a loop with time.sleep
    # Streamlit re-runs scripts, so we simulate *one frame* per re-run or use a loop inside
    # "st.empty" allows us to update in-place without full rerun inside a loop
    
    # Let's run a short loop of frames
    fps = 0
    t0 = time.time()
    
    # Simulate 10 frames then yield to Streamlit rerender check (or endless loop with breaks)
    # Ideally, inside Streamlit, we loop forever until logic breaks it.
    
    step = 0
    while st.session_state.running:
        # Data Generation
        sim_data = gen.simulate_scenario(targets, duration=0.1) # 100ms chunks
        signal = sim_data['if_signal'] # This is the mixed signal
        
        # Signal Processing
        rd_map, spec, metadata, phot_params = get_all_features(signal, fs=gen.fs)
        
        # AI Inference
        probs = ai.predict(rd_map, spec, metadata)
        classes = ai.get_classes()
        top_idx = np.argmax(probs)
        pred_class = classes[top_idx]
        conf = probs[top_idx]
        
        # Visualization Update
        with rd_plot.container():
            fig_rd = go.Figure(data=go.Heatmap(z=rd_map, colorscale='Viridis'))
            fig_rd.update_layout(title="Range-Doppler Map", margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_rd, width="stretch", key=f"rd_{step}")
            
        with spec_plot.container():
            fig_sp = go.Figure(data=go.Heatmap(z=spec, colorscale='Magma'))
            fig_sp.update_layout(title="Micro-Doppler Spectrogram", margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_sp, width="stretch", key=f"sp_{step}")
            
        with ai_table.container():
            st.markdown("### AI Classification Results")
            # Create a simple dataframe-like display
            # We highlight the predicted class
            
            # Custom HTML for AI output
            st.markdown(f"""
            <div style="background-color: #1e2130; padding: 15px; border-radius: 5px; border-left: 5px solid #00ff00;">
                <h2 style="margin:0; color: #fff;">DETECTED: <span style="color: #00ff00;">{pred_class.upper()}</span></h2>
                <p style="margin:0; color: #aaa;">Confidence: {conf*100:.1f}%</p>
                <p style="margin:0; color: #aaa;">Range Estimate: {phot_params['pulse_width']*1e6:.1f} ?? (Need estimator)</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.bar_chart(probs) # Simple bar chart often looks cleaner than complex plotly for probs
            
        with log_feed.container():
            st.write("System Logs")
            st.code(f"Frame {step}: Processing {len(signal)} samples.\nNoise Floor: {phot_params['noise_power']:.2f}\nClutter: {phot_params['clutter_power']:.2f}", language="text")
            
        step += 1
        time.sleep(0.1) # limit FPS
        # Check stop
        # Streamlit doesn't handle "break" from button easily inside loop. 
        # Actually this loop blocks the UI. 
        # Better pattern: Run ONE step, then using st.rerun()?
        # Or distinct "Stop" button outside?
        # Streamlit execution model is tricky with infinite loops.
        # We will iterate a few times then break to allow UI inputs to process?
        # No, `st.empty()` allows updates inside loop. But `Stop` button won't register until script finishes.
        # We can avoid infinite loop here and use `st.rerun` at end of frame?
        # Let's try Loop with a small "check info" placeholder? No.
        
        # For this execution, we'll run 20 frames then pause to let user click things.
        if step > 50:
             break
             
    if st.session_state.running:
        st.rerun() # Trigger next batch
