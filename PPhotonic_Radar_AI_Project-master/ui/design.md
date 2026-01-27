# User Interface Design: Photonic Radar Command Center

## 1. Visual Hierarchy
The interface follows a "Command & Control" aesthetic (Dark Mode, High Contrast).

### Header
- **Left**: System Status (Online/Offline)
- **Center**: Title "PHOENIX-RADAR: Cognitive Photonic System"
- **Right**: System Metrics (SNR: X dB, FPS: Y)

### Sidebar (Controls)
Segmented into expanders:
1.  **Radar Physics**: Bandwidth, Chirp Duration, Laser Linewidth.
2.  **Simulation Targets**: Add/Edit/Remove targets (Range, Speed, Type).
3.  **Channel**: Noise Floor, Weather condition (Placeholder).
4.  **AI Config**: Model Threshold, Debug Mode.
5.  **Actions**: "RUN SIMULATION", "STOP".

### Main Content (The "Tactical View")
Two primary layouts via Tabs:

**Tab 1: Live Monitoring**
-   **Row 1**: AI Classification Panel (Large Text + Probability Bars).
-   **Row 2**:
    *   **Col 1**: Range-Doppler Map (Interactive Heatmap).
    *   **Col 2**: Micro-Doppler Spectrogram (Interactive Heatmap).
-   **Row 3**: Raw Time-Domain Signal (Expander, mostly for debugging).

**Tab 2: System Health**
-   Laser Phase Noise Plots.
-   Fiber Dispersion Analysis.
-   Estimated Resolution vs Theoretical Resolution comparison.

## 2. User Flow
1.  **Launch**: App starts in "Idle" mode with default config.
2.  **Config**: User tweaks "Target Velocity" to 50 m/s in sidebar.
3.  **Simulate**: User clicks "RUN".
4.  **Pipeline**:
    *   `src.simulation` generates new waveforms.
    *   `src.dsp` processes maps.
    *   `src.ai` predicts "Missile".
5.  **Update**: UI redraws Heatmaps and flashes "DETECTED: MISSILE" in Red.

## 3. Technology
-   **Framework**: Streamlit.
-   **Plotting**: Plotly Graph Objects (`go.Heatmap`) for performance (WebGL).
-   **State**: `st.session_state` handles the "Running" flag and "Last Frame" data.
