# UI Design: Guided Radar Walkthrough

## Goal
To demystify the Photonic Radar pipeline by breaking the instant simulation into an interactive, step-by-step educational story.

## User Flow Strategy
We use a **Wizard Interface** (Previous / Next) instead of the single-page dashboard. The simulation runs incrementally or reveals data incrementally.

### Step 1: Photonic Signal Generation ⚡
*   **Concept**: Laser physics and FMCW modulation.
*   **Controls**: Slider for `Laser Linewidth` and `Chirp Bandwidth`.
*   **Visual**:
    *   Plot 1: Optical frequency vs Time (The Chirp).
    *   Plot 2: Phase Noise impacts (Ideal vs Noisy).
*   **Output**: "Transmitted Signal ($Tx$)" generated.

### Step 2: Target Simulation 🎯
*   **Concept**: Time-of-Flight and Doppler Effect.
*   **Controls**: "Add Target" (Range, Velocity).
*   **Visual**:
    *   Plot: Transmitted Pulse vs Received Echo (Time Domain).
    *   Highlight: The time delay $\Delta t$ and phase shift.
*   **Output**: "Received Signal ($Rx$)" generated.

### Step 3: Spectrum Analysis (DSP) 🌈
*   **Concept**: Recovering info via Mixing and FFT.
*   **Controls**: Window Function (Hamming/Hann).
*   **Visual**:
    *   **2D-FFT**: The Range-Doppler Map appears.
    *   **STFT**: The Spectrogram appears.
*   **Insight**: "Look at the bright spot at (150m, 30m/s). That's your target."

### Step 4: AI Decision (The Brain) 🧠
*   **Concept**: Neural Network Inference.
*   **Action**: "Analyse Signature" button.
*   **Visual**:
    *   **Input**: The RD Map from Step 3.
    *   **Output**: Class Label (e.g., "Drone") + Confidence Bar.
    *   **XAI**: "Why? Micro-Doppler modulation detected."

### Step 5: Performance Evaluation 📊
*   **Concept**: System Validation.
*   **Visual**:
    *   **SNR Gauge**: Green/Red zone.
    *   **Pd/FAR**: Statistical reliability.
*   **Conclusion**: "Mission Success/Failure".

## Technical Implementation
*   **State Management**: Use `st.session_state.step` (1 to 5).
*   **Pipeline Access**:
    *   We likely need to expose *intermediate* functions of `src.pipeline` or allow the pipeline to return "Partial Frames".
    *   *Simpler Approach*: Run full pipeline, but hide tabs/plots until the specific step is active.

## educational Value
*   Users learn that **Bandwidth = Range Resolution**.
*   Users see how **Noise** ruins the detection in Step 3.
*   Users understand **Doppler** shift in Step 2.
