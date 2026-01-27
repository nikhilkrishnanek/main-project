# Interview & Viva Prep: PHOENIX-RADAR Technical Points

Use these talking points to defend your project and explain the technical depth to researchers or recruiters.

## 1. Why High Bandwidth? 
> "In radar, range resolution $\Delta R$ is inversely proportional to bandwidth $B$ ($\Delta R = c/2B$). By using Microwave Photonics, we can generate bandwidths exceeding 10 GHz, which allows us to resolve targets just 1.5 cm apart. This is impossible for standard electronic radars."

## 2. Why Hybrid CNN-LSTM?
> "A target's signature has two dimensions: **Spatial** (the shape of the Doppler spectrum at a specific moment) and **Temporal** (how that shape changes over time). The CNN branch extracts spatial micro-Doppler features (like rotor blade vibration), while the LSTM branch learns the long-term kinematic behavior, making the system incredibly robust against decoys."

## 3. How do you handle Data Association?
> "We use **Global Nearest Neighbor (GNN)** data association. We predict where each track will be using the Kalman filter's propagation step and then solve a 2D assignment problem to link the nearest detections to those predicted states. This ensures that even if two drones cross paths, their unique IDs remain persistent."

## 4. What is the role of Taylor Windowing?
> "Traditional Hann/Hamming windows have a trade-off between main-lobe width and sidelobe level. **Taylor windows** allow us to specify the peak sidelobe level (PSL) while minimizing the broadening of the main peak. This is critical for detecting a small drone lurking near the 'shadow' of a large aircraft."

## 5. Explain the XAI (Explainable AI) Layer
> "Black-box AI isn't trusted in defense. Our XAI layer uses **calibrated confidence** (via temperature scaling) and **narrative generation**. It looks at the neural network's activation and links it to physical features—for example, it identifies 'periodic sidebands in the spectrogram' to explain why a target was classified as a drone."

## 6. Real-time Optimization
> "To achieve **35 FPS**, we vectorized the signal synthesis in NumPy and used optimized convolution (fast-time/slow-time FFTs). Our pipeline latency is dominated by DSP (12ms), but we've ensured the AI inference runs in parallel on the GPU to stay within our 100ms real-time target loop."
