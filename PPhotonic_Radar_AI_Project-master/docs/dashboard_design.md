# Dashboard Design: Strategic Radar Command

This document explains the UI/UX rationale for the **PHOENIX-RADAR** tactical dashboard redesign.

## 1. Visual Hierarchy & Theme
The dashboard uses a **Military-Grade Tactical Theme** characterized by:
- **Monochrome-Green Aesthetic**: Maximizes readability in low-light environments and reduces ocular strain for long-duration surveillance.
- **Dark Neutral Backgrounds (#050805)**: Enhances the contrast of signal blips (PPI) and spectral components (Doppler).
- **Tactical Grid Layout**: Balanced 2-column grid for spatial (PPI) and spectral (Waterfall) awareness.

## 2. Component Rationale

### PPI (Plan Position Indicator)
Legacy radars used analog PPI displays. Our digital implementation provides:
- **Polar Perspective**: Maintains the "North-up" or "Heading-up" perspective familiar to defense operators.
- **ID Persistence**: Tracking IDs are anchored to visual blips, reducing cognitive load during target-heavy scenarios (e.g., Drone Swarms).

### Doppler Waterfall
While the RD-map shows a snapshot, the **Doppler Waterfall** shows the temporal evolution of the target's signature:
- **Spatiotemporal Awareness**: Operators can visually detect rhythmic pulsing (rotor blades) or sudden accelerations (missile launch) over time.

### Threat Alert Panel
Uses **Pre-attentive Processing** principles:
- **Color Coding**: Red (High Risk) vs Yellow (Medium Risk) based on proximity and classification.
- **Textual Summary**: Condensed tactical data (ID, Range, Velocity) for rapid decision support.

## 3. Real-Time Interactions
The dashboard is designed for **10 Hz real-time updates**, ensuring that the visual representation accurately reflects the underlying photonic and kinematic simulation frames.
- **Tactical Sweep Trigger**: Manual trigger to simulate "Lock-on" or "Search" modes.
- **Scenario Injection**: Ability to switch between Swarm, Aircraft, and Missile presets without system restart.
