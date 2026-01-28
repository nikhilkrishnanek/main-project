
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Slot

class DopplerWaterfall(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Configure Plot
        self.plot_widget = pg.PlotWidget(title="Doppler-Time Waterfall")
        self.plot_widget.setLabel('left', 'Time', units='frames')
        self.plot_widget.setLabel('bottom', 'Doppler Speed', units='m/s')
        self.layout.addWidget(self.plot_widget)
        
        # Image Item for Heatmap
        self.img = pg.ImageItem()
        self.plot_widget.addItem(self.img)
        
        # Colormap (Viridis-like)
        # simplistic lookup table
        pos = np.array([0.0, 0.5, 1.0])
        color = np.array([[0,0,0,255], [0, 255, 0, 255], [255, 255, 0, 255]], dtype=np.ubyte)
        map = pg.ColorMap(pos, color)
        self.img.setLookupTable(map.getLookupTable(0.0, 1.0, 256))
        
        # Data Buffer
        self.history_len = 100
        self.n_doppler = 64 # Default
        self.buffer = np.zeros((self.n_doppler, self.history_len))

    @Slot(dict)
    def update_display(self, frame_data):
        rd_map = frame_data.get('rd_map')
        if rd_map is None:
            return
            
        # Sum over Range to get Doppler Profile
        # rd_map shape: (Doppler, Range) or (Range, Doppler)? 
        # In orchestrator: rd_map = self.dsp.process_frame(pulse_matrix)
        # RadarDSPEngine usually returns (Doppler, Range) after fft2 and fftshift?
        # Let's assume axes 0 is Doppler, 1 is Range.
        
        doppler_profile = np.sum(rd_map, axis=1)
        
        # Normalize
        norm = np.max(doppler_profile) + 1e-9
        doppler_profile = doppler_profile / norm
        
        # 1. Check and Resize Buffer if needed
        if len(doppler_profile) != self.buffer.shape[0]:
            self.n_doppler = len(doppler_profile)
            self.buffer = np.zeros((self.n_doppler, self.history_len))
            
        # 2. Update Buffer
        self.buffer = np.roll(self.buffer, -1, axis=1)
        self.buffer[:, -1] = doppler_profile
        
        self.img.setImage(self.buffer.T) # Transpose for Time on Y
        
        # Scale X axis (placeholder values)
        # We need v_scale from bridge or frame_data to be accurate
