"""
Hardware Integration Interfaces
===============================

Abstract base classes and placeholders for physical hardware integration.
Supports SDR, FPGA, and Optical Hardware Control.
"""

from abc import ABC, abstractmethod
import numpy as np

class SDRInterface(ABC):
    """Placeholder for high-speed I/Q data acquisition from SDR/Digitizers."""
    
    @abstractmethod
    def connect(self, address: str):
        """Establish connection to SDR (e.g., USRP, LabView-FPGA)."""
        pass

    @abstractmethod
    def stream_iq(self, num_samples: int) -> np.ndarray:
        """Fetch raw I/Q samples from the hardware buffer."""
        pass

class OpticalController(ABC):
    """Interface for controlling Photonic Bench components (Laser, MZM, PD)."""
    
    @abstractmethod
    def set_laser_power(self, power_dbm: float):
        """Control tunable laser power via SCPI/GPIB."""
        pass

    @abstractmethod
    def set_mzm_bias(self, voltage: float):
        """Adjust Mach-Zehnder Modulator bias for optimal linearity."""
        pass

class FPGAAcceleration(ABC):
    """Hooks for offloading FFT/DSP to FPGA gateware."""
    
    @abstractmethod
    def compute_rd_map_hdl(self, signal: np.ndarray) -> np.ndarray:
        """Offload Range-Doppler computation to FPGA via PCIe/DMA."""
        pass
