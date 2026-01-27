"""
Data Integration Interfaces
===========================

Hooks for integrating real-world radar datasets and field recordings.
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseDataLoader(ABC):
    """Interface for loading external radar data (HDF5, MAT, Bin)."""
    
    @abstractmethod
    def load_field_recording(self, file_path: str) -> np.ndarray:
        """Load raw ADC recording from field trials."""
        pass

    @abstractmethod
    def sync_with_simulation(self, simulated: np.ndarray, real: np.ndarray):
        """Cross-validate simulation results with experimental data."""
        pass
