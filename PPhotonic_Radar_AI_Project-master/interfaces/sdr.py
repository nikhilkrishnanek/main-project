# ===== src/rtl_sdr_receiver.py =====
import numpy as np
try:
    from rtlsdr import RtlSdr
    HAS_RTLSDR = True
except (ImportError, Exception):
    HAS_RTLSDR = False

class RTLRadar:
    def __init__(self, center_freq=100e6, sample_rate=2.4e6, gain='auto'):
        if not HAS_RTLSDR:
            raise ImportError("librtlsdr not found. Please install it or use simulation mode.")
        self.sdr = RtlSdr()
        self.sdr.sample_rate = sample_rate
        self.sdr.center_freq = center_freq
        self.sdr.gain = gain

    def read_samples(self, n=4096):
        if not HAS_RTLSDR:
            return np.zeros(n)
        samples = self.sdr.read_samples(n)
        return samples

    def close(self):
        if HAS_RTLSDR and hasattr(self, 'sdr'):
            self.sdr.close()
