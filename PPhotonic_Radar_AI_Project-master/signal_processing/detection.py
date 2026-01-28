"""
Radar Detection and Target Extraction Module
=============================================

This module implements the tactical detection stage of the radar signal chain. 
It converts processed spectral maps into discrete target observations through 
adaptive thresholding and spatial clustering.

Key Components:
---------------
1. CFAR (Constant False Alarm Rate) Detectors:
   - Cell-Averaging (CA-CFAR): Estimates the local noise floor by averaging 
     neighboring 'training' cells. Optimal in homogeneous Gaussian noise.
   - Ordered-Statistics (OS-CFAR): Uses the k-th rank statistic of training 
     cells. Resistant to 'target masking' in dense multi-target environments.

2. Centroiding & Clustering:
   - Since a single target may illuminate multiple Range-Doppler bins, 
     connected-component labeling is used to group adjacent detections. 
     The local peak of each cluster is selected as the representative observation.

Mathematical Foundation:
------------------------
The detection threshold (T) is defined as: T = alpha * P_noise
where 'alpha' is the threshold multiplier calculated to maintain a constant 
Probability of False Alarm (Pfa).

Author: Senior Radar Systems Engineer
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
from core.config import get_config
from core.logger import log_event
from signal_processing.transforms import compute_range_doppler_map


def matched_filter_synthesis(received_signal: np.ndarray, 
                             template_waveform: np.ndarray) -> np.ndarray:
    """
    Applies a matched filter via FFT-based circular convolution.
    
    This maximizes the SNR for a known pulse shape (e.g., LFM chirp) in 
    stationary Gaussian noise.
    """
    n_conv = len(received_signal) + len(template_waveform) - 1
    # Use next power of 2 for FFT efficiency
    n_fft = 1 << (n_conv - 1).bit_length()
    
    spectrum_rx = np.fft.fft(received_signal, n_fft)
    spectrum_tx_matched = np.fft.fft(np.conj(template_waveform[::-1]), n_fft)
    
    filtered_output = np.fft.ifft(spectrum_rx * spectrum_tx_matched)[:n_conv]
    return np.abs(filtered_output)


def ca_cfar_detector(range_doppler_intensity: np.ndarray, 
                     guard_cells: int = 2, 
                     training_cells: int = 8, 
                     pfa_target: float = 1e-4, 
                     power_floor: float = 1e-3, 
                     cognitive_alpha_scale: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Refined 2D Cell-Averaging CFAR.
    
    Calculates threshold 'alpha' based on the specified Pfa and the 
    number of training cells available in the reference window.
    """
    from scipy.signal import fftconvolve

    # 1. Window Geometry
    # Full window includes the Cell-Under-Test (CUT), Guard, and Training regions
    window_side = 2 * training_cells + 2 * guard_cells + 1
    guard_side = 2 * guard_cells + 1

    # Convolution kernels (Masking)
    kernel_full = np.ones((window_side, window_side), dtype=float)
    kernel_guard = np.ones((guard_side, guard_side), dtype=float)

    # 2. Local Noise Floor Estimation
    # sum_full: sum of all cells in the sliding window
    # sum_guard: sum of cells in the guard + CUT region
    sum_full = fftconvolve(range_doppler_intensity, kernel_full, mode="same")
    sum_guard = fftconvolve(range_doppler_intensity, kernel_guard, mode="same")

    n_training = (window_side**2) - (guard_side**2)
    n_training = max(1, n_training)
    
    # 3. Alpha Thresholding Calculation (Square-law detector assumption)
    # Ref: α = N * (Pfa^(-1/N) - 1)
    alpha = n_training * (pfa_target ** (-1.0 / n_training) - 1.0)
    alpha *= cognitive_alpha_scale

    # Estimated noise power (Average of training cells)
    training_total_power = sum_full - sum_guard
    local_noise_estimate = training_total_power / n_training
    local_noise_estimate[local_noise_estimate <= 0] = 1e-12

    # 4. Binary Decision
    detection_threshold = local_noise_estimate * alpha
    detection_binary_map = (range_doppler_intensity > detection_threshold) & (range_doppler_intensity > power_floor)

    return detection_binary_map, float(alpha)


def os_cfar_detector(range_doppler_intensity: np.ndarray, 
                     guard_cells: int = 2, 
                     training_cells: int = 8, 
                     pfa_target: float = 1e-6, 
                     rank_k: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Ordered-Statistics CFAR for outlier-tolerant noise estimation.
    """
    from scipy.ndimage import generic_filter
    
    window_side = 2 * training_cells + 2 * guard_cells + 1
    guard_side = 2 * guard_cells + 1
    n_training = (window_side**2) - (guard_side**2)
    
    # Selection of k-th rank (typically ~75th percentile for multi-target robustness)
    if rank_k is None:
        rank_k = int(0.75 * n_training)
    
    # Define hollow footprint mask (exclude guard/CUT)
    footprint = np.ones((window_side, window_side), dtype=bool)
    idx_start = training_cells
    idx_end = training_cells + guard_side
    footprint[idx_start:idx_end, idx_start:idx_end] = False
    
    def calculate_kth_statistic(buffer):
        sorted_samples = np.sort(buffer)
        return sorted_samples[rank_k - 1]

    # Non-linear sliding window filter
    noise_statistics_map = generic_filter(range_doppler_intensity, calculate_kth_statistic, 
                                        footprint=footprint, mode='constant', cval=0.0)
    
    # Empirical alpha scaling for target Pfa (approximate)
    alpha_os = 10**(3.5 / 10) 
    
    detection_threshold = noise_statistics_map * alpha_os
    detection_binary_map = range_doppler_intensity > detection_threshold
    
    return detection_binary_map, float(alpha_os)


def cluster_and_centroid_detections(detection_map: np.ndarray, 
                                   intensity_map: np.ndarray) -> List[Tuple[int, int]]:
    """
    Aggregates point-detections into target clusters and identifies local peaks.
    
    Returns a list of (row_idx, col_idx) for the centroid of each target.
    """
    from scipy.ndimage import label
    
    if not np.any(detection_map):
        return []
        
    labeled_regions, num_clusters = label(detection_map)
    target_centroids = []
    
    for cluster_id in range(1, num_clusters + 1):
        # Coordinates belonging to the current cluster
        pixel_coords = np.argwhere(labeled_regions == cluster_id)
        
        # Centroiding strategy: Peak Intensity Selection (Subpixel estimation could follow)
        intensities = [intensity_map[r, c] for r, c in pixel_coords]
        peak_idx = np.argmax(intensities)
        target_centroids.append(tuple(pixel_coords[peak_idx]))
        
    return target_centroids


def execute_detection_pipeline(beat_signal_complex: np.ndarray, 
                               num_pulses: int = 64, 
                               samples_per_pulse: int = 64, 
                               sampling_rate_hz: float = 4096, 
                               n_fft_range: int = 128, 
                               n_fft_doppler: int = 128,
                               algorithm: str = "ca", 
                               enable_clustering: bool = True, 
                               **kwargs) -> Dict:
    """
    Standardized entry point for the Radar Detection stage.
    """
    if beat_signal_complex is None or len(beat_signal_complex) == 0:
        log_event("[DETECTION] Null input received.", level="warning")
        return {
            "rd_map": np.zeros((n_fft_doppler, n_fft_range)),
            "det_map": np.zeros((n_fft_doppler, n_fft_range), dtype=bool),
            "detections": [],
            "stats": {"num_detections": 0, "error": "Empty signal"}
        }

    # 1. Transform to Range-Doppler domain
    range_doppler_intensity = compute_range_doppler_map(
        beat_signal_complex, 
        num_pulses=num_pulses, 
        samples_per_pulse=samples_per_pulse,
        n_fft_range=n_fft_range, 
        n_fft_doppler=n_fft_doppler
    )

    # 2. Apply Thresholding Algorithm
    if algorithm.lower().startswith("ca"):
        detection_map, alpha = ca_cfar_detector(range_doppler_intensity, **kwargs)
    else:
        detection_map, alpha = os_cfar_detector(range_doppler_intensity, **kwargs)

    # 3. Geometric Post-Processing
    if enable_clustering:
        peak_coords = cluster_and_centroid_detections(detection_map, range_doppler_intensity)
    else:
        peak_coords = list(zip(*np.where(detection_map)))
        
    # Format detections as (Doppler_Idx, Range_Idx, Intensity_dB)
    formatted_detections = [(int(r), int(c), float(range_doppler_intensity[r, c])) for r, c in peak_coords]

    pipeline_stats = {
        "num_detections": len(formatted_detections),
        "threshold_multiplier": alpha,
        "is_clustered": enable_clustering
    }

    log_event(f"[DETECTION] Completed. Detections identified: {pipeline_stats['num_detections']}", level="info")

    return {
        "rd_map": range_doppler_intensity,
        "det_map": detection_map,
        "detections": formatted_detections,
        "stats": pipeline_stats
    }
