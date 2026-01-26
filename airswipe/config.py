"""
Configuration module for AirSwipe.

All tunable parameters in one place for easy experimentation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    """AirSwipe configuration parameters."""
    
    # ==========================================================================
    # Audio Settings
    # ==========================================================================
    sample_rate: int = 48000              # Hz - preferred for ultrasonic work
    carrier_freq: float = 18500.0         # Hz - ultrasonic carrier (17-19.5 kHz)
    tone_amplitude: float = 0.3           # 0-1, amplitude of output tone
    
    # Carrier scan candidates (Hz)
    carrier_candidates: List[float] = field(default_factory=lambda: [
        16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500
    ])
    
    # ==========================================================================
    # STFT Settings
    # ==========================================================================
    fft_size: int = 2048                  # N - FFT window size
    hop_size: int = 512                   # H - samples between frames
    window_type: str = "hann"             # Window function
    
    # Derived: freq resolution = sample_rate / fft_size ≈ 23.4 Hz at 48kHz
    # Derived: time resolution = hop_size / sample_rate ≈ 10.7 ms at 48kHz
    
    # ==========================================================================
    # Feature Extraction
    # ==========================================================================
    doppler_bandwidth: float = 500.0      # Hz - bandwidth around carrier for analysis
    
    # Background subtraction
    baseline_window_sec: float = 2.0      # Seconds of idle to compute baseline
    baseline_update_rate: float = 0.01    # EMA alpha for baseline update
    
    # ==========================================================================
    # Segmentation / Event Detection
    # ==========================================================================
    activity_threshold: float = 0.5       # A(t) threshold for event start
    min_event_duration_sec: float = 0.15  # Minimum gesture duration
    max_event_duration_sec: float = 1.0   # Maximum gesture duration
    cooldown_sec: float = 0.5             # Cooldown between gestures
    
    # ==========================================================================
    # Classification
    # ==========================================================================
    confidence_threshold: float = 0.6     # Minimum confidence for gesture
    smoothing_frames: int = 5             # Frames for majority voting
    
    # ==========================================================================
    # Model Settings (CNN)
    # ==========================================================================
    patch_time_frames: int = 64           # T - time frames in input patch
    patch_freq_bins: int = 64             # F - frequency bins in input patch
    
    # ==========================================================================
    # Tracking (Part 2)
    # ==========================================================================
    velocity_ema_alpha: float = 0.85      # EMA smoothing for v_hat
    position_decay: float = 0.98          # Drift control decay rate
    velocity_scale: float = 5.0           # Normalization scale for D(t)
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    spectrogram_history_sec: float = 3.0  # Seconds of spectrogram to display
    ui_update_interval_ms: int = 50       # UI refresh rate
    
    # ==========================================================================
    # Dataset Collection
    # ==========================================================================
    dataset_dir: str = "data"
    segment_duration_sec: float = 0.6     # Duration of recorded segments
    
    # ==========================================================================
    # Derived Properties
    # ==========================================================================
    @property
    def freq_resolution(self) -> float:
        """Frequency resolution in Hz."""
        return self.sample_rate / self.fft_size
    
    @property
    def time_resolution(self) -> float:
        """Time resolution in seconds."""
        return self.hop_size / self.sample_rate
    
    @property
    def min_event_frames(self) -> int:
        """Minimum event duration in frames."""
        return int(self.min_event_duration_sec / self.time_resolution)
    
    @property
    def max_event_frames(self) -> int:
        """Maximum event duration in frames."""
        return int(self.max_event_duration_sec / self.time_resolution)
    
    @property
    def cooldown_frames(self) -> int:
        """Cooldown duration in frames."""
        return int(self.cooldown_sec / self.time_resolution)
    
    @property
    def doppler_bins(self) -> int:
        """Number of frequency bins in doppler bandwidth."""
        return int(self.doppler_bandwidth / self.freq_resolution)
    
    def get_carrier_bin(self) -> int:
        """Get the FFT bin index for current carrier frequency."""
        return int(self.carrier_freq / self.freq_resolution)
    
    def get_band_slice(self) -> Tuple[int, int]:
        """Get (start, end) bin indices for the analysis band around carrier."""
        center = self.get_carrier_bin()
        half_width = self.doppler_bins
        return (center - half_width, center + half_width)
