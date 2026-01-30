"""
Digital Signal Processing module for AirSwipe.

Handles STFT computation, feature extraction, and background subtraction.
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from typing import Tuple, Optional
from dataclasses import dataclass

from .config import Config


@dataclass
class DopplerFeatures:
    """Features extracted from a single time frame."""
    e_above: float      # Energy above carrier (approach)
    e_below: float      # Energy below carrier (recede)
    d: float            # Signed Doppler proxy (e_above - e_below)
    a: float            # Activity level (e_above + e_below)
    carrier_power: float  # Power at carrier frequency
    
    @property
    def direction(self) -> int:
        """Movement direction: +1 toward, -1 away, 0 ambiguous."""
        if self.a < 1e-6:
            return 0
        ratio = self.d / (self.a + 1e-10)
        if ratio > 0.3:
            return 1
        elif ratio < -0.3:
            return -1
        return 0


class DSP:
    """
    Signal processor for Doppler feature extraction.
    
    Computes STFT and extracts energy features around the carrier
    frequency to detect motion-induced Doppler shifts.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Pre-compute window function
        self._window = signal.windows.get_window(
            config.window_type, config.fft_size
        )
        
        # Pre-compute frequency bins
        self._freqs = rfftfreq(config.fft_size, 1.0 / config.sample_rate)
        
        # Background baseline (running median)
        self._baseline: Optional[np.ndarray] = None
        self._baseline_samples: list = []
        self._baseline_ready = False
        
        # Cache bin indices for carrier band
        self._update_carrier_bins()
    
    def _update_carrier_bins(self):
        """Update frequency bin indices for current carrier."""
        cf = self.config.carrier_freq
        bw = self.config.doppler_bandwidth
        
        # Find carrier bin
        self._carrier_idx = np.argmin(np.abs(self._freqs - cf))
        
        # Band boundaries
        self._band_low = np.argmin(np.abs(self._freqs - (cf - bw)))
        self._band_high = np.argmin(np.abs(self._freqs - (cf + bw)))
        
        # Above/below carrier (excluding carrier itself)
        carrier_margin = 2  # Exclude ±2 bins around carrier peak
        self._below_slice = slice(self._band_low, self._carrier_idx - carrier_margin)
        self._above_slice = slice(self._carrier_idx + carrier_margin, self._band_high)
        self._carrier_slice = slice(
            self._carrier_idx - carrier_margin,
            self._carrier_idx + carrier_margin + 1
        )
    
    def set_carrier(self, freq: float):
        """Update carrier frequency and recalculate bins."""
        self.config.carrier_freq = freq
        self._update_carrier_bins()
        self._reset_baseline()
    
    def _reset_baseline(self):
        """Reset background baseline."""
        self._baseline = None
        self._baseline_samples = []
        self._baseline_ready = False
    
    def compute_spectrum(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute magnitude spectrum of audio.
        
        Args:
            audio: Audio samples (at least fft_size samples)
            
        Returns:
            Magnitude spectrum (linear scale)
        """
        # Take last fft_size samples
        segment = audio[-self.config.fft_size:]
        
        # Apply window and compute FFT
        windowed = segment * self._window
        spectrum = np.abs(rfft(windowed))
        
        return spectrum
    
    def compute_log_spectrum(self, audio: np.ndarray, 
                             floor_db: float = -80) -> np.ndarray:
        """
        Compute log-magnitude spectrum.
        
        Args:
            audio: Audio samples
            floor_db: Minimum dB value
            
        Returns:
            Log-magnitude spectrum in dB
        """
        spectrum = self.compute_spectrum(audio)
        # Convert to dB with floor
        log_spec = 20 * np.log10(spectrum + 1e-10)
        log_spec = np.maximum(log_spec, floor_db)
        return log_spec
    
    def compute_spectrogram(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram of audio signal.
        
        Args:
            audio: Audio samples
            
        Returns:
            (times, freqs, Sxx) - time bins, frequency bins, power spectral density
            Sxx shape is (n_freqs, n_times) for use with pcolormesh(times, freqs, Sxx)
        """
        freqs, times, Sxx = signal.spectrogram(
            audio,
            fs=self.config.sample_rate,
            window=self._window,
            nperseg=self.config.fft_size,
            noverlap=self.config.fft_size - self.config.hop_size,
            scaling='density',
            mode='magnitude'
        )
        # scipy returns (freqs, times, Sxx) with Sxx shape (n_freqs, n_times)
        return times, freqs, Sxx
    
    def update_baseline(self, spectrum: np.ndarray):
        """
        Update background baseline with new spectrum.
        
        Call this during idle periods to build baseline.
        """
        # Store samples until we have enough
        baseline_frames = int(
            self.config.baseline_window_sec / self.config.time_resolution
        )
        
        if len(self._baseline_samples) < baseline_frames:
            self._baseline_samples.append(spectrum.copy())
        
        if len(self._baseline_samples) >= baseline_frames:
            # Compute median baseline
            stacked = np.stack(self._baseline_samples, axis=0)
            self._baseline = np.median(stacked, axis=0)
            self._baseline_ready = True
    
    def subtract_baseline(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Subtract background baseline from spectrum.
        
        Args:
            spectrum: Input magnitude spectrum
            
        Returns:
            Background-subtracted spectrum (non-negative)
        """
        if self._baseline is None:
            return spectrum
        
        # Subtract and clip to non-negative
        subtracted = spectrum - self._baseline
        return np.maximum(subtracted, 0)
    
    def extract_features(self, audio: np.ndarray, 
                        use_baseline: bool = False) -> DopplerFeatures:
        """
        Extract Doppler features from audio.
        
        Args:
            audio: Audio samples (at least fft_size)
            use_baseline: Whether to apply background subtraction.
                          Default False to match training data collection.
            
        Returns:
            DopplerFeatures containing e_above, e_below, d, a
        """
        spectrum = self.compute_spectrum(audio)
        
        if use_baseline and self._baseline is not None:
            spectrum = self.subtract_baseline(spectrum)
        
        # Extract band energies (sum of squared magnitudes)
        below = spectrum[self._below_slice]
        above = spectrum[self._above_slice]
        carrier = spectrum[self._carrier_slice]
        
        e_below = np.sum(below ** 2)
        e_above = np.sum(above ** 2)
        carrier_power = np.sum(carrier ** 2)
        
        # Doppler features
        d = e_above - e_below  # Signed difference
        a = e_above + e_below  # Total activity
        
        return DopplerFeatures(
            e_above=e_above,
            e_below=e_below,
            d=d,
            a=a,
            carrier_power=carrier_power
        )
    
    def get_carrier_band(self, spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get spectrum focused on carrier frequency band.
        
        Args:
            spectrum: Full magnitude spectrum
            
        Returns:
            (freqs, magnitudes) for the carrier band only
        """
        band_slice = slice(self._band_low, self._band_high)
        return self._freqs[band_slice], spectrum[band_slice]
    
    @property
    def freqs(self) -> np.ndarray:
        """Frequency bins for full spectrum."""
        return self._freqs
    
    @property
    def baseline_ready(self) -> bool:
        """Whether baseline has been computed."""
        return self._baseline_ready


class CarrierScanner:
    """
    Automatic carrier frequency selection.
    
    Tests candidate frequencies and selects the one with best
    SNR (signal-to-noise ratio) for the specific hardware.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._dsp = DSP(config)
    
    def measure_ambient(self, audio_rx, duration: float = 5.0) -> np.ndarray:
        """
        Measure ambient noise spectrum (no tone playing).
        
        Args:
            audio_rx: AudioRx instance (must be running)
            duration: Seconds to record
            
        Returns:
            Average magnitude spectrum of ambient noise
        """
        import time
        
        print(f"[SCAN] Recording ambient noise for {duration:.1f}s...")
        time.sleep(duration)
        
        audio = audio_rx.get_buffer(duration)
        _, _, Sxx = self._dsp.compute_spectrogram(audio)
        
        # Average over time
        ambient_spectrum = np.mean(Sxx, axis=1)
        return ambient_spectrum
    
    def score_carrier(self, audio_duplex, freq: float, 
                     duration: float = 1.5) -> Tuple[float, float]:
        """
        Score a candidate carrier frequency.
        
        Args:
            audio_duplex: AudioDuplex instance
            freq: Carrier frequency to test
            duration: Test duration
            
        Returns:
            (snr_score, stability_score)
        """
        import time
        
        # Set carrier frequency
        audio_duplex.carrier_freq = freq
        time.sleep(0.2)  # Let it stabilize
        
        # Record
        time.sleep(duration)
        audio = audio_duplex.get_buffer(duration)
        
        # Compute spectrum
        _, freqs, Sxx = self._dsp.compute_spectrogram(audio)
        mean_spectrum = np.mean(Sxx, axis=1)
        
        # Find peak near expected carrier
        carrier_idx = np.argmin(np.abs(freqs - freq))
        peak_region = slice(max(0, carrier_idx - 5), carrier_idx + 6)
        
        peak_power = np.max(mean_spectrum[peak_region])
        
        # Noise floor: median of surrounding ±1kHz excluding peak
        noise_low = np.argmin(np.abs(freqs - (freq - 1000)))
        noise_high = np.argmin(np.abs(freqs - (freq + 1000)))
        
        noise_region = np.concatenate([
            mean_spectrum[noise_low:carrier_idx - 10],
            mean_spectrum[carrier_idx + 10:noise_high]
        ])
        noise_floor = np.median(noise_region) if len(noise_region) > 0 else 1e-10
        
        # SNR in dB
        snr = 10 * np.log10(peak_power / (noise_floor + 1e-10))
        
        # Stability: variance of peak power over time
        time_slices = Sxx[peak_region, :]
        peak_over_time = np.max(time_slices, axis=0)
        stability = 1.0 / (np.std(peak_over_time) / np.mean(peak_over_time) + 0.1)
        
        return snr, stability
    
    def scan(self, audio_duplex, candidates: Optional[list] = None) -> Tuple[float, dict]:
        """
        Scan carrier frequencies and select best one.
        
        Args:
            audio_duplex: AudioDuplex instance (must be running)
            candidates: List of frequencies to test (uses config default if None)
            
        Returns:
            (best_freq, results_dict)
        """
        if candidates is None:
            candidates = self.config.carrier_candidates
        
        print(f"[SCAN] Testing {len(candidates)} carrier frequencies...")
        
        results = {}
        for freq in candidates:
            print(f"[SCAN] Testing {freq:.0f} Hz...", end=" ", flush=True)
            snr, stability = self.score_carrier(audio_duplex, freq)
            
            # Combined score
            score = snr * 0.7 + stability * 0.3
            results[freq] = {
                'snr': snr,
                'stability': stability,
                'score': score
            }
            print(f"SNR={snr:.1f}dB, stability={stability:.2f}, score={score:.2f}")
        
        # Select best
        best_freq = max(results, key=lambda f: results[f]['score'])
        print(f"\n[SCAN] Best carrier: {best_freq:.0f} Hz "
              f"(SNR={results[best_freq]['snr']:.1f}dB)")
        
        return best_freq, results
