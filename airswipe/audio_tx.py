"""
Audio transmission module - generates and plays ultrasonic carrier tone.

Handles continuous tone generation with phase continuity for clean output.
"""

import numpy as np
import threading
from typing import Optional

try:
    import sounddevice as sd
except ImportError:
    raise ImportError("sounddevice required: pip install sounddevice")

from .config import Config


class AudioTx:
    """
    Ultrasonic tone transmitter.
    
    Generates a continuous sine wave at the carrier frequency and plays
    it through the system speakers. Maintains phase continuity across
    audio buffer callbacks.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._stream: Optional[sd.OutputStream] = None
        self._phase: float = 0.0
        self._running: bool = False
        self._lock = threading.Lock()
        
        # Current carrier (can be updated after carrier scan)
        self._carrier_freq = config.carrier_freq
        self._amplitude = config.tone_amplitude
    
    @property
    def carrier_freq(self) -> float:
        """Current carrier frequency."""
        return self._carrier_freq
    
    @carrier_freq.setter
    def carrier_freq(self, freq: float):
        """Update carrier frequency (thread-safe)."""
        with self._lock:
            self._carrier_freq = freq
    
    @property
    def amplitude(self) -> float:
        """Current tone amplitude."""
        return self._amplitude
    
    @amplitude.setter
    def amplitude(self, amp: float):
        """Update amplitude (thread-safe)."""
        with self._lock:
            self._amplitude = np.clip(amp, 0.0, 1.0)
    
    def _generate_samples(self, num_frames: int) -> np.ndarray:
        """
        Generate continuous sine wave samples.
        
        Maintains phase continuity between calls for clean audio.
        """
        with self._lock:
            freq = self._carrier_freq
            amp = self._amplitude
        
        # Time indices for this block
        t = np.arange(num_frames) / self.config.sample_rate
        
        # Generate sine wave with current phase offset
        samples = amp * np.sin(2 * np.pi * freq * t + self._phase)
        
        # Update phase for next block (maintain continuity)
        phase_increment = 2 * np.pi * freq * num_frames / self.config.sample_rate
        self._phase = (self._phase + phase_increment) % (2 * np.pi)
        
        return samples.astype(np.float32)
    
    def _output_callback(self, outdata: np.ndarray, frames: int,
                         time_info, status):
        """Sounddevice output callback."""
        if status:
            print(f"[TX] Output status: {status}")
        
        outdata[:, 0] = self._generate_samples(frames)
    
    def start(self):
        """Start playing the carrier tone."""
        if self._running:
            return
        
        print(f"[TX] Starting tone at {self._carrier_freq:.0f} Hz, "
              f"amplitude {self._amplitude:.2f}")
        
        self._stream = sd.OutputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self._output_callback,
            blocksize=self.config.hop_size,
        )
        self._stream.start()
        self._running = True
        print("[TX] Tone started ✓")
    
    def stop(self):
        """Stop playing the carrier tone."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        self._phase = 0.0
        print("[TX] Tone stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if transmitter is active."""
        return self._running
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
