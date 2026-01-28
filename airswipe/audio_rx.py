"""
Audio reception module - captures microphone input.

Provides frame-based access to recorded audio for DSP processing.
"""

import numpy as np
import threading
from collections import deque
from typing import Optional

try:
    import sounddevice as sd
except ImportError:
    raise ImportError("sounddevice required: pip install sounddevice")

from .config import Config


class AudioRx:
    """
    Microphone input receiver.
    
    Captures audio from the system microphone and provides frame-based
    access for STFT processing. Uses a circular buffer to store recent
    audio history.
    """
    
    def __init__(self, config: Config, buffer_duration: float = 5.0):
        """
        Initialize receiver.
        
        Args:
            config: AirSwipe configuration
            buffer_duration: Seconds of audio history to retain
        """
        self.config = config
        self._stream: Optional[sd.InputStream] = None
        self._running: bool = False
        
        # Circular buffer for audio samples
        buffer_size = int(buffer_duration * config.sample_rate)
        self._buffer = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        
        # Frame counter for timing
        self._frame_count: int = 0
        
        # Callback for real-time processing
        self._frame_callback = None
    
    def _input_callback(self, indata: np.ndarray, frames: int,
                        time_info, status):
        """Sounddevice input callback."""
        if status:
            print(f"[RX] Input status: {status}")
        
        # Extract mono channel
        samples = indata[:, 0].copy()
        
        with self._lock:
            self._buffer.extend(samples)
            self._frame_count += 1
        
        # Notify callback if registered
        if self._frame_callback is not None:
            self._frame_callback(samples)
    
    def start(self):
        """Start recording from microphone."""
        if self._running:
            return
        
        print(f"[RX] Starting microphone capture at {self.config.sample_rate} Hz")
        
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self._input_callback,
            blocksize=self.config.hop_size,
        )
        self._stream.start()
        self._running = True
        print("[RX] Recording started ✓")
    
    def stop(self):
        """Stop recording."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        print("[RX] Recording stopped")
    
    def get_buffer(self, duration: float) -> np.ndarray:
        """
        Get the most recent audio samples.
        
        Args:
            duration: Seconds of audio to retrieve
            
        Returns:
            Audio samples as numpy array
        """
        num_samples = int(duration * self.config.sample_rate)
        
        with self._lock:
            buffer_list = list(self._buffer)
        
        if len(buffer_list) < num_samples:
            # Pad with zeros if not enough samples
            padding = np.zeros(num_samples - len(buffer_list), dtype=np.float32)
            return np.concatenate([padding, buffer_list])
        
        return np.array(buffer_list[-num_samples:], dtype=np.float32)
    
    def get_frames(self, num_frames: int) -> np.ndarray:
        """
        Get most recent audio as frame-aligned chunks.
        
        Args:
            num_frames: Number of hop-sized frames to retrieve
            
        Returns:
            Audio samples covering num_frames * hop_size samples
        """
        num_samples = num_frames * self.config.hop_size
        return self.get_buffer(num_samples / self.config.sample_rate)
    
    def set_frame_callback(self, callback):
        """
        Register callback for real-time frame processing.
        
        The callback receives each new audio block as it arrives.
        
        Args:
            callback: Function(samples: np.ndarray) -> None
        """
        self._frame_callback = callback
    
    @property
    def is_running(self) -> bool:
        """Check if receiver is active."""
        return self._running
    
    @property
    def buffer_level(self) -> float:
        """Current buffer fill level (0-1)."""
        with self._lock:
            return len(self._buffer) / self._buffer.maxlen
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


class AudioDuplex:
    """
    Combined transmitter and receiver for simultaneous play/record.
    
    Uses a single full-duplex stream for tighter synchronization.
    """
    
    def __init__(self, config: Config, buffer_duration: float = 5.0):
        self.config = config
        self._stream: Optional[sd.Stream] = None
        self._running: bool = False
        
        # Circular buffer for received audio
        buffer_size = int(buffer_duration * config.sample_rate)
        self._buffer = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        
        # Tone generation state
        self._phase: float = 0.0
        self._carrier_freq = config.carrier_freq
        self._amplitude = config.tone_amplitude
        
        # Callback for real-time processing
        self._frame_callback = None
    
    @property
    def carrier_freq(self) -> float:
        return self._carrier_freq
    
    @carrier_freq.setter
    def carrier_freq(self, freq: float):
        with self._lock:
            self._carrier_freq = freq
    
    def _duplex_callback(self, indata: np.ndarray, outdata: np.ndarray,
                         frames: int, time_info, status):
        """Combined I/O callback."""
        # Only print occasional overflow warnings, not every single one
        if status and 'overflow' not in str(status):
            print(f"[DUPLEX] Status: {status}")
        
        # === OUTPUT: Generate tone ===
        with self._lock:
            freq = self._carrier_freq
            amp = self._amplitude
        
        t = np.arange(frames) / self.config.sample_rate
        outdata[:, 0] = amp * np.sin(2 * np.pi * freq * t + self._phase)
        
        phase_increment = 2 * np.pi * freq * frames / self.config.sample_rate
        self._phase = (self._phase + phase_increment) % (2 * np.pi)
        
        # === INPUT: Store samples ===
        samples = indata[:, 0].copy()
        with self._lock:
            self._buffer.extend(samples)
        
        if self._frame_callback is not None:
            self._frame_callback(samples)
    
    def start(self):
        """Start duplex audio stream."""
        if self._running:
            return
        
        print(f"[DUPLEX] Starting at {self.config.sample_rate} Hz, "
              f"carrier {self._carrier_freq:.0f} Hz")
        
        # Use larger blocksize to prevent buffer overflow
        self._stream = sd.Stream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self._duplex_callback,
            blocksize=2048,  # Larger block for stability
            latency='high',  # Prioritize stability over latency
        )
        self._stream.start()
        self._running = True
        print("[DUPLEX] Stream started ✓")
    
    def stop(self):
        """Stop duplex stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        self._phase = 0.0
        print("[DUPLEX] Stream stopped")
    
    def get_buffer(self, duration: float) -> np.ndarray:
        """Get most recent audio samples."""
        num_samples = int(duration * self.config.sample_rate)
        
        with self._lock:
            buffer_list = list(self._buffer)
        
        if len(buffer_list) < num_samples:
            padding = np.zeros(num_samples - len(buffer_list), dtype=np.float32)
            return np.concatenate([padding, buffer_list])
        
        return np.array(buffer_list[-num_samples:], dtype=np.float32)
    
    def set_frame_callback(self, callback):
        """Register callback for real-time processing."""
        self._frame_callback = callback
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
