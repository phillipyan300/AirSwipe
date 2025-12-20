#!/usr/bin/env python3
"""
AirSwipe - Acoustic Gesture Sensing for macOS

Detects hand swipe gestures using ultrasonic audio and Doppler shift analysis.
Uses only the built-in MacBook speaker and microphone.

Usage:
    python airswipe.py --visualize     # Phase 0: See spectrogram
    python airswipe.py --detect        # Phase 1: Motion detection
    python airswipe.py --gestures      # Phase 2+: Full gesture recognition

Author: AirSwipe Project
License: MIT
"""

import argparse
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
from scipy.fft import rfft, rfftfreq

try:
    import sounddevice as sd
except OSError as e:
    print(f"Error loading sounddevice: {e}")
    print("Try: pip install sounddevice")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """AirSwipe configuration parameters."""
    
    # Audio settings
    sample_rate: int = 48000          # Hz - standard audio rate
    carrier_freq: float = 18500.0     # Hz - ultrasonic carrier (18-19.5 kHz range)
    tone_amplitude: float = 0.3       # 0-1, keep low to be unobtrusive
    
    # Analysis settings
    fft_size: int = 2048              # FFT window size
    hop_size: int = 512               # Samples between FFT windows
    window_duration: float = 0.5      # Seconds of audio to analyze
    
    # Detection settings
    motion_threshold: float = 2.0     # Motion score threshold
    cooldown_time: float = 0.8        # Seconds between gesture triggers
    doppler_bandwidth: float = 500.0  # Hz around carrier to analyze
    
    # Visualization
    spectrogram_duration: float = 3.0  # Seconds of history to show
    freq_range: tuple = (17000, 20000) # Hz range to display


# ============================================================================
# Gesture Types
# ============================================================================

class GestureType(Enum):
    """Recognized gesture types."""
    NONE = "none"
    SWIPE_TOWARD = "toward"   # Hand moving toward laptop
    SWIPE_AWAY = "away"       # Hand moving away from laptop
    SWIPE_LEFT = "left"       # Lateral left swipe
    SWIPE_RIGHT = "right"     # Lateral right swipe


# ============================================================================
# Audio Engine
# ============================================================================

class AudioEngine:
    """
    Handles audio I/O: tone generation and microphone capture.
    
    The engine plays a continuous ultrasonic tone while simultaneously
    recording from the microphone to capture reflections.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.audio_buffer = deque(maxlen=int(config.sample_rate * 5))  # 5 sec buffer
        self._stream = None
        self._phase = 0.0  # For continuous tone generation
        
    def _generate_tone(self, num_samples: int) -> np.ndarray:
        """Generate continuous sine wave at carrier frequency."""
        t = np.arange(num_samples) / self.config.sample_rate
        # Maintain phase continuity across calls
        phase_increment = 2 * np.pi * self.config.carrier_freq * num_samples / self.config.sample_rate
        samples = self.config.tone_amplitude * np.sin(
            2 * np.pi * self.config.carrier_freq * t + self._phase
        )
        self._phase = (self._phase + phase_increment) % (2 * np.pi)
        return samples.astype(np.float32)
    
    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """Combined input/output callback for simultaneous play and record."""
        if status:
            print(f"Audio status: {status}")
        
        # Generate output tone
        outdata[:, 0] = self._generate_tone(frames)
        
        # Store input samples
        self.audio_buffer.extend(indata[:, 0].copy())
    
    def start(self):
        """Start audio I/O stream."""
        if self.running:
            return
            
        print(f"🔊 Starting audio engine...")
        print(f"   Carrier frequency: {self.config.carrier_freq:.0f} Hz")
        print(f"   Sample rate: {self.config.sample_rate} Hz")
        
        self._stream = sd.Stream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=self.config.hop_size
        )
        self._stream.start()
        self.running = True
        print("   Audio stream started ✓")
    
    def stop(self):
        """Stop audio I/O stream."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.running = False
        print("🔇 Audio engine stopped")
    
    def get_recent_audio(self, duration: float) -> np.ndarray:
        """Get the most recent audio samples."""
        num_samples = int(duration * self.config.sample_rate)
        buffer_list = list(self.audio_buffer)
        if len(buffer_list) < num_samples:
            # Pad with zeros if not enough samples yet
            padding = np.zeros(num_samples - len(buffer_list))
            return np.concatenate([padding, buffer_list])
        return np.array(buffer_list[-num_samples:])


# ============================================================================
# Signal Processor
# ============================================================================

class SignalProcessor:
    """
    Processes audio to extract motion features.
    
    Computes spectrograms and analyzes energy distribution around
    the carrier frequency to detect Doppler shifts from motion.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._window = signal.windows.hann(config.fft_size)
        
    def compute_spectrogram(self, audio: np.ndarray) -> tuple:
        """
        Compute spectrogram of audio signal.
        
        Returns:
            freqs: Frequency bins
            times: Time bins  
            Sxx: Power spectral density
        """
        freqs, times, Sxx = signal.spectrogram(
            audio,
            fs=self.config.sample_rate,
            window=self._window,
            nperseg=self.config.fft_size,
            noverlap=self.config.fft_size - self.config.hop_size,
            scaling='density'
        )
        return freqs, times, Sxx
    
    def compute_motion_score(self, audio: np.ndarray) -> tuple:
        """
        Compute motion score based on energy spread around carrier.
        
        The idea: At rest, most energy is at the carrier frequency.
        With motion, energy spreads to adjacent frequencies (Doppler).
        
        Returns:
            score: Motion score (higher = more motion)
            doppler_shift: Estimated Doppler shift direction (-1, 0, +1)
        """
        # Compute FFT
        windowed = audio[-self.config.fft_size:] * self._window
        spectrum = np.abs(rfft(windowed))
        freqs = rfftfreq(self.config.fft_size, 1.0 / self.config.sample_rate)
        
        # Find carrier bin
        carrier_idx = np.argmin(np.abs(freqs - self.config.carrier_freq))
        
        # Define frequency bands
        bandwidth_bins = int(self.config.doppler_bandwidth / (self.config.sample_rate / self.config.fft_size))
        
        # Carrier energy (narrow band around carrier)
        carrier_start = max(0, carrier_idx - 2)
        carrier_end = min(len(spectrum), carrier_idx + 3)
        carrier_energy = np.sum(spectrum[carrier_start:carrier_end] ** 2)
        
        # Off-carrier energy (sidebands)
        lower_start = max(0, carrier_idx - bandwidth_bins)
        lower_end = carrier_start
        upper_start = carrier_end
        upper_end = min(len(spectrum), carrier_idx + bandwidth_bins)
        
        lower_energy = np.sum(spectrum[lower_start:lower_end] ** 2) if lower_end > lower_start else 0
        upper_energy = np.sum(spectrum[upper_start:upper_end] ** 2) if upper_end > upper_start else 0
        
        sideband_energy = lower_energy + upper_energy
        
        # Motion score: ratio of sideband to carrier energy
        if carrier_energy > 1e-10:
            score = sideband_energy / carrier_energy
        else:
            score = 0.0
        
        # Doppler direction: positive shift = toward, negative = away
        if lower_energy + upper_energy > 1e-10:
            doppler_shift = (upper_energy - lower_energy) / (upper_energy + lower_energy)
        else:
            doppler_shift = 0.0
            
        return score, doppler_shift
    
    def get_carrier_band_spectrum(self, audio: np.ndarray) -> tuple:
        """Get spectrum focused on carrier frequency band."""
        windowed = audio[-self.config.fft_size:] * self._window
        spectrum = np.abs(rfft(windowed))
        freqs = rfftfreq(self.config.fft_size, 1.0 / self.config.sample_rate)
        
        # Filter to frequency range of interest
        mask = (freqs >= self.config.freq_range[0]) & (freqs <= self.config.freq_range[1])
        return freqs[mask], spectrum[mask]


# ============================================================================
# Gesture Detector
# ============================================================================

class GestureDetector:
    """
    Detects and classifies gestures from motion scores.
    
    Uses temporal patterns and Doppler shift direction to
    classify swipe gestures.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.motion_history = deque(maxlen=50)  # Recent motion scores
        self.last_gesture_time = 0.0
        self.gesture_callback: Optional[Callable] = None
        
    def update(self, motion_score: float, doppler_shift: float) -> Optional[GestureType]:
        """
        Update detector with new motion data.
        
        Returns detected gesture if any.
        """
        current_time = time.time()
        self.motion_history.append((current_time, motion_score, doppler_shift))
        
        # Check cooldown
        if current_time - self.last_gesture_time < self.config.cooldown_time:
            return None
        
        # Detect motion burst
        if motion_score > self.config.motion_threshold:
            # Classify gesture based on Doppler direction
            gesture = self._classify_gesture(doppler_shift)
            
            if gesture != GestureType.NONE:
                self.last_gesture_time = current_time
                if self.gesture_callback:
                    self.gesture_callback(gesture)
                return gesture
                
        return None
    
    def _classify_gesture(self, doppler_shift: float) -> GestureType:
        """Classify gesture based on Doppler shift direction."""
        # Threshold for direction classification
        direction_threshold = 0.3
        
        if doppler_shift > direction_threshold:
            return GestureType.SWIPE_TOWARD
        elif doppler_shift < -direction_threshold:
            return GestureType.SWIPE_AWAY
        else:
            # Ambiguous direction - could be lateral swipe
            # For now, default to toward
            return GestureType.SWIPE_TOWARD


# ============================================================================
# System Actions
# ============================================================================

class SystemActions:
    """Execute macOS system actions in response to gestures."""
    
    @staticmethod
    def execute(gesture: GestureType):
        """Execute system action for gesture."""
        actions = {
            GestureType.SWIPE_RIGHT: SystemActions.next_desktop,
            GestureType.SWIPE_LEFT: SystemActions.prev_desktop,
            GestureType.SWIPE_TOWARD: SystemActions.mission_control,
            GestureType.SWIPE_AWAY: SystemActions.lock_screen,
        }
        
        action = actions.get(gesture)
        if action:
            print(f"🎯 Executing: {gesture.value}")
            action()
    
    @staticmethod
    def next_desktop():
        """Switch to next desktop/space."""
        # Simulate Ctrl+Right arrow
        script = '''
        tell application "System Events"
            key code 124 using control down
        end tell
        '''
        subprocess.run(['osascript', '-e', script], capture_output=True)
    
    @staticmethod
    def prev_desktop():
        """Switch to previous desktop/space."""
        script = '''
        tell application "System Events"
            key code 123 using control down
        end tell
        '''
        subprocess.run(['osascript', '-e', script], capture_output=True)
    
    @staticmethod
    def mission_control():
        """Open Mission Control."""
        script = '''
        tell application "System Events"
            key code 126 using control down
        end tell
        '''
        subprocess.run(['osascript', '-e', script], capture_output=True)
    
    @staticmethod
    def lock_screen():
        """Lock the screen."""
        subprocess.run(['pmset', 'displaysleepnow'], capture_output=True)


# ============================================================================
# Visualization
# ============================================================================

class SpectrogramVisualizer:
    """Real-time spectrogram visualization."""
    
    def __init__(self, config: Config, audio_engine: AudioEngine, processor: SignalProcessor):
        self.config = config
        self.audio_engine = audio_engine
        self.processor = processor
        self.fig = None
        self.ax_spec = None
        self.ax_score = None
        self.motion_scores = deque(maxlen=100)
        
    def start(self):
        """Start visualization."""
        # Create figure with dark theme
        plt.style.use('dark_background')
        self.fig, (self.ax_spec, self.ax_score) = plt.subplots(
            2, 1, figsize=(12, 8),
            gridspec_kw={'height_ratios': [3, 1]}
        )
        self.fig.suptitle('AirSwipe - Acoustic Gesture Sensing', fontsize=14, fontweight='bold')
        
        # Spectrogram axis
        self.ax_spec.set_ylabel('Frequency (Hz)')
        self.ax_spec.set_xlabel('Time (s)')
        self.ax_spec.set_title('Spectrogram (motion causes frequency spread)')
        
        # Motion score axis
        self.ax_score.set_ylabel('Motion Score')
        self.ax_score.set_xlabel('Time')
        self.ax_score.set_title('Motion Detection')
        self.ax_score.axhline(y=self.config.motion_threshold, color='r', 
                              linestyle='--', label='Threshold')
        self.ax_score.legend(loc='upper right')
        
        # Animation
        ani = FuncAnimation(self.fig, self._update, interval=50, blit=False, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()
        
    def _update(self, frame):
        """Update visualization frame."""
        # Get recent audio
        audio = self.audio_engine.get_recent_audio(self.config.spectrogram_duration)
        
        if len(audio) < self.config.fft_size:
            return
        
        # Compute spectrogram
        freqs, times, Sxx = self.processor.compute_spectrogram(audio)
        
        # Filter to frequency range
        freq_mask = (freqs >= self.config.freq_range[0]) & (freqs <= self.config.freq_range[1])
        
        # Compute motion score
        score, doppler = self.processor.compute_motion_score(audio)
        self.motion_scores.append(score)
        
        # Update spectrogram
        self.ax_spec.clear()
        self.ax_spec.pcolormesh(
            times, freqs[freq_mask], 
            10 * np.log10(Sxx[freq_mask, :] + 1e-10),
            shading='gouraud', cmap='magma'
        )
        self.ax_spec.axhline(y=self.config.carrier_freq, color='cyan', 
                             linestyle='--', alpha=0.5, label=f'Carrier ({self.config.carrier_freq:.0f} Hz)')
        self.ax_spec.set_ylabel('Frequency (Hz)')
        self.ax_spec.set_xlabel('Time (s)')
        self.ax_spec.set_title(f'Spectrogram | Doppler: {doppler:+.2f}')
        self.ax_spec.legend(loc='upper right')
        
        # Update motion score plot
        self.ax_score.clear()
        scores_list = list(self.motion_scores)
        self.ax_score.plot(scores_list, color='lime', linewidth=2)
        self.ax_score.axhline(y=self.config.motion_threshold, color='red', 
                              linestyle='--', alpha=0.7, label='Threshold')
        self.ax_score.set_ylabel('Motion Score')
        self.ax_score.set_xlabel('Samples')
        
        # Highlight if motion detected
        status = "🟢 MOTION DETECTED!" if score > self.config.motion_threshold else "⚪ Waiting..."
        self.ax_score.set_title(f'Motion Detection | Score: {score:.2f} | {status}')
        self.ax_score.set_ylim(0, max(5, max(scores_list) * 1.2) if scores_list else 5)
        self.ax_score.legend(loc='upper right')
        
        self.fig.canvas.draw_idle()


# ============================================================================
# Main Application
# ============================================================================

class AirSwipe:
    """Main AirSwipe application."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.audio_engine = AudioEngine(self.config)
        self.processor = SignalProcessor(self.config)
        self.detector = GestureDetector(self.config)
        self.running = False
        
    def run_visualize(self):
        """Run Phase 0: Visualization mode."""
        print("\n" + "=" * 60)
        print("  AirSwipe - Phase 0: Feasibility Testing")
        print("=" * 60)
        print("\nThis mode shows a real-time spectrogram.")
        print("Wave your hand in front of the laptop to see motion!")
        print("\nLook for:")
        print("  • Energy spread around the carrier frequency")
        print("  • Spikes in the motion score graph")
        print("\nPress Ctrl+C or close window to exit.\n")
        
        try:
            self.audio_engine.start()
            time.sleep(0.5)  # Let audio buffer fill
            
            visualizer = SpectrogramVisualizer(
                self.config, self.audio_engine, self.processor
            )
            visualizer.start()
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.audio_engine.stop()
    
    def run_detect(self):
        """Run Phase 1: Motion detection mode."""
        print("\n" + "=" * 60)
        print("  AirSwipe - Phase 1: Motion Detection")
        print("=" * 60)
        print("\nListening for motion...")
        print("Wave your hand toward/away from the laptop.")
        print("\nPress Ctrl+C to exit.\n")
        
        try:
            self.audio_engine.start()
            self.running = True
            
            while self.running:
                time.sleep(0.05)  # 50ms update rate
                
                audio = self.audio_engine.get_recent_audio(0.1)
                if len(audio) < self.config.fft_size:
                    continue
                    
                score, doppler = self.processor.compute_motion_score(audio)
                gesture = self.detector.update(score, doppler)
                
                # Print status
                bar_len = int(min(score, 10) * 5)
                bar = "█" * bar_len + "░" * (50 - bar_len)
                direction = "→" if doppler > 0 else "←" if doppler < 0 else "○"
                
                status = f"\rScore: [{bar}] {score:5.2f} {direction} "
                if gesture:
                    status += f"  🎯 {gesture.value.upper()}!"
                print(status, end="", flush=True)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            self.running = False
            self.audio_engine.stop()
    
    def run_gestures(self, execute_actions: bool = False):
        """Run Phase 2+: Full gesture recognition with optional system actions."""
        print("\n" + "=" * 60)
        print("  AirSwipe - Gesture Recognition")
        print("=" * 60)
        print("\nActive gesture detection enabled.")
        if execute_actions:
            print("⚠️  System actions ENABLED - gestures will trigger macOS actions!")
        else:
            print("ℹ️  System actions disabled (use --execute to enable)")
        print("\nGesture mappings:")
        print("  • Swipe toward → Mission Control")
        print("  • Swipe away   → Lock Screen")
        print("\nPress Ctrl+C to exit.\n")
        
        if execute_actions:
            self.detector.gesture_callback = SystemActions.execute
        
        try:
            self.audio_engine.start()
            self.running = True
            
            while self.running:
                time.sleep(0.05)
                
                audio = self.audio_engine.get_recent_audio(0.1)
                if len(audio) < self.config.fft_size:
                    continue
                    
                score, doppler = self.processor.compute_motion_score(audio)
                gesture = self.detector.update(score, doppler)
                
                # Status display
                if score > self.config.motion_threshold * 0.5:
                    print(f"\r🔊 Motion score: {score:.2f} | Doppler: {doppler:+.2f}    ", 
                          end="", flush=True)
                if gesture:
                    print(f"\n✨ Detected: {gesture.value}")
                    
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            self.running = False
            self.audio_engine.stop()


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AirSwipe - Acoustic Gesture Sensing for macOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python airswipe.py --visualize           # See spectrogram (Phase 0)
  python airswipe.py --detect              # Motion detection (Phase 1)
  python airswipe.py --gestures            # Gesture recognition (Phase 2)
  python airswipe.py --gestures --execute  # Enable system actions

  # Custom settings:
  python airswipe.py --visualize --freq 19000 --threshold 1.5
        """
    )
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--visualize', '-v', action='store_true',
                      help='Phase 0: Show real-time spectrogram')
    mode.add_argument('--detect', '-d', action='store_true',
                      help='Phase 1: Motion detection mode')
    mode.add_argument('--gestures', '-g', action='store_true',
                      help='Phase 2+: Full gesture recognition')
    
    # Configuration
    parser.add_argument('--freq', type=float, default=18500,
                        help='Carrier frequency in Hz (default: 18500)')
    parser.add_argument('--threshold', type=float, default=2.0,
                        help='Motion detection threshold (default: 2.0)')
    parser.add_argument('--amplitude', type=float, default=0.3,
                        help='Tone amplitude 0-1 (default: 0.3)')
    parser.add_argument('--execute', '-x', action='store_true',
                        help='Execute system actions on gesture detection')
    
    args = parser.parse_args()
    
    # Build config
    config = Config(
        carrier_freq=args.freq,
        motion_threshold=args.threshold,
        tone_amplitude=args.amplitude,
    )
    
    # Run appropriate mode
    app = AirSwipe(config)
    
    if args.visualize:
        app.run_visualize()
    elif args.detect:
        app.run_detect()
    elif args.gestures:
        app.run_gestures(execute_actions=args.execute)


if __name__ == "__main__":
    main()


