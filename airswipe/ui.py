"""
User interface module for AirSwipe.

Real-time visualization of spectrogram, features, and motion tracking.
"""

import numpy as np
import time
from typing import Optional, Callable
from collections import deque

from .config import Config
from .dsp import DopplerFeatures
from .tracker import TrackingState
from .segmentation import GestureLabel, GestureEvent


class ConsoleUI:
    """
    Simple console-based UI for terminal display.
    
    Shows real-time motion bars and gesture detection.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._last_gesture: Optional[GestureEvent] = None
        self._gesture_display_time = 0.0
    
    def update(self, features: DopplerFeatures, 
               tracking: Optional[TrackingState] = None,
               gesture: Optional[GestureEvent] = None):
        """Update console display."""
        
        if gesture is not None:
            self._last_gesture = gesture
            self._gesture_display_time = time.time()
        
        # Build activity bar
        activity = min(features.a / (self.config.activity_threshold * 5), 1.0)
        bar_len = int(activity * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        
        # Direction indicator
        if features.d > 0:
            direction = "◀ LEFT "
        elif features.d < 0:
            direction = " RIGHT▶"
        else:
            direction = "   ○   "
        
        # Velocity bar (centered)
        if tracking:
            v = tracking.v_hat
            v_pos = int((v + 1) * 20)  # Map [-1,1] to [0,40]
            v_bar = "─" * 20 + "│" + "─" * 20
            v_bar = v_bar[:v_pos] + "●" + v_bar[v_pos+1:]
        else:
            v_bar = ""
        
        # Gesture display
        gesture_str = ""
        if self._last_gesture and time.time() - self._gesture_display_time < 1.5:
            g = self._last_gesture
            if g.label != GestureLabel.NONE:
                emoji = "👈" if g.label == GestureLabel.LEFT else "👉"
                gesture_str = f"  {emoji} {g.label.value.upper()} ({g.confidence:.0%})"
        
        # Print
        status = f"\r[{bar}] {direction} A={features.a:.2f} D={features.d:+.2f}"
        if v_bar:
            status += f" V:[{v_bar}]"
        status += gesture_str
        
        print(status + " " * 10, end="", flush=True)
    
    def print_gesture(self, event: GestureEvent):
        """Print gesture detection."""
        if event.label == GestureLabel.NONE:
            return
        
        emoji = "👈" if event.label == GestureLabel.LEFT else "👉"
        print(f"\n✨ {emoji} {event.label.value.upper()} "
              f"(confidence: {event.confidence:.0%}, "
              f"duration: {event.duration:.2f}s)")


class MatplotlibUI:
    """
    Matplotlib-based visualization with spectrogram and tracking plots.
    
    Shows:
    - Live spectrogram around carrier frequency
    - D(t) and A(t) feature traces
    - Motion tracking visualization (velocity strip chart, position dot)
    """
    
    def __init__(self, config: Config, audio_source, dsp, tracker=None):
        self.config = config
        self.audio = audio_source
        self.dsp = dsp
        self.tracker = tracker
        
        # History for plotting
        self._d_history = deque(maxlen=150)
        self._a_history = deque(maxlen=150)
        
        # Gesture overlay
        self._gesture_text = ""
        self._gesture_time = 0.0
        
        self._fig = None
        self._running = False
    
    def start(self):
        """Start the visualization."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        plt.style.use('dark_background')
        
        # Create figure with subplots
        if self.tracker:
            self._fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            self._ax_spec = axes[0, 0]
            self._ax_features = axes[0, 1]
            self._ax_velocity = axes[1, 0]
            self._ax_position = axes[1, 1]
        else:
            self._fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            self._ax_spec = axes[0]
            self._ax_features = axes[1]
            self._ax_velocity = None
            self._ax_position = None
        
        self._fig.suptitle('AirSwipe - Ultrasonic Gesture Sensing', 
                          fontsize=14, fontweight='bold', color='#00ff88')
        
        # Animation
        self._anim = FuncAnimation(
            self._fig, self._update_plot,
            interval=self.config.ui_update_interval_ms,
            blit=False, cache_frame_data=False
        )
        
        self._running = True
        plt.tight_layout()
        plt.show()
    
    def _update_plot(self, frame):
        """Update all plots."""
        # Get audio and compute spectrogram
        audio = self.audio.get_buffer(self.config.spectrogram_history_sec)
        
        if len(audio) < self.config.fft_size:
            return
        
        times, freqs, Sxx = self.dsp.compute_spectrogram(audio)
        features = self.dsp.extract_features(audio)
        
        # Store feature history
        self._d_history.append(features.d)
        self._a_history.append(features.a)
        
        # Update tracker
        if self.tracker:
            tracking = self.tracker.update(features)
        
        # === Spectrogram plot ===
        self._ax_spec.clear()
        
        # Filter to carrier band
        cf = self.config.carrier_freq
        bw = self.config.doppler_bandwidth * 1.5
        freq_mask = (freqs >= cf - bw) & (freqs <= cf + bw)
        
        # Log magnitude
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        self._ax_spec.pcolormesh(
            times, freqs[freq_mask], Sxx_db[freq_mask, :],
            shading='gouraud', cmap='inferno',
            vmin=-60, vmax=-10
        )
        self._ax_spec.axhline(y=cf, color='cyan', linestyle='--', 
                              alpha=0.6, linewidth=1)
        self._ax_spec.set_ylabel('Frequency (Hz)', color='#aaa')
        self._ax_spec.set_xlabel('Time (s)', color='#aaa')
        self._ax_spec.set_title(f'Spectrogram (carrier: {cf:.0f} Hz)', 
                               color='#00ff88')
        
        # === Feature plot ===
        self._ax_features.clear()
        d_arr = np.array(self._d_history)
        a_arr = np.array(self._a_history)
        
        if len(d_arr) > 1:
            x = np.arange(len(d_arr))
            self._ax_features.plot(x, d_arr, color='#00ffff', 
                                   linewidth=2, label='D(t) Doppler')
            self._ax_features.plot(x, a_arr, color='#ff8800', 
                                   linewidth=1.5, alpha=0.7, label='A(t) Activity')
            self._ax_features.axhline(y=0, color='#444', linestyle='-')
            self._ax_features.axhline(y=self.config.activity_threshold, 
                                     color='#ff4444', linestyle='--', 
                                     alpha=0.5, label='Threshold')
        
        self._ax_features.set_ylabel('Feature Value', color='#aaa')
        self._ax_features.set_xlabel('Frame', color='#aaa')
        self._ax_features.set_title('Doppler Features', color='#00ff88')
        self._ax_features.legend(loc='upper right', fontsize=8)
        
        # Gesture overlay
        if self._gesture_text and time.time() - self._gesture_time < 2.0:
            self._ax_features.text(
                0.5, 0.9, self._gesture_text,
                transform=self._ax_features.transAxes,
                fontsize=24, ha='center', va='top',
                color='#00ff00', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#000', alpha=0.8)
            )
        
        # === Velocity plot (Part 2) ===
        if self._ax_velocity and self.tracker:
            self._ax_velocity.clear()
            v_history = self.tracker.v_history
            
            if len(v_history) > 1:
                x = np.arange(len(v_history))
                self._ax_velocity.fill_between(
                    x, 0, v_history, 
                    where=v_history >= 0, color='#00ff88', alpha=0.6
                )
                self._ax_velocity.fill_between(
                    x, 0, v_history,
                    where=v_history < 0, color='#ff4488', alpha=0.6
                )
                self._ax_velocity.plot(x, v_history, color='white', linewidth=1)
            
            self._ax_velocity.axhline(y=0, color='#444', linestyle='-')
            self._ax_velocity.set_ylim(-1.1, 1.1)
            self._ax_velocity.set_ylabel('Velocity', color='#aaa')
            self._ax_velocity.set_xlabel('Frame', color='#aaa')
            self._ax_velocity.set_title('v̂(t) Motion Velocity', color='#00ff88')
        
        # === Position plot (Part 2) ===
        if self._ax_position and self.tracker:
            self._ax_position.clear()
            
            # Position bar with dot
            x_pos = self.tracker.position
            
            # Draw bar
            self._ax_position.barh(0, 2, left=-1, height=0.3, 
                                   color='#333', edgecolor='#555')
            
            # Draw position dot
            dot_color = '#00ff88' if tracking.is_active else '#666'
            self._ax_position.scatter([x_pos], [0], s=500, c=dot_color, 
                                     zorder=10, edgecolor='white')
            
            # Labels
            self._ax_position.text(-1, -0.3, 'LEFT', ha='center', 
                                   color='#00ffff', fontsize=10)
            self._ax_position.text(1, -0.3, 'RIGHT', ha='center',
                                   color='#ff4488', fontsize=10)
            self._ax_position.text(0, 0.3, f'x̂ = {x_pos:+.2f}', ha='center',
                                   color='white', fontsize=12)
            
            self._ax_position.set_xlim(-1.3, 1.3)
            self._ax_position.set_ylim(-0.6, 0.6)
            self._ax_position.axis('off')
            self._ax_position.set_title('x̂(t) Position Tracking', color='#00ff88')
        
        self._fig.canvas.draw_idle()
    
    def show_gesture(self, event: GestureEvent):
        """Display gesture detection overlay."""
        if event.label == GestureLabel.NONE:
            return
        
        emoji = "👈" if event.label == GestureLabel.LEFT else "👉"
        self._gesture_text = f"{emoji} {event.label.value.upper()}"
        self._gesture_time = time.time()
    
    def stop(self):
        """Stop visualization."""
        self._running = False
        if self._fig:
            import matplotlib.pyplot as plt
            plt.close(self._fig)


def create_ui(config: Config, audio_source, dsp, tracker=None, 
              mode: str = "matplotlib"):
    """
    Factory function to create appropriate UI.
    
    Args:
        config: AirSwipe configuration
        audio_source: Audio source (AudioDuplex or AudioRx)
        dsp: DSP processor
        tracker: Optional motion tracker
        mode: "matplotlib" or "console"
        
    Returns:
        UI instance
    """
    if mode == "matplotlib":
        return MatplotlibUI(config, audio_source, dsp, tracker)
    else:
        return ConsoleUI(config)
