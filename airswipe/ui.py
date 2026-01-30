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
        self._d_history = deque(maxlen=20)  # Track recent D values
        self._frame_count = 0
    
    def update(self, features: DopplerFeatures, 
               tracking: Optional[TrackingState] = None,
               gesture: Optional[GestureEvent] = None):
        """Update console display."""
        
        self._frame_count += 1
        self._d_history.append(features.d)
        
        if gesture is not None:
            self._last_gesture = gesture
            self._gesture_display_time = time.time()
        
        # Build activity bar
        activity = min(features.a / (self.config.activity_threshold * 5), 1.0)
        bar_len = int(activity * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        
        # D history sparkline (shows trend)
        if len(self._d_history) >= 5:
            recent_d = list(self._d_history)[-10:]  # Last 10 values
            d_trend = ""
            for d in recent_d:
                if d > 5:
                    d_trend += "▲"  # Strong positive
                elif d > 0.5:
                    d_trend += "△"  # Weak positive
                elif d < -5:
                    d_trend += "▼"  # Strong negative
                elif d < -0.5:
                    d_trend += "▽"  # Weak negative
                else:
                    d_trend += "·"  # Near zero
        else:
            d_trend = "." * 10
        
        # Direction indicator based on recent trend
        recent_sum = sum(list(self._d_history)[-5:]) if len(self._d_history) >= 5 else 0
        if recent_sum > 2:
            direction = "◀◀"
        elif recent_sum > 0.5:
            direction = "◀ "
        elif recent_sum < -2:
            direction = "▶▶"
        elif recent_sum < -0.5:
            direction = " ▶"
        else:
            direction = "○ "
        
        # Gesture display
        gesture_str = ""
        if self._last_gesture and time.time() - self._gesture_display_time < 2.0:
            g = self._last_gesture
            emoji = "👈" if g.label == GestureLabel.LEFT else ("👉" if g.label == GestureLabel.RIGHT else "·")
            gesture_str = f" → {emoji} {g.label.value.upper()} ({g.confidence:.0%})"
        
        # Print with D history
        status = f"\r[{bar}] {direction} D:[{d_trend}] A={features.a:6.1f}"
        status += gesture_str
        
        print(status + " " * 10, end="", flush=True)
    
    def print_gesture(self, event: GestureEvent):
        """Print gesture detection with details."""
        emoji = "👈" if event.label == GestureLabel.LEFT else ("👉" if event.label == GestureLabel.RIGHT else "○")
        
        # Calculate summary stats from the event
        if event.features:
            d_vals = [f.d for f in event.features]
            mean_d = np.mean(d_vals)
            signed_area = sum(d_vals)
            info = f"mean_d={mean_d:+.1f}, area={signed_area:+.0f}"
        else:
            info = ""
        
        print(f"\n✨ {emoji} {event.label.value.upper()} "
              f"(conf={event.confidence:.0%}, dur={event.duration:.2f}s, {info})")


class MatplotlibUI:
    """
    Matplotlib-based visualization with spectrogram and tracking plots.
    
    Shows:
    - Live spectrogram around carrier frequency
    - D(t) and A(t) feature traces
    - Motion tracking visualization (velocity strip chart, position dot)
    - Optional: Gesture detection with ML predictions
    """
    
    def __init__(self, config: Config, audio_source, dsp, tracker=None,
                 segmenter=None, classifier=None):
        self.config = config
        self.audio = audio_source
        self.dsp = dsp
        self.tracker = tracker
        self.segmenter = segmenter
        self.classifier = classifier
        
        # History for plotting
        self._d_history = deque(maxlen=150)
        self._a_history = deque(maxlen=150)
        
        # Gesture detection history
        self._prediction_history = []  # List of (time, label, confidence)
        self._current_prediction = None  # (label, confidence, time_remaining)
        self._prediction_display_duration = 1.0  # seconds to show prediction (shorter for responsiveness)
        
        # Gesture overlay
        self._gesture_text = ""
        self._gesture_time = 0.0
        
        self._fig = None
        self._running = False
        self._ax_prediction = None
    
    def start(self):
        """Start the visualization."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.gridspec import GridSpec
        
        plt.style.use('dark_background')
        
        # Create figure with subplots
        # Layout depends on what features are enabled
        has_detection = self.segmenter is not None and self.classifier is not None
        
        if has_detection:
            # 3 rows: spectrogram, features, prediction panel
            self._fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)
            self._ax_spec = self._fig.add_subplot(gs[0])
            self._ax_features = self._fig.add_subplot(gs[1])
            self._ax_prediction = self._fig.add_subplot(gs[2])
            self._ax_velocity = None
            self._ax_position = None
        elif self.tracker:
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
        try:
            plt.tight_layout()
        except Exception:
            pass  # GridSpec layouts may not be compatible with tight_layout
        plt.show()
    
    def _update_plot(self, frame):
        """Update all plots."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        # Get audio and compute spectrogram
        audio = self.audio.get_buffer(self.config.spectrogram_history_sec)
        
        if len(audio) < self.config.fft_size:
            return
        
        times, freqs, Sxx = self.dsp.compute_spectrogram(audio)
        # Use use_baseline=False to match training data
        features = self.dsp.extract_features(audio, use_baseline=False)
        
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
        bw = self.config.doppler_bandwidth * 2  # ±1000 Hz around carrier
        freq_min = cf - bw
        freq_max = cf + bw
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
        
        # Log magnitude
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        if np.any(freq_mask):
            freqs_filtered = freqs[freq_mask]
            Sxx_filtered = Sxx_db[freq_mask, :]
            
            # Use imshow for more reliable display
            extent = [times.min(), times.max(), freqs_filtered.min(), freqs_filtered.max()]
            self._ax_spec.imshow(
                Sxx_filtered, 
                aspect='auto', 
                origin='lower',
                extent=extent,
                cmap='inferno',
                vmin=-60, vmax=-10
            )
            self._ax_spec.axhline(y=cf, color='cyan', linestyle='--', 
                                  alpha=0.8, linewidth=2, label=f'Carrier {cf:.0f}Hz')
        else:
            # Fallback if no data in range
            self._ax_spec.text(0.5, 0.5, 'No data in carrier band', 
                              transform=self._ax_spec.transAxes, ha='center')
        
        self._ax_spec.set_ylabel('Frequency (Hz)', color='#aaa')
        self._ax_spec.set_xlabel('Time (s)', color='#aaa')
        self._ax_spec.set_title(f'Spectrogram ({freq_min:.0f}-{freq_max:.0f} Hz, carrier: {cf:.0f})', 
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
            # Scale threshold to match data range for visibility
            vis_threshold = max(self.config.activity_threshold, np.percentile(a_arr, 80) * 0.5) if len(a_arr) > 10 else self.config.activity_threshold
            self._ax_features.axhline(y=vis_threshold, 
                                     color='#ff4444', linestyle='--', 
                                     alpha=0.5, label='Threshold')
            self._ax_features.legend(loc='upper right', fontsize=8)
        
        self._ax_features.set_ylabel('Feature Value', color='#aaa')
        self._ax_features.set_xlabel('Frame', color='#aaa')
        self._ax_features.set_title('Doppler Features', color='#00ff88')
        
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
        
        # === Prediction panel (when detection enabled) ===
        if self._ax_prediction is not None and self.segmenter and self.classifier:
            self._ax_prediction.clear()
            
            # Process gesture detection
            gesture_event = self.segmenter.update(features)
            
            if gesture_event and len(gesture_event.features) >= 3:
                # Run classifier
                d_values = np.array([f.d for f in gesture_event.features])
                a_values = np.array([f.a for f in gesture_event.features])
                
                # Check D variance - reject flat/noisy segments
                d_variance = np.var(d_values)
                
                if d_variance < self.config.min_d_variance:
                    # Not enough D variation - probably not a real gesture
                    print(f"[REJECTED] D variance too low: {d_variance:.1f} < {self.config.min_d_variance} "
                          f"(range: [{d_values.min():.1f}, {d_values.max():.1f}])")
                else:
                    # Check classifier type and predict accordingly
                    classifier_name = self.classifier.__class__.__name__
                    
                    if 'CNN1D' in classifier_name:
                        # CNN expects raw D(t) series
                        pred = self.classifier.predict(d_values)
                        feat_vec = None  # CNN doesn't use hand-crafted features
                        
                        # Debug: print D values and probabilities
                        print(f"[CNN1D] D var:{d_variance:.0f} range:[{d_values.min():.1f}, {d_values.max():.1f}], "
                              f"probs: {pred.probabilities}, pred: {pred.label.value} ({pred.confidence:.1%})")
                    elif hasattr(self.classifier, 'extract_features'):
                        # Logistic/Baseline expects feature vector
                        feat_vec = self.classifier.extract_features(d_values, a_values, gesture_event.duration)
                        feat_vec_norm = self.classifier.normalize_features(feat_vec)
                        pred = self.classifier.predict(feat_vec_norm)
                    else:
                        # Fallback - try direct prediction
                        pred = self.classifier.predict(d_values)
                        feat_vec = None
                    
                    # Apply confidence gating - reject low-confidence predictions
                    final_label = pred.label
                    if pred.confidence < self.config.confidence_threshold:
                        # Import GestureLabel for NONE
                        from airswipe.segmentation import GestureLabel
                        final_label = GestureLabel.NONE
                    
                    # Store prediction WITH features for display
                    self._current_prediction = (final_label, pred.confidence, time.time(), feat_vec)
                    self._prediction_history.append((time.time(), final_label, pred.confidence))
                
                # Keep only last 10 predictions
                if len(self._prediction_history) > 10:
                    self._prediction_history = self._prediction_history[-10:]
            
            # Draw prediction display
            self._ax_prediction.set_xlim(0, 10)
            self._ax_prediction.set_ylim(0, 1)
            self._ax_prediction.axis('off')
            
            # Check current activity level
            current_activity = features.a
            is_active = current_activity > self.config.activity_threshold
            
            # Show prediction only when:
            # 1. We have a recent prediction (within display duration)
            # 2. AND either activity is high OR the prediction is very recent (< 0.5s)
            if self._current_prediction:
                # Unpack - handle both old (3-tuple) and new (4-tuple) formats
                if len(self._current_prediction) == 4:
                    label, conf, pred_time, feat_vec = self._current_prediction
                else:
                    label, conf, pred_time = self._current_prediction
                    feat_vec = None
                    
                age = time.time() - pred_time
                
                if age < self._prediction_display_duration:
                    # Show big prediction
                    # Handle both GestureLabel enum and string
                    label_str = label.value if hasattr(label, 'value') else str(label)
                    if label_str == 'left':
                        emoji, color = '<--', '#00ffff'
                        text = 'LEFT'
                    elif label_str == 'right':
                        emoji, color = '-->', '#ff4488'
                        text = 'RIGHT'
                    else:
                        emoji, color = '---', '#888888'
                        text = 'NONE'
                    
                    # Fade out effect
                    alpha = max(0.3, 1.0 - (age / self._prediction_display_duration) * 0.7)
                    
                    self._ax_prediction.text(
                        5, 0.75, f'{emoji} {text}',
                        ha='center', va='center',
                        fontsize=40, fontweight='bold', color=color, alpha=alpha
                    )
                    self._ax_prediction.text(
                        5, 0.55, f'Confidence: {conf:.0%}',
                        ha='center', va='center',
                        fontsize=14, color='#aaa', alpha=alpha
                    )
                    
                    # Show key features
                    if feat_vec is not None:
                        # Features: mean_d[0], max_d[1], min_d[2], slope_d[3], signed_area[4], 
                        #           duration[5], max_a[6], std_d[7], peak_order[8], peak_asymmetry[9]
                        slope = feat_vec[3]
                        max_a = feat_vec[6]
                        peak_order = feat_vec[8]
                        peak_asym = feat_vec[9]
                        max_d = feat_vec[1]
                        min_d = feat_vec[2]
                        
                        feature_text = (
                            f"slope={slope:+.1f}  peak_asym={peak_asym:+.3f}  "
                            f"peak_order={peak_order:+.2f}  max_A={max_a:.0f}"
                        )
                        self._ax_prediction.text(
                            5, 0.35, feature_text,
                            ha='center', va='center',
                            fontsize=11, color='#888', alpha=alpha,
                            family='monospace'
                        )
                        
                        # Show max_d and min_d on second line
                        feature_text2 = f"max_d={max_d:+.0f}  min_d={min_d:+.0f}"
                        self._ax_prediction.text(
                            5, 0.22, feature_text2,
                            ha='center', va='center',
                            fontsize=11, color='#666', alpha=alpha,
                            family='monospace'
                        )
                else:
                    # Waiting for gesture - show current activity and state
                    if current_activity >= self.config.activity_cap:
                        # Noise spike detected
                        self._ax_prediction.text(
                            5, 0.6, 'NOISE SPIKE (ignored)',
                            ha='center', va='center',
                            fontsize=18, color='#ff6666'
                        )
                    else:
                        self._ax_prediction.text(
                            5, 0.6, 'Waiting for gesture...',
                            ha='center', va='center',
                            fontsize=20, color='#555'
                        )
                    self._ax_prediction.text(
                        5, 0.35, f'Activity: {current_activity:.0f} (range: {self.config.activity_threshold:.0f}-{self.config.activity_cap:.0f})',
                        ha='center', va='center',
                        fontsize=12, color='#444'
                    )
            else:
                # No prediction yet - show activity
                if current_activity >= self.config.activity_cap:
                    self._ax_prediction.text(
                        5, 0.6, 'NOISE SPIKE (ignored)',
                        ha='center', va='center',
                        fontsize=18, color='#ff6666'
                    )
                else:
                    self._ax_prediction.text(
                        5, 0.6, 'Waiting for gesture...',
                        ha='center', va='center',
                        fontsize=20, color='#555'
                    )
                self._ax_prediction.text(
                    5, 0.35, f'Activity: {current_activity:.0f} (range: {self.config.activity_threshold:.0f}-{self.config.activity_cap:.0f})',
                    ha='center', va='center',
                    fontsize=12, color='#444'
                )
            
            # Draw recent history bar at bottom
            if self._prediction_history:
                bar_width = 0.8
                for i, (_, label, conf) in enumerate(self._prediction_history[-10:]):
                    x = i * 1.0 + 0.5
                    label_str = label.value if hasattr(label, 'value') else str(label)
                    if label_str == 'left':
                        color = '#00ffff'
                    elif label_str == 'right':
                        color = '#ff4488'
                    else:
                        color = '#444444'
                    self._ax_prediction.add_patch(
                        mpatches.Rectangle((x - bar_width/2, 0), bar_width, 0.1,
                                           facecolor=color, alpha=0.7)
                    )
            
            self._ax_prediction.set_title('Gesture Prediction', color='#00ff88', fontsize=14)
        
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
