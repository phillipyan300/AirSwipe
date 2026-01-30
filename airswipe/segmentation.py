"""
Gesture segmentation module for AirSwipe.

Detects candidate gesture windows from continuous feature streams.
"""

import numpy as np
from collections import deque
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from .config import Config
from .dsp import DopplerFeatures


class GestureLabel(Enum):
    """Gesture classification labels."""
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class GestureEvent:
    """A detected gesture event."""
    label: GestureLabel
    confidence: float
    start_time: float
    end_time: float
    features: List[DopplerFeatures]
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def mean_d(self) -> float:
        """Mean signed Doppler proxy."""
        return np.mean([f.d for f in self.features])
    
    @property
    def max_a(self) -> float:
        """Maximum activity level."""
        return np.max([f.a for f in self.features])


class SegmentationState(Enum):
    """State machine states for event detection."""
    IDLE = "idle"
    ONSET = "onset"
    ACTIVE = "active"


class GestureSegmenter:
    """
    Segments continuous feature stream into gesture events.
    
    Uses a simple state machine with energy thresholding:
    - IDLE: Waiting for activity above threshold
    - ONSET: Activity detected, accumulating frames
    - ACTIVE: Gesture in progress, waiting for offset
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # State machine
        self._state = SegmentationState.IDLE
        self._event_start_time: float = 0.0
        self._event_features: List[DopplerFeatures] = []
        
        # Timing
        self._last_gesture_time: float = 0.0
        self._frame_time = config.time_resolution
        
        # Adaptive threshold
        self._activity_history = deque(maxlen=200)
        self._threshold = config.activity_threshold
        
        # Callbacks
        self._on_gesture = None
    
    def set_gesture_callback(self, callback):
        """
        Register callback for gesture detection.
        
        Args:
            callback: Function(GestureEvent) -> None
        """
        self._on_gesture = callback
    
    def update(self, features: DopplerFeatures, timestamp: Optional[float] = None) -> Optional[GestureEvent]:
        """
        Process a new feature frame.
        
        Args:
            features: Extracted Doppler features for this frame
            timestamp: Optional timestamp (uses time.time() if None)
            
        Returns:
            GestureEvent if a gesture was completed, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Track activity for adaptive thresholding
        self._activity_history.append(features.a)
        
        # Check cooldown
        if timestamp - self._last_gesture_time < self.config.cooldown_sec:
            return None
        
        event = None
        
        if self._state == SegmentationState.IDLE:
            # Check for onset (must be above threshold but below cap)
            # Very high activity is likely noise/crackle, not a gesture
            if features.a > self._threshold and features.a < self.config.activity_cap:
                self._state = SegmentationState.ONSET
                self._event_start_time = timestamp
                self._event_features = [features]
        
        elif self._state == SegmentationState.ONSET:
            # Abort if activity spikes too high (noise/crackle)
            if features.a >= self.config.activity_cap:
                self._state = SegmentationState.IDLE
                self._event_features = []
                return None
            
            self._event_features.append(features)
            
            # Check if we have minimum duration
            duration = timestamp - self._event_start_time
            if duration >= self.config.min_event_duration_sec:
                self._state = SegmentationState.ACTIVE
            
            # Check for premature offset (false trigger)
            if features.a < self._threshold * 0.5:
                self._state = SegmentationState.IDLE
                self._event_features = []
        
        elif self._state == SegmentationState.ACTIVE:
            # Abort if activity spikes too high (noise/crackle contaminated the gesture)
            if features.a >= self.config.activity_cap:
                self._state = SegmentationState.IDLE
                self._event_features = []
                return None
            
            self._event_features.append(features)
            duration = timestamp - self._event_start_time
            
            # Check for offset or max duration
            if features.a < self._threshold * 0.5 or duration > self.config.max_event_duration_sec:
                # Gesture complete - classify it
                event = self._classify_event(timestamp)
                
                if event.label != GestureLabel.NONE:
                    self._last_gesture_time = timestamp
                    if self._on_gesture:
                        self._on_gesture(event)
                
                # Reset state
                self._state = SegmentationState.IDLE
                self._event_features = []
        
        return event
    
    def _classify_event(self, end_time: float) -> GestureEvent:
        """
        Classify accumulated features into a gesture.
        
        Uses simple heuristics based on signed Doppler integral.
        """
        if len(self._event_features) < 3:
            return GestureEvent(
                label=GestureLabel.NONE,
                confidence=0.0,
                start_time=self._event_start_time,
                end_time=end_time,
                features=self._event_features
            )
        
        # Compute summary statistics
        d_values = np.array([f.d for f in self._event_features])
        a_values = np.array([f.a for f in self._event_features])
        
        # Signed area under D(t) curve
        signed_area = np.sum(d_values)
        
        # Peak activity
        peak_a = np.max(a_values)
        
        # Mean and variance of D
        mean_d = np.mean(d_values)
        
        # Classification based on signed area
        # Positive D = energy above carrier = hand approaching = leftward motion
        # Negative D = energy below carrier = hand receding = rightward motion
        # (This mapping depends on your coordinate system - adjust as needed)
        
        # Normalize by duration for confidence
        duration = end_time - self._event_start_time
        normalized_area = signed_area / (len(d_values) + 1)
        
        # Thresholds for classification
        direction_threshold = self._threshold * 2
        
        if normalized_area > direction_threshold:
            label = GestureLabel.LEFT
            confidence = min(1.0, normalized_area / (direction_threshold * 3))
        elif normalized_area < -direction_threshold:
            label = GestureLabel.RIGHT
            confidence = min(1.0, -normalized_area / (direction_threshold * 3))
        else:
            label = GestureLabel.NONE
            confidence = 0.0
        
        return GestureEvent(
            label=label,
            confidence=confidence,
            start_time=self._event_start_time,
            end_time=end_time,
            features=self._event_features.copy()
        )
    
    def update_threshold(self, factor: float = 1.0):
        """
        Update activity threshold based on recent history.
        
        Args:
            factor: Multiplier for computed threshold
        """
        if len(self._activity_history) > 50:
            # Use percentile of recent activity
            baseline = np.percentile(list(self._activity_history), 75)
            self._threshold = max(
                self.config.activity_threshold,
                baseline * 2 * factor
            )
    
    @property
    def state(self) -> SegmentationState:
        """Current segmentation state."""
        return self._state
    
    @property
    def threshold(self) -> float:
        """Current activity threshold."""
        return self._threshold
    
    def reset(self):
        """Reset segmenter state."""
        self._state = SegmentationState.IDLE
        self._event_features = []
        self._activity_history.clear()


class TemporalSmoother:
    """
    Smooths gesture predictions over time.
    
    Uses majority voting over a sliding window to reduce
    spurious classifications.
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._predictions = deque(maxlen=window_size)
    
    def update(self, label: GestureLabel, confidence: float) -> Tuple[GestureLabel, float]:
        """
        Add prediction and return smoothed result.
        
        Args:
            label: Raw predicted label
            confidence: Raw confidence
            
        Returns:
            (smoothed_label, smoothed_confidence)
        """
        self._predictions.append((label, confidence))
        
        if len(self._predictions) < self.window_size:
            return label, confidence
        
        # Count labels
        label_counts = {}
        label_confidences = {}
        
        for lbl, conf in self._predictions:
            if lbl not in label_counts:
                label_counts[lbl] = 0
                label_confidences[lbl] = []
            label_counts[lbl] += 1
            label_confidences[lbl].append(conf)
        
        # Majority vote
        majority_label = max(label_counts, key=label_counts.get)
        majority_conf = np.mean(label_confidences[majority_label])
        
        return majority_label, majority_conf
    
    def reset(self):
        """Clear prediction history."""
        self._predictions.clear()
