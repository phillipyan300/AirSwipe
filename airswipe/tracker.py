"""
Motion tracking module for AirSwipe (Part 2).

Produces continuous velocity and position signals from Doppler features
for real-time motion visualization.
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional
from dataclasses import dataclass

from .config import Config
from .dsp import DopplerFeatures


@dataclass
class TrackingState:
    """Current tracking state."""
    v_hat: float        # Smoothed velocity proxy
    x_hat: float        # Integrated position proxy
    v_raw: float        # Raw velocity before smoothing
    is_active: bool     # Whether motion is detected


class MotionTracker:
    """
    Converts Doppler features into continuous motion signals.
    
    Produces:
    - v_hat(t): Smoothed velocity proxy (positive = leftward, negative = rightward)
    - x_hat(t): Integrated position proxy (bounded, auto-centering)
    
    Uses EMA smoothing and drift control during idle.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Current state
        self._v_hat: float = 0.0
        self._x_hat: float = 0.0
        self._v_raw: float = 0.0
        
        # History for visualization
        self._v_history = deque(maxlen=200)
        self._x_history = deque(maxlen=200)
        
        # Smoothing parameters
        self._alpha = config.velocity_ema_alpha
        self._decay = config.position_decay
        self._scale = config.velocity_scale
        
        # Activity detection
        self._activity_threshold = config.activity_threshold
        self._is_active = False
    
    def update(self, features: DopplerFeatures) -> TrackingState:
        """
        Process new features and update tracking state.
        
        Args:
            features: Doppler features from DSP
            
        Returns:
            Current TrackingState
        """
        # Normalize D(t) to velocity proxy
        v_raw = np.clip(features.d / self._scale, -1.0, 1.0)
        self._v_raw = v_raw
        
        # Detect activity
        self._is_active = features.a > self._activity_threshold
        
        if self._is_active:
            # Active: EMA smoothing
            self._v_hat = self._alpha * self._v_hat + (1 - self._alpha) * v_raw
            
            # Integrate position
            self._x_hat += self._v_hat * 0.1  # Scale factor for position
        else:
            # Idle: decay toward neutral
            self._v_hat *= self._decay
            self._x_hat *= self._decay
        
        # Clamp position to [-1, 1]
        self._x_hat = np.clip(self._x_hat, -1.0, 1.0)
        
        # Store history
        self._v_history.append(self._v_hat)
        self._x_history.append(self._x_hat)
        
        return TrackingState(
            v_hat=self._v_hat,
            x_hat=self._x_hat,
            v_raw=v_raw,
            is_active=self._is_active
        )
    
    @property
    def velocity(self) -> float:
        """Current smoothed velocity."""
        return self._v_hat
    
    @property
    def position(self) -> float:
        """Current position."""
        return self._x_hat
    
    @property
    def v_history(self) -> np.ndarray:
        """Recent velocity history."""
        return np.array(self._v_history)
    
    @property
    def x_history(self) -> np.ndarray:
        """Recent position history."""
        return np.array(self._x_history)
    
    def reset(self):
        """Reset tracker to neutral state."""
        self._v_hat = 0.0
        self._x_hat = 0.0
        self._v_history.clear()
        self._x_history.clear()
    
    def set_parameters(self, alpha: Optional[float] = None,
                       decay: Optional[float] = None,
                       scale: Optional[float] = None):
        """Update tracking parameters."""
        if alpha is not None:
            self._alpha = np.clip(alpha, 0.0, 0.99)
        if decay is not None:
            self._decay = np.clip(decay, 0.0, 0.999)
        if scale is not None:
            self._scale = max(0.1, scale)


class KalmanTracker:
    """
    Kalman filter-based motion tracker for smoother tracking.
    
    State: [position, velocity]
    Uses constant velocity model with process noise.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # State: [x, v]
        self._state = np.zeros(2)
        
        # State covariance
        self._P = np.eye(2) * 0.1
        
        # Process noise covariance
        dt = config.time_resolution
        q = 0.1  # Process noise intensity
        self._Q = np.array([
            [dt**4/4, dt**3/2],
            [dt**3/2, dt**2]
        ]) * q
        
        # Measurement noise
        self._R = np.array([[0.5]])  # Velocity measurement noise
        
        # State transition matrix (constant velocity)
        self._F = np.array([
            [1, dt],
            [0, 1]
        ])
        
        # Measurement matrix (we observe velocity)
        self._H = np.array([[0, 1]])
        
        # History
        self._v_history = deque(maxlen=200)
        self._x_history = deque(maxlen=200)
        
        self._is_active = False
        self._scale = config.velocity_scale
        self._activity_threshold = config.activity_threshold
    
    def update(self, features: DopplerFeatures) -> TrackingState:
        """
        Kalman filter update with new measurement.
        """
        # Normalize measurement
        v_measured = np.clip(features.d / self._scale, -1.0, 1.0)
        
        self._is_active = features.a > self._activity_threshold
        
        # Predict step
        self._state = self._F @ self._state
        self._P = self._F @ self._P @ self._F.T + self._Q
        
        if self._is_active:
            # Update step (only when active)
            z = np.array([v_measured])
            y = z - self._H @ self._state  # Innovation
            S = self._H @ self._P @ self._H.T + self._R  # Innovation covariance
            K = self._P @ self._H.T @ np.linalg.inv(S)  # Kalman gain
            
            self._state = self._state + K @ y
            self._P = (np.eye(2) - K @ self._H) @ self._P
        else:
            # Drift toward zero when idle
            self._state *= 0.98
        
        # Clamp position
        self._state[0] = np.clip(self._state[0], -1.0, 1.0)
        
        # Store history
        self._v_history.append(self._state[1])
        self._x_history.append(self._state[0])
        
        return TrackingState(
            v_hat=self._state[1],
            x_hat=self._state[0],
            v_raw=v_measured,
            is_active=self._is_active
        )
    
    @property
    def velocity(self) -> float:
        return self._state[1]
    
    @property
    def position(self) -> float:
        return self._state[0]
    
    @property
    def v_history(self) -> np.ndarray:
        return np.array(self._v_history)
    
    @property
    def x_history(self) -> np.ndarray:
        return np.array(self._x_history)
    
    def reset(self):
        self._state = np.zeros(2)
        self._P = np.eye(2) * 0.1
        self._v_history.clear()
        self._x_history.clear()
