"""
AirSwipe - Ultrasonic Doppler Gesture Recognition for macOS

A gesture system using the laptop speaker to emit an ultrasonic carrier
and the microphone to record reflections. Uses STFT features to detect
Doppler motion for left/right swipe classification.
"""

__version__ = "0.1.0"
__author__ = "AirSwipe Project"

from .config import Config
