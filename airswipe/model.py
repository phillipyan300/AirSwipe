"""
Classification models for AirSwipe.

Includes baseline logistic regression and tiny CNN classifier.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import pickle
from pathlib import Path

from .config import Config
from .segmentation import GestureLabel


@dataclass
class Prediction:
    """Model prediction result."""
    label: GestureLabel
    confidence: float
    probabilities: np.ndarray  # [none, left, right]


class BaselineClassifier:
    """
    Simple logistic regression classifier.
    
    Uses hand-crafted features from Doppler statistics:
    - Mean, max, min of D(t)
    - Slope of D(t) (linear fit)
    - Signed area under D(t)
    - Duration above threshold
    - Max activity A(t)
    """
    
    FEATURE_NAMES = [
        'mean_d', 'max_d', 'min_d', 'slope_d', 
        'signed_area', 'duration', 'max_a', 'std_d', 
        'peak_order', 'peak_asymmetry'
    ]
    
    def __init__(self, config: Config):
        self.config = config
        self._weights: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self._fitted = False
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        # Labels for prediction (set during fit, or default to 3-class)
        self._labels = [GestureLabel.NONE, GestureLabel.LEFT, GestureLabel.RIGHT]
    
    def extract_features(self, d_values: np.ndarray, 
                        a_values: np.ndarray,
                        duration: float,
                        center: bool = True) -> np.ndarray:
        """
        Extract summary features from D(t) and A(t) time series.
        
        Args:
            d_values: Signed Doppler proxy over time
            a_values: Activity values over time
            duration: Event duration in seconds
            center: If True, center D values (subtract mean) to remove offset
            
        Returns:
            Feature vector
        """
        if len(d_values) < 2:
            return np.zeros(len(self.FEATURE_NAMES))
        
        # Per-sample centering: remove offset so we focus on the PATTERN
        # This makes features invariant to baseline drift
        if center:
            d_mean = np.mean(d_values)
            d_centered = d_values - d_mean
        else:
            d_centered = d_values
            d_mean = np.mean(d_values)
        
        # Basic statistics (on centered values)
        mean_d = np.mean(d_centered)  # Will be ~0 if centered
        max_d = np.max(d_centered)
        min_d = np.min(d_centered)
        std_d = np.std(d_centered)
        
        # Slope via linear regression (on centered values)
        # This captures: did D go from high to low (negative slope = left)
        #                or low to high (positive slope = right)
        x = np.arange(len(d_centered))
        slope_d = np.polyfit(x, d_centered, 1)[0] if len(d_centered) > 2 else 0
        
        # Signed area of centered values
        # After centering, this captures asymmetry: 
        # - If D started high and ended low, area will be positive in first half, negative in second
        signed_area = np.sum(d_centered)
        
        # Activity (unchanged - this is about magnitude, not direction)
        max_a = np.max(a_values)
        
        # Peak order: did the max come before or after the min?
        # This directly encodes the temporal pattern:
        #   LEFT:  D goes high→low, so max comes BEFORE min → negative peak_order
        #   RIGHT: D goes low→high, so min comes BEFORE max → positive peak_order
        #   NONE:  random, near zero
        t_max = np.argmax(d_centered)
        t_min = np.argmin(d_centered)
        peak_order = (t_max - t_min) / len(d_centered)  # Normalize by length
        
        # Peak asymmetry: which peak is bigger in magnitude?
        # This is a NONLINEAR comparison that linear classifier can't learn from raw max/min
        #   LEFT:  |min_d| > max_d → negative asymmetry
        #   RIGHT: max_d > |min_d| → positive asymmetry
        #   NONE:  balanced → near zero
        abs_min = abs(min_d)
        denom = max_d + abs_min + 1e-8  # Avoid division by zero
        peak_asymmetry = (max_d - abs_min) / denom
        
        features = np.array([
            mean_d, max_d, min_d, slope_d,
            signed_area, duration, max_a, std_d, 
            peak_order, peak_asymmetry
        ])
        
        return features
    
    def predict(self, features: np.ndarray) -> Prediction:
        """
        Predict gesture from features.
        
        Args:
            features: Feature vector
            
        Returns:
            Prediction with label, confidence, probabilities
        """
        if not self._fitted:
            # Use simple heuristic if not trained
            return self._heuristic_predict(features)
        
        # Logistic regression
        logits = features @ self._weights + self._bias
        probs = self._softmax(logits)
        
        label_idx = np.argmax(probs)
        
        return Prediction(
            label=self._labels[label_idx],
            confidence=probs[label_idx],
            probabilities=probs
        )
    
    def _heuristic_predict(self, features: np.ndarray) -> Prediction:
        """
        Simple rule-based prediction (used before training).
        """
        # Extract key features
        mean_d = features[0]
        signed_area = features[4]
        max_a = features[6]
        
        # Check if there's enough activity
        if max_a < self.config.activity_threshold:
            return Prediction(
                label=GestureLabel.NONE,
                confidence=0.9,
                probabilities=np.array([0.9, 0.05, 0.05])
            )
        
        # Classify based on signed area
        threshold = self.config.activity_threshold * 5
        
        if signed_area > threshold:
            conf = min(0.95, 0.5 + signed_area / (threshold * 4))
            return Prediction(
                label=GestureLabel.LEFT,
                confidence=conf,
                probabilities=np.array([0.1, conf, 1 - conf - 0.1])
            )
        elif signed_area < -threshold:
            conf = min(0.95, 0.5 - signed_area / (threshold * 4))
            return Prediction(
                label=GestureLabel.RIGHT,
                confidence=conf,
                probabilities=np.array([0.1, 1 - conf - 0.1, conf])
            )
        else:
            return Prediction(
                label=GestureLabel.NONE,
                confidence=0.6,
                probabilities=np.array([0.6, 0.2, 0.2])
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            learning_rate: float = 0.01, 
            epochs: int = 1000,
            labels: Optional[List[GestureLabel]] = None):
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - indices into labels list
            learning_rate: SGD learning rate
            epochs: Training epochs
            labels: List of GestureLabel in order of class indices
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Set labels for prediction
        if labels is not None:
            self._labels = labels
        elif n_classes == 2:
            self._labels = [GestureLabel.LEFT, GestureLabel.RIGHT]
        else:
            self._labels = [GestureLabel.NONE, GestureLabel.LEFT, GestureLabel.RIGHT]
        
        # Initialize weights
        self._weights = np.random.randn(n_features, n_classes) * 0.01
        self._bias = np.zeros(n_classes)
        
        # One-hot encode labels
        y_onehot = np.eye(n_classes)[y]
        
        # SGD training with early stopping
        best_loss = float('inf')
        patience = 50  # Stop if no improvement for 50 epochs
        patience_counter = 0
        
        for epoch in range(epochs):
            # Forward pass
            logits = X @ self._weights + self._bias
            probs = self._softmax(logits)
            
            # Cross-entropy loss
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-10), axis=1))
            
            # Backward pass
            grad = (probs - y_onehot) / n_samples
            grad_w = X.T @ grad
            grad_b = np.sum(grad, axis=0)
            
            # Update
            self._weights -= learning_rate * grad_w
            self._bias -= learning_rate * grad_b
            
            # Early stopping check
            if loss < best_loss - 1e-4:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 100 == 0:
                accuracy = np.mean(np.argmax(probs, axis=1) == y)
                print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.3f}")
            
            # Stop early if converged
            if patience_counter >= patience and accuracy == 1.0:
                accuracy = np.mean(np.argmax(probs, axis=1) == y)
                print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.3f} (early stop)")
                break
        
        self._fitted = True
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def save(self, path: str):
        """Save model to file."""
        # Convert GestureLabel to strings for pickling
        label_strings = [l.value for l in self._labels]
        data = {
            'weights': self._weights,
            'bias': self._bias,
            'fitted': self._fitted,
            'feature_mean': getattr(self, '_feature_mean', None),
            'feature_std': getattr(self, '_feature_std', None),
            'labels': label_strings
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._weights = data['weights']
        self._bias = data['bias']
        self._fitted = data['fitted']
        self._feature_mean = data.get('feature_mean')
        self._feature_std = data.get('feature_std')
        # Load labels (with backwards compatibility)
        if 'labels' in data:
            self._labels = [GestureLabel(l) for l in data['labels']]
        else:
            self._labels = [GestureLabel.NONE, GestureLabel.LEFT, GestureLabel.RIGHT]
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using stored mean/std."""
        if self._feature_mean is not None and self._feature_std is not None:
            return (features - self._feature_mean) / self._feature_std
        return features


# ============================================================================
# CNN Classifier (requires PyTorch)
# ============================================================================

def create_cnn_classifier(config: Config):
    """
    Create tiny CNN classifier for spectrogram patches.
    
    Requires PyTorch. Returns None if not available.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("[MODEL] PyTorch not available, CNN disabled")
        return None
    
    class TinyCNN(nn.Module):
        """
        Small CNN for gesture classification from spectrogram patches.
        
        Input: (batch, 1, T, F) - time x frequency spectrogram patch
        Output: (batch, 3) - logits for [none, left, right]
        """
        
        def __init__(self, time_frames: int, freq_bins: int):
            super().__init__()
            
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.pool1 = nn.MaxPool2d(2)
            
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool2 = nn.MaxPool2d(2)
            
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 3)
            
            self.dropout = nn.Dropout(0.3)
        
        def forward(self, x):
            # x: (batch, 1, T, F)
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            
            x = F.relu(self.bn3(self.conv3(x)))
            
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            
            x = self.dropout(x)
            x = self.fc(x)
            
            return x
        
        def predict(self, x) -> Prediction:
            """Single sample prediction."""
            self.eval()
            with torch.no_grad():
                if not isinstance(x, torch.Tensor):
                    x = torch.FloatTensor(x)
                if x.dim() == 2:
                    x = x.unsqueeze(0).unsqueeze(0)
                elif x.dim() == 3:
                    x = x.unsqueeze(0)
                
                logits = self(x)
                probs = F.softmax(logits, dim=-1).numpy()[0]
                
                label_idx = np.argmax(probs)
                labels = [GestureLabel.NONE, GestureLabel.LEFT, GestureLabel.RIGHT]
                
                return Prediction(
                    label=labels[label_idx],
                    confidence=probs[label_idx],
                    probabilities=probs
                )
    
    return TinyCNN(config.patch_time_frames, config.patch_freq_bins)
