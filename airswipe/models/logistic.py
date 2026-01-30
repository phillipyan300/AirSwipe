"""
Logistic Regression classifier for hand-crafted features.
"""

import numpy as np
from typing import List, Optional

from .base import BaseModel, Prediction
from ..segmentation import GestureLabel


class LogisticClassifier(BaseModel):
    """
    Logistic regression on hand-crafted features.
    
    Input: Feature vector (10 features from D(t) and A(t))
    Output: Class probabilities
    
    Features extracted:
    - mean_d, max_d, min_d, slope_d, signed_area
    - duration, max_a, std_d, peak_order, peak_asymmetry
    """
    
    FEATURE_NAMES = [
        'mean_d', 'max_d', 'min_d', 'slope_d', 
        'signed_area', 'duration', 'max_a', 'std_d', 
        'peak_order', 'peak_asymmetry'
    ]
    
    def __init__(self, config):
        super().__init__(config)
        self._weights: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
    
    def extract_features(self, d_values: np.ndarray, 
                        a_values: np.ndarray,
                        duration: float,
                        center: bool = True) -> np.ndarray:
        """
        Extract summary features from D(t) and A(t) time series.
        
        Args:
            d_values: Signed Doppler proxy over time
            a_values: Activity level over time
            duration: Duration in seconds
            center: Whether to center D values (subtract mean)
            
        Returns:
            Feature vector (10,)
        """
        if len(d_values) < 3:
            return np.zeros(len(self.FEATURE_NAMES))
        
        # Per-sample centering
        if center:
            d_mean = np.mean(d_values)
            d_centered = d_values - d_mean
        else:
            d_centered = d_values
            d_mean = np.mean(d_values)
        
        # Basic statistics (on centered values)
        mean_d = np.mean(d_centered)
        max_d = np.max(d_centered)
        min_d = np.min(d_centered)
        std_d = np.std(d_centered)
        
        # Slope via linear regression
        x = np.arange(len(d_centered))
        slope_d = np.polyfit(x, d_centered, 1)[0] if len(d_centered) > 2 else 0
        
        # Signed area
        signed_area = np.sum(d_centered)
        
        # Activity
        max_a = np.max(a_values)
        
        # Peak order: did the max come before or after the min?
        t_max = np.argmax(d_centered)
        t_min = np.argmin(d_centered)
        peak_order = (t_max - t_min) / len(d_centered)
        
        # Peak asymmetry: which peak is bigger in magnitude?
        abs_min = abs(min_d)
        denom = max_d + abs_min + 1e-8
        peak_asymmetry = (max_d - abs_min) / denom
        
        features = np.array([
            mean_d, max_d, min_d, slope_d,
            signed_area, duration, max_a, std_d, 
            peak_order, peak_asymmetry
        ])
        
        return features
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            learning_rate: float = 0.1, 
            epochs: int = 500,
            verbose: bool = True):
        """
        Train logistic regression with SGD.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            learning_rate: SGD learning rate
            epochs: Maximum training epochs
            verbose: Print progress
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Set labels based on number of classes
        if n_classes == 2:
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
        patience = 50
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
            accuracy = np.mean(np.argmax(probs, axis=1) == y)
            
            if loss < best_loss - 1e-4:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.3f}")
            
            if patience_counter >= patience and accuracy == 1.0:
                if verbose:
                    print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.3f} (early stop)")
                break
        
        self._fitted = True
    
    def predict(self, features: np.ndarray) -> Prediction:
        """Predict class for a single feature vector."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Normalize if parameters are set
        features_norm = self.normalize(features)
        
        # Logistic regression
        logits = features_norm @ self._weights + self._bias
        probs = self._softmax(logits)
        
        label_idx = np.argmax(probs)
        
        return Prediction(
            label=self._labels[label_idx],
            confidence=probs[label_idx],
            probabilities=probs
        )
    
    def predict_batch(self, X: np.ndarray) -> List[Prediction]:
        """Predict classes for multiple samples."""
        return [self.predict(x) for x in X]
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        if x.ndim == 1:
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        else:
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _get_save_data(self) -> dict:
        return {
            'weights': self._weights,
            'bias': self._bias,
        }
    
    def _load_model_data(self, data: dict):
        self._weights = data['weights']
        self._bias = data['bias']
