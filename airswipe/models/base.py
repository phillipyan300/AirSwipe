"""
Base class and common types for all models.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pickle

from ..segmentation import GestureLabel


@dataclass
class Prediction:
    """Classification prediction result."""
    label: GestureLabel
    confidence: float
    probabilities: np.ndarray


class BaseModel(ABC):
    """
    Abstract base class for all gesture classifiers.
    
    All models must implement:
    - fit(): Train the model
    - predict(): Make predictions
    - save()/load(): Persistence
    """
    
    def __init__(self, config):
        self.config = config
        self._fitted = False
        self._labels: List[GestureLabel] = [GestureLabel.LEFT, GestureLabel.RIGHT]
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
    
    @property
    def name(self) -> str:
        """Model name for display."""
        return self.__class__.__name__
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Train the model.
        
        Args:
            X: Training data (format depends on model type)
            y: Labels
            **kwargs: Model-specific training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> Prediction:
        """
        Predict class for a single sample.
        
        Args:
            x: Single sample (format depends on model type)
            
        Returns:
            Prediction with label, confidence, probabilities
        """
        pass
    
    @abstractmethod
    def predict_batch(self, X: np.ndarray) -> List[Prediction]:
        """
        Predict classes for multiple samples.
        
        Args:
            X: Batch of samples
            
        Returns:
            List of Predictions
        """
        pass
    
    def set_labels(self, labels: List[GestureLabel]):
        """Set the label mapping for predictions."""
        self._labels = labels
    
    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization parameters (computed from training data)."""
        self._feature_mean = mean
        self._feature_std = std
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply normalization to features."""
        if self._feature_mean is not None and self._feature_std is not None:
            return (x - self._feature_mean) / (self._feature_std + 1e-8)
        return x
    
    def save(self, path: str):
        """Save model to file."""
        data = self._get_save_data()
        data['labels'] = [l.value for l in self._labels]
        data['feature_mean'] = self._feature_mean
        data['feature_std'] = self._feature_std
        data['fitted'] = self._fitted
        data['model_type'] = self.name
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self._labels = [GestureLabel(l) for l in data.get('labels', ['left', 'right'])]
        self._feature_mean = data.get('feature_mean')
        self._feature_std = data.get('feature_std')
        self._fitted = data.get('fitted', True)
        
        self._load_model_data(data)
    
    @abstractmethod
    def _get_save_data(self) -> dict:
        """Get model-specific data to save."""
        pass
    
    @abstractmethod
    def _load_model_data(self, data: dict):
        """Load model-specific data."""
        pass
