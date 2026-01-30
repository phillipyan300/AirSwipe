"""
Model implementations for AirSwipe gesture classification.

Available models:
- LogisticClassifier: Simple logistic regression on hand-crafted features
- CNN1D: 1D CNN on D(t) time series
- CNN2D: 2D CNN on spectrogram patches
"""

from .base import BaseModel, Prediction
from .logistic import LogisticClassifier

# Lazy imports for PyTorch models (only import when needed)
CNN1DClassifier = None
CNN2DClassifier = None

__all__ = [
    'BaseModel',
    'Prediction', 
    'LogisticClassifier',
    'CNN1DClassifier',
    'CNN2DClassifier',
    'get_model'
]


def get_model(model_type: str, config):
    """
    Factory function to get a model by name.
    
    Args:
        model_type: One of 'logistic', 'cnn1d', 'cnn2d'
        config: Config object
        
    Returns:
        Model instance
    """
    global CNN1DClassifier, CNN2DClassifier
    
    if model_type == 'logistic':
        return LogisticClassifier(config)
    
    elif model_type == 'cnn1d':
        if CNN1DClassifier is None:
            from .cnn1d import CNN1DClassifier as _CNN1D
            CNN1DClassifier = _CNN1D
        return CNN1DClassifier(config)
    
    elif model_type == 'cnn2d':
        if CNN2DClassifier is None:
            from .cnn2d import CNN2DClassifier as _CNN2D
            CNN2DClassifier = _CNN2D
        return CNN2DClassifier(config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: logistic, cnn1d, cnn2d")
