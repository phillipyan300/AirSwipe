"""
2D CNN classifier operating on spectrogram patches.

This model learns patterns directly from the time-frequency representation.
"""

import numpy as np
from typing import List, Optional, Tuple

from .base import BaseModel, Prediction
from ..segmentation import GestureLabel

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CNN2DClassifier(BaseModel):
    """
    2D Convolutional Neural Network on spectrogram patches.
    
    Architecture:
        Input: Spectrogram [batch, 1, freq_bins, time_frames]
        Conv2D(16) → ReLU → MaxPool
        Conv2D(32) → ReLU → MaxPool
        Conv2D(64) → ReLU → MaxPool
        Flatten → Dense(128) → ReLU → Dropout
        Dense(n_classes) → Softmax
    
    Benefits:
    - Sees full time-frequency information
    - Can learn complex spectral patterns
    - Most powerful but requires more data
    """
    
    def __init__(self, config, freq_bins: int = 64, time_frames: int = 64):
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for CNN2D. Install with: pip install torch")
        
        self.freq_bins = freq_bins
        self.time_frames = time_frames
        self._model: Optional[nn.Module] = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _build_model(self, n_classes: int):
        """Build the CNN architecture."""
        
        class CNN2D(nn.Module):
            def __init__(self, freq_bins: int, time_frames: int, n_classes: int):
                super().__init__()
                
                # Convolutional layers
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                
                self.pool = nn.MaxPool2d(2)
                self.dropout = nn.Dropout(0.4)
                
                # Calculate flattened size after convolutions
                # After 3 pooling layers: size / 8
                flat_h = freq_bins // 8
                flat_w = time_frames // 8
                flat_size = 64 * flat_h * flat_w
                
                # Fully connected layers
                self.fc1 = nn.Linear(flat_size, 128)
                self.fc2 = nn.Linear(128, n_classes)
            
            def forward(self, x):
                # x: [batch, 1, freq_bins, time_frames]
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = self.pool(F.relu(self.conv3(x)))
                
                x = x.view(x.size(0), -1)  # Flatten
                x = self.dropout(F.relu(self.fc1(x)))
                x = self.fc2(x)
                
                return x
        
        return CNN2D(self.freq_bins, self.time_frames, n_classes).to(self._device)
    
    def preprocess(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Preprocess spectrogram for the model.
        
        - Resize to (freq_bins, time_frames)
        - Convert to log scale
        - Normalize
        
        Args:
            spectrogram: Raw spectrogram (freq, time)
            
        Returns:
            Preprocessed spectrogram of shape (freq_bins, time_frames)
        """
        from scipy.ndimage import zoom
        
        # Resize if needed
        if spectrogram.shape != (self.freq_bins, self.time_frames):
            zoom_factors = (self.freq_bins / spectrogram.shape[0], 
                          self.time_frames / spectrogram.shape[1])
            spectrogram = zoom(spectrogram, zoom_factors, order=1)
        
        # Convert to log scale (dB)
        spectrogram = 10 * np.log10(spectrogram + 1e-10)
        
        # Normalize to [0, 1]
        min_val = spectrogram.min()
        max_val = spectrogram.max()
        if max_val > min_val:
            spectrogram = (spectrogram - min_val) / (max_val - min_val)
        
        return spectrogram.astype(np.float32)
    
    def extract_patch(self, audio: np.ndarray, dsp) -> np.ndarray:
        """
        Extract spectrogram patch from audio.
        
        Args:
            audio: Audio samples
            dsp: DSP object for computing spectrogram
            
        Returns:
            Spectrogram patch around carrier frequency
        """
        times, freqs, Sxx = dsp.compute_spectrogram(audio)
        
        # Filter to carrier band
        cf = self.config.carrier_freq
        bw = self.config.doppler_bandwidth * 2
        freq_min = cf - bw
        freq_max = cf + bw
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
        
        Sxx_filtered = Sxx[freq_mask, :]
        
        return self.preprocess(Sxx_filtered)
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100,
            batch_size: int = 16,
            learning_rate: float = 0.001,
            verbose: bool = True):
        """
        Train the 2D CNN.
        
        Args:
            X: Input spectrograms (n_samples, freq_bins, time_frames)
            y: Labels (n_samples,)
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate for Adam optimizer
            verbose: Print progress
        """
        n_classes = len(np.unique(y))
        
        # Set labels
        if n_classes == 2:
            self._labels = [GestureLabel.LEFT, GestureLabel.RIGHT]
        else:
            self._labels = [GestureLabel.NONE, GestureLabel.LEFT, GestureLabel.RIGHT]
        
        # Build model
        self._model = self._build_model(n_classes)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self._device)  # [N, 1, H, W]
        y_tensor = torch.LongTensor(y).to(self._device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self._model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                
                outputs = self._model(batch_x)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
            
            accuracy = correct / total
            avg_loss = total_loss / len(loader)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={avg_loss:.4f}, accuracy={accuracy:.3f}")
            
            # Early stopping if perfect accuracy
            if accuracy == 1.0 and epoch > 20:
                if verbose:
                    print(f"Epoch {epoch}: loss={avg_loss:.4f}, accuracy={accuracy:.3f} (early stop)")
                break
        
        self._fitted = True
    
    def predict(self, spectrogram: np.ndarray) -> Prediction:
        """
        Predict class for a single spectrogram.
        
        Args:
            spectrogram: Spectrogram patch
            
        Returns:
            Prediction
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Preprocess
        x = self.preprocess(spectrogram) if spectrogram.max() > 1 else spectrogram
        x_tensor = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0).to(self._device)
        
        # Predict
        self._model.eval()
        with torch.no_grad():
            logits = self._model(x_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        
        label_idx = np.argmax(probs)
        
        return Prediction(
            label=self._labels[label_idx],
            confidence=probs[label_idx],
            probabilities=probs
        )
    
    def predict_batch(self, X: np.ndarray) -> List[Prediction]:
        """Predict classes for multiple spectrograms."""
        return [self.predict(x) for x in X]
    
    def _get_save_data(self) -> dict:
        return {
            'model_state': self._model.state_dict() if self._model else None,
            'freq_bins': self.freq_bins,
            'time_frames': self.time_frames,
        }
    
    def _load_model_data(self, data: dict):
        self.freq_bins = data.get('freq_bins', 64)
        self.time_frames = data.get('time_frames', 64)
        
        # Rebuild model with correct number of classes
        n_classes = len(self._labels)
        self._model = self._build_model(n_classes)
        
        if data.get('model_state'):
            self._model.load_state_dict(data['model_state'])
        
        self._model.eval()
