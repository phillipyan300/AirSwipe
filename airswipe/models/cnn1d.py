"""
1D CNN classifier operating on D(t) time series.

This model learns temporal patterns directly from the Doppler signal,
rather than relying on hand-crafted features.
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


class CNN1DClassifier(BaseModel):
    """
    1D Convolutional Neural Network on D(t) time series.
    
    Architecture:
        Input: D(t) sequence [batch, 1, seq_len]
        Conv1D(16) → ReLU → MaxPool
        Conv1D(32) → ReLU → MaxPool
        Flatten → Dense(64) → ReLU → Dropout
        Dense(n_classes) → Softmax
    
    Benefits over logistic regression:
    - Learns temporal patterns automatically
    - Translation invariant (swipe at start or end of window)
    - No manual feature engineering needed
    """
    
    def __init__(self, config, seq_len: int = 50):
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for CNN1D. Install with: pip install torch")
        
        self.seq_len = seq_len  # Expected input sequence length
        self._model: Optional[nn.Module] = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _build_model(self, n_classes: int):
        """Build the CNN architecture."""
        
        class CNN1D(nn.Module):
            def __init__(self, seq_len: int, n_classes: int):
                super().__init__()
                
                # Convolutional layers
                self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
                self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
                self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                
                self.pool = nn.MaxPool1d(2)
                self.dropout = nn.Dropout(0.3)
                
                # Calculate flattened size after convolutions
                # After 3 pooling layers: seq_len / 8
                flat_size = 64 * (seq_len // 8)
                
                # Fully connected layers
                self.fc1 = nn.Linear(flat_size, 64)
                self.fc2 = nn.Linear(64, n_classes)
            
            def forward(self, x):
                # x: [batch, 1, seq_len]
                x = self.pool(F.relu(self.conv1(x)))  # [batch, 16, seq_len/2]
                x = self.pool(F.relu(self.conv2(x)))  # [batch, 32, seq_len/4]
                x = self.pool(F.relu(self.conv3(x)))  # [batch, 64, seq_len/8]
                
                x = x.view(x.size(0), -1)  # Flatten
                x = self.dropout(F.relu(self.fc1(x)))
                x = self.fc2(x)
                
                return x
        
        return CNN1D(self.seq_len, n_classes).to(self._device)
    
    def preprocess(self, d_values: np.ndarray) -> np.ndarray:
        """
        Preprocess D(t) sequence for the model.
        
        - Pad or truncate to seq_len
        - Normalize to zero mean, unit variance
        
        Args:
            d_values: Raw D(t) values
            
        Returns:
            Preprocessed sequence of shape (seq_len,)
        """
        # Pad or truncate
        if len(d_values) < self.seq_len:
            # Pad with zeros at the end
            padded = np.zeros(self.seq_len)
            padded[:len(d_values)] = d_values
            d_values = padded
        elif len(d_values) > self.seq_len:
            # Take the middle portion
            start = (len(d_values) - self.seq_len) // 2
            d_values = d_values[start:start + self.seq_len]
        
        # Normalize (zero mean, unit variance)
        mean = np.mean(d_values)
        std = np.std(d_values) + 1e-8
        normalized = (d_values - mean) / std
        
        return normalized.astype(np.float32)
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100,
            batch_size: int = 16,
            learning_rate: float = 0.001,
            verbose: bool = True):
        """
        Train the 1D CNN.
        
        Args:
            X: Input sequences (n_samples, seq_len) - raw D(t) values
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
        
        # Preprocess all samples
        X_processed = np.array([self.preprocess(x) for x in X])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_processed).unsqueeze(1).to(self._device)  # [N, 1, seq_len]
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
    
    def predict(self, d_values: np.ndarray) -> Prediction:
        """
        Predict class for a single D(t) sequence.
        
        Args:
            d_values: D(t) values (any length, will be preprocessed)
            
        Returns:
            Prediction
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Preprocess
        x = self.preprocess(d_values)
        x_tensor = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0).to(self._device)  # [1, 1, seq_len]
        
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
        """Predict classes for multiple D(t) sequences."""
        return [self.predict(x) for x in X]
    
    def _get_save_data(self) -> dict:
        return {
            'model_state': self._model.state_dict() if self._model else None,
            'seq_len': self.seq_len,
        }
    
    def _load_model_data(self, data: dict):
        self.seq_len = data.get('seq_len', 50)
        
        # Rebuild model with correct number of classes
        n_classes = len(self._labels)
        self._model = self._build_model(n_classes)
        
        if data.get('model_state'):
            self._model.load_state_dict(data['model_state'])
        
        self._model.eval()
