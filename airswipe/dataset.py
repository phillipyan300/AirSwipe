"""
Dataset collection and management for AirSwipe.

Tools for recording labeled gesture samples and preparing training data.
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from .config import Config
from .segmentation import GestureLabel


@dataclass
class Sample:
    """A single recorded gesture sample."""
    label: str
    audio: np.ndarray
    timestamp: str
    duration: float
    carrier_freq: float
    sample_rate: int
    distance_cm: Optional[int] = None
    speed: Optional[str] = None  # slow, medium, fast
    notes: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary (audio as list for JSON)."""
        d = asdict(self)
        d['audio'] = self.audio.tolist()
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Sample':
        """Load from dictionary."""
        d['audio'] = np.array(d['audio'], dtype=np.float32)
        return cls(**d)


class DatasetCollector:
    """
    Interactive dataset collection tool.
    
    Records labeled gesture samples with metadata for training.
    """
    
    def __init__(self, config: Config, audio_source, dsp):
        self.config = config
        self.audio = audio_source
        self.dsp = dsp
        
        # Storage
        self.samples: List[Sample] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Recording state
        self._recording = False
        self._current_label: Optional[GestureLabel] = None
    
    def record_sample(self, label: GestureLabel, 
                     duration: Optional[float] = None,
                     distance_cm: Optional[int] = None,
                     speed: Optional[str] = None,
                     notes: Optional[str] = None) -> Sample:
        """
        Record a single labeled sample.
        
        Args:
            label: Gesture label
            duration: Recording duration (uses config default if None)
            distance_cm: Distance from laptop in cm
            speed: Gesture speed (slow/medium/fast)
            notes: Additional notes
            
        Returns:
            Recorded Sample
        """
        if duration is None:
            duration = self.config.segment_duration_sec
        
        # Record audio
        print(f"Recording {label.value} for {duration:.1f}s...")
        time.sleep(0.1)  # Small delay to prepare
        
        audio = self.audio.get_buffer(duration)
        
        sample = Sample(
            label=label.value,
            audio=audio.copy(),
            timestamp=datetime.now().isoformat(),
            duration=duration,
            carrier_freq=self.config.carrier_freq,
            sample_rate=self.config.sample_rate,
            distance_cm=distance_cm,
            speed=speed,
            notes=notes
        )
        
        self.samples.append(sample)
        print(f"✓ Recorded sample #{len(self.samples)} ({label.value})")
        
        return sample
    
    def interactive_collection(self, target_per_class: int = 30):
        """
        Interactive collection session.
        
        Guides user through recording samples for each class.
        """
        print("\n" + "=" * 60)
        print("  Dataset Collection Session")
        print("=" * 60)
        print(f"\nTarget: {target_per_class} samples per class")
        print("\nLabels: left, right, none")
        print("\nCommands:")
        print("  l - record LEFT swipe")
        print("  r - record RIGHT swipe")
        print("  n - record NONE (idle)")
        print("  s - show statistics")
        print("  q - quit and save")
        print("\n" + "-" * 60)
        
        try:
            while True:
                # Show current counts
                counts = self._get_label_counts()
                print(f"\nCurrent: left={counts.get('left', 0)}, "
                      f"right={counts.get('right', 0)}, "
                      f"none={counts.get('none', 0)}")
                
                cmd = input("\nCommand (l/r/n/s/q): ").strip().lower()
                
                if cmd == 'l':
                    print("\n>>> Perform LEFT swipe when ready...")
                    time.sleep(1)
                    self.record_sample(GestureLabel.LEFT)
                
                elif cmd == 'r':
                    print("\n>>> Perform RIGHT swipe when ready...")
                    time.sleep(1)
                    self.record_sample(GestureLabel.RIGHT)
                
                elif cmd == 'n':
                    print("\n>>> Stay still (recording idle)...")
                    time.sleep(0.5)
                    self.record_sample(GestureLabel.NONE)
                
                elif cmd == 's':
                    self._print_statistics()
                
                elif cmd == 'q':
                    break
                
                else:
                    print("Unknown command")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted")
        
        # Save on exit
        self.save()
    
    def _get_label_counts(self) -> Dict[str, int]:
        """Get count of samples per label."""
        counts = {}
        for s in self.samples:
            counts[s.label] = counts.get(s.label, 0) + 1
        return counts
    
    def _print_statistics(self):
        """Print dataset statistics."""
        counts = self._get_label_counts()
        total = len(self.samples)
        
        print("\n" + "=" * 40)
        print("  Dataset Statistics")
        print("=" * 40)
        print(f"\nTotal samples: {total}")
        print("\nPer class:")
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count} ({100*count/total:.1f}%)")
        print("=" * 40)
    
    def save(self, path: Optional[str] = None):
        """
        Save dataset to disk.
        
        Args:
            path: Save path (uses default if None)
        """
        if path is None:
            path = Path(self.config.dataset_dir) / f"dataset_{self.session_id}.json"
        else:
            path = Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = {
            'session_id': self.session_id,
            'config': {
                'carrier_freq': self.config.carrier_freq,
                'sample_rate': self.config.sample_rate,
                'fft_size': self.config.fft_size,
                'hop_size': self.config.hop_size,
            },
            'samples': [s.to_dict() for s in self.samples]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        print(f"\n✓ Saved {len(self.samples)} samples to {path}")
    
    def load(self, path: str):
        """Load dataset from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.session_id = data.get('session_id', 'loaded')
        self.samples = [Sample.from_dict(s) for s in data['samples']]
        
        print(f"✓ Loaded {len(self.samples)} samples")


class DatasetProcessor:
    """
    Prepares dataset for training.
    
    Extracts features and creates train/val/test splits.
    """
    
    def __init__(self, config: Config, dsp):
        self.config = config
        self.dsp = dsp
    
    def extract_spectrogram_patches(self, samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract spectrogram patches from samples.
        
        Args:
            samples: List of Sample objects
            
        Returns:
            (X, y) - patches array (N, T, F), labels array (N,)
        """
        X_list = []
        y_list = []
        
        label_map = {'none': 0, 'left': 1, 'right': 2}
        
        for sample in samples:
            # Compute spectrogram
            _, freqs, Sxx = self.dsp.compute_spectrogram(sample.audio)
            
            # Extract carrier band
            cf = self.config.carrier_freq
            bw = self.config.doppler_bandwidth
            freq_mask = (freqs >= cf - bw) & (freqs <= cf + bw)
            
            patch = Sxx[freq_mask, :]
            
            # Convert to log magnitude
            patch = 10 * np.log10(patch + 1e-10)
            
            # Resize to target dimensions
            patch = self._resize_patch(patch, 
                                       self.config.patch_freq_bins,
                                       self.config.patch_time_frames)
            
            X_list.append(patch)
            y_list.append(label_map[sample.label])
        
        return np.array(X_list), np.array(y_list)
    
    def extract_features(self, samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract hand-crafted features for baseline classifier.
        
        Returns:
            (X, y) - feature matrix (N, n_features), labels (N,)
        """
        from .model import BaselineClassifier
        
        X_list = []
        y_list = []
        
        label_map = {'none': 0, 'left': 1, 'right': 2}
        classifier = BaselineClassifier(self.config)
        
        for sample in samples:
            # Compute features over time
            _, _, Sxx = self.dsp.compute_spectrogram(sample.audio)
            n_frames = Sxx.shape[1]
            
            # Extract D(t) and A(t) series
            d_values = []
            a_values = []
            
            for t in range(n_frames):
                # Get single frame spectrum
                frame_audio = sample.audio[t * self.config.hop_size:
                                          t * self.config.hop_size + self.config.fft_size]
                if len(frame_audio) < self.config.fft_size:
                    break
                
                features = self.dsp.extract_features(frame_audio, use_baseline=False)
                d_values.append(features.d)
                a_values.append(features.a)
            
            if len(d_values) < 3:
                continue
            
            # Extract summary features
            feat_vec = classifier.extract_features(
                np.array(d_values),
                np.array(a_values),
                sample.duration
            )
            
            X_list.append(feat_vec)
            y_list.append(label_map[sample.label])
        
        return np.array(X_list), np.array(y_list)
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15) -> Dict:
        """
        Split data into train/val/test sets.
        
        Args:
            X: Features or patches
            y: Labels
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            
        Returns:
            Dict with 'train', 'val', 'test' keys
        """
        n = len(y)
        indices = np.random.permutation(n)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        return {
            'train': (X[train_idx], y[train_idx]),
            'val': (X[val_idx], y[val_idx]),
            'test': (X[test_idx], y[test_idx])
        }
    
    def _resize_patch(self, patch: np.ndarray, 
                     target_f: int, target_t: int) -> np.ndarray:
        """Resize spectrogram patch to target dimensions."""
        from scipy.ndimage import zoom
        
        current_f, current_t = patch.shape
        
        zoom_f = target_f / current_f
        zoom_t = target_t / current_t
        
        return zoom(patch, (zoom_f, zoom_t), order=1)
