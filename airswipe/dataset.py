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
        print("  a - AUTO mode (guided collection)")
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
                
                cmd = input("\nCommand (l/r/n/a/s/q): ").strip().lower()
                
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
                
                elif cmd == 'a':
                    self.auto_collection(target_per_class)
                
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
    
    def auto_collection(self, target_per_class: int = 30):
        """
        Automatic guided collection with countdowns.
        
        Cycles through gestures automatically with audio/visual cues.
        """
        print("\n" + "=" * 60)
        print("  AUTO COLLECTION MODE")
        print("=" * 60)
        print("\nThis will guide you through recording samples automatically.")
        print("Press Ctrl+C at any time to stop and return to menu.")
        print("\nGet ready! Starting in 3 seconds...")
        time.sleep(3)
        
        # Determine how many more samples we need for each class
        counts = self._get_label_counts()
        left_needed = max(0, target_per_class - counts.get('left', 0))
        right_needed = max(0, target_per_class - counts.get('right', 0))
        none_needed = max(0, target_per_class - counts.get('none', 0))
        
        # Build collection schedule - interleave for variety
        schedule = []
        max_needed = max(left_needed, right_needed, none_needed)
        
        for i in range(max_needed):
            if i < left_needed:
                schedule.append(GestureLabel.LEFT)
            if i < right_needed:
                schedule.append(GestureLabel.RIGHT)
            if i < none_needed:
                schedule.append(GestureLabel.NONE)
        
        if not schedule:
            print("\n✓ Already have enough samples for all classes!")
            return
        
        total = len(schedule)
        print(f"\nWill collect {total} samples:")
        print(f"  - {left_needed} LEFT swipes")
        print(f"  - {right_needed} RIGHT swipes")  
        print(f"  - {none_needed} NONE (idle)")
        print("\n" + "-" * 60)
        
        try:
            for idx, label in enumerate(schedule):
                progress = f"[{idx + 1}/{total}]"
                
                # Announce what's coming
                if label == GestureLabel.LEFT:
                    instruction = "LEFT SWIPE (hand moves →)"
                elif label == GestureLabel.RIGHT:
                    instruction = "RIGHT SWIPE (hand moves ←)"
                else:
                    instruction = "NONE (stay still)"
                
                print(f"\n{progress} Next: {instruction}")
                
                # Countdown
                for countdown in [3, 2, 1]:
                    print(f"  {countdown}...", end=" ", flush=True)
                    time.sleep(1)
                
                # Record
                if label == GestureLabel.NONE:
                    print("HOLD STILL!")
                else:
                    print("GO!")
                
                self.record_sample(label)
                
                # Brief pause between samples
                time.sleep(0.5)
                
                # Progress update every 5 samples
                if (idx + 1) % 5 == 0:
                    counts = self._get_label_counts()
                    print(f"\n  Progress: left={counts.get('left', 0)}, "
                          f"right={counts.get('right', 0)}, "
                          f"none={counts.get('none', 0)}")
            
            print("\n" + "=" * 60)
            print("  AUTO COLLECTION COMPLETE!")
            print("=" * 60)
            self._print_statistics()
            
        except KeyboardInterrupt:
            print("\n\n⚠ Auto collection interrupted")
            print("Returning to menu (samples so far are saved)...")
    
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
        
        # Auto-detect labels present in dataset
        labels_present = sorted(set(s.label for s in samples))
        label_map = {label: i for i, label in enumerate(labels_present)}
        print(f"  Labels: {labels_present} -> {label_map}")
        
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
    
    def extract_d_series(self, samples: List[Sample], seq_len: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract D(t) time series for 1D CNN.
        
        Args:
            samples: List of Sample objects
            seq_len: Target sequence length (will pad/truncate)
            
        Returns:
            (X, y) - D(t) series (N, seq_len), labels (N,)
        """
        X_list = []
        y_list = []
        
        # Auto-detect labels
        labels_present = sorted(set(s.label for s in samples))
        label_map = {label: i for i, label in enumerate(labels_present)}
        
        for sample in samples:
            # Extract D(t) series from audio
            d_values = []
            
            n_frames = len(sample.audio) // self.config.hop_size
            for t in range(n_frames):
                offset = t * self.config.hop_size
                frame = sample.audio[offset:offset + self.config.fft_size]
                if len(frame) < self.config.fft_size:
                    break
                
                features = self.dsp.extract_features(frame, use_baseline=False)
                d_values.append(features.d)
            
            if len(d_values) < 3:
                continue
            
            # Pad or truncate to seq_len
            d_array = np.array(d_values)
            if len(d_array) < seq_len:
                padded = np.zeros(seq_len)
                padded[:len(d_array)] = d_array
                d_array = padded
            elif len(d_array) > seq_len:
                # Take middle portion
                start = (len(d_array) - seq_len) // 2
                d_array = d_array[start:start + seq_len]
            
            X_list.append(d_array)
            y_list.append(label_map[sample.label])
        
        return np.array(X_list, dtype=np.float32), np.array(y_list)
    
    def extract_spectrograms(self, samples: List[Sample], 
                            freq_bins: int = 64, time_frames: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract spectrogram patches for 2D CNN.
        
        Args:
            samples: List of Sample objects
            freq_bins: Target frequency bins
            time_frames: Target time frames
            
        Returns:
            (X, y) - Spectrograms (N, freq_bins, time_frames), labels (N,)
        """
        from scipy.ndimage import zoom
        
        X_list = []
        y_list = []
        
        # Auto-detect labels
        labels_present = sorted(set(s.label for s in samples))
        label_map = {label: i for i, label in enumerate(labels_present)}
        
        for sample in samples:
            # Compute spectrogram
            times, freqs, Sxx = self.dsp.compute_spectrogram(sample.audio)
            
            # Filter to carrier band
            cf = self.config.carrier_freq
            bw = self.config.doppler_bandwidth * 2
            freq_min = cf - bw
            freq_max = cf + bw
            freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
            
            Sxx_filtered = Sxx[freq_mask, :]
            
            if Sxx_filtered.size == 0:
                continue
            
            # Convert to dB
            Sxx_db = 10 * np.log10(Sxx_filtered + 1e-10)
            
            # Resize to target dimensions
            if Sxx_db.shape != (freq_bins, time_frames):
                zoom_factors = (freq_bins / Sxx_db.shape[0], time_frames / Sxx_db.shape[1])
                Sxx_db = zoom(Sxx_db, zoom_factors, order=1)
            
            # Normalize to [0, 1]
            Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-8)
            
            X_list.append(Sxx_norm)
            y_list.append(label_map[sample.label])
        
        return np.array(X_list, dtype=np.float32), np.array(y_list)
    
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
