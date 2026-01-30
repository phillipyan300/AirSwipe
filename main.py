#!/usr/bin/env python3
"""
AirSwipe - Ultrasonic Doppler Gesture Recognition for macOS

Detects hand swipe gestures using ultrasonic audio and Doppler shift analysis.
Uses only the built-in MacBook speaker and microphone.

Usage:
    python main.py scan           # Find best carrier frequency
    python main.py visualize      # Real-time spectrogram and features
    python main.py detect         # Gesture detection mode
    python main.py track          # Motion tracking visualization (Part 2)
    python main.py collect        # Collect training data
    python main.py train          # Train classifier

Author: AirSwipe Project
License: MIT
"""

import argparse
import sys
import time

from airswipe import Config
from airswipe.audio_rx import AudioDuplex
from airswipe.dsp import DSP, CarrierScanner
from airswipe.segmentation import GestureSegmenter, GestureLabel
from airswipe.tracker import MotionTracker, KalmanTracker
from airswipe.model import BaselineClassifier
from airswipe.ui import ConsoleUI, MatplotlibUI


def cmd_scan(args):
    """Scan for best carrier frequency."""
    config = Config(carrier_freq=args.freq)
    
    print("\n" + "=" * 60)
    print("  AirSwipe - Carrier Frequency Scan")
    print("=" * 60)
    print("\nThis will test different ultrasonic frequencies to find")
    print("the best carrier for your Mac's speaker/mic hardware.")
    print("\nKeep hands away from laptop during scan.\n")
    
    with AudioDuplex(config) as audio:
        scanner = CarrierScanner(config)
        time.sleep(1)  # Let audio stabilize
        
        best_freq, results = scanner.scan(audio)
        
        print("\n" + "-" * 60)
        print("Results:")
        print("-" * 60)
        for freq in sorted(results.keys()):
            r = results[freq]
            marker = " ★" if freq == best_freq else ""
            print(f"  {freq:,.0f} Hz: SNR={r['snr']:.1f}dB, "
                  f"stability={r['stability']:.2f}{marker}")
        
        print(f"\n✓ Recommended carrier: {best_freq:,.0f} Hz")
        print(f"\nUse: python main.py detect --freq {best_freq:.0f}")


def cmd_visualize(args):
    """Real-time visualization mode."""
    from pathlib import Path
    
    config = Config(
        carrier_freq=args.freq,
        activity_threshold=args.threshold,
        tone_amplitude=args.amplitude
    )
    
    print("\n" + "=" * 60)
    print("  AirSwipe - Real-time Visualization")
    print("=" * 60)
    print("\nShowing live spectrogram and Doppler features.")
    print("Wave your hand in front of the laptop to see motion!")
    
    dsp = DSP(config)
    tracker = MotionTracker(config) if args.tracking else None
    segmenter = None
    classifier = None
    
    # Load model for detection if requested
    if args.detect:
        import pickle
        model_path = Path(args.model) if args.model else Path("data/model.pkl")
        if model_path.exists():
            try:
                from airswipe.segmentation import GestureSegmenter
                
                segmenter = GestureSegmenter(config)
                
                # Detect model type from file
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                model_type = data.get('model_type', 'BaselineClassifier')
                
                if 'CNN1D' in model_type:
                    from airswipe.models.cnn1d import CNN1DClassifier
                    classifier = CNN1DClassifier(config, seq_len=data.get('seq_len', 50))
                    classifier.load(str(model_path))
                    print(f"\n✓ Detection enabled - loaded CNN1D from {model_path}")
                elif 'CNN2D' in model_type:
                    from airswipe.models.cnn2d import CNN2DClassifier
                    classifier = CNN2DClassifier(config)
                    classifier.load(str(model_path))
                    print(f"\n✓ Detection enabled - loaded CNN2D from {model_path}")
                else:
                    from airswipe.model import BaselineClassifier
                    classifier = BaselineClassifier(config)
                    classifier.load(str(model_path))
                    print(f"\n✓ Detection enabled - loaded Logistic from {model_path}")
                    
            except Exception as e:
                print(f"\n⚠ Could not load model: {e}")
                print("  Visualization only (no detection)")
        else:
            print(f"\n⚠ Model not found at {model_path}")
            print("  Run 'python main.py train' first, or use --model to specify path")
    
    print("\nClose window or Ctrl+C to exit.\n")
    
    with AudioDuplex(config) as audio:
        time.sleep(0.5)  # Let buffer fill
        
        ui = MatplotlibUI(config, audio, dsp, tracker, 
                         segmenter=segmenter, classifier=classifier)
        ui.start()


def cmd_detect(args):
    """Gesture detection mode."""
    from pathlib import Path
    import numpy as np
    import pickle
    
    config = Config(
        carrier_freq=args.freq,
        activity_threshold=args.threshold,
        tone_amplitude=args.amplitude,
        cooldown_sec=args.cooldown
    )
    
    print("\n" + "=" * 60)
    print("  AirSwipe - Gesture Detection")
    print("=" * 60)
    print(f"\nCarrier: {config.carrier_freq:,.0f} Hz")
    print(f"Threshold: {config.activity_threshold}")
    
    dsp = DSP(config)
    segmenter = GestureSegmenter(config)
    ui = ConsoleUI(config)
    
    # Try to load trained model - detect model type from file
    model_path = Path(args.model) if args.model else Path("data/model.pkl")
    use_trained_model = False
    classifier = None
    model_type = 'logistic'  # default
    
    if model_path.exists():
        try:
            # Peek at model file to detect type
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            model_type = data.get('model_type', 'BaselineClassifier')
            
            # Load appropriate model class
            if 'CNN1D' in model_type:
                from airswipe.models.cnn1d import CNN1DClassifier
                classifier = CNN1DClassifier(config, seq_len=data.get('seq_len', 50))
                classifier.load(str(model_path))
                model_type = 'cnn1d'
                print(f"✓ Loaded CNN1D model from {model_path}")
            elif 'CNN2D' in model_type:
                from airswipe.models.cnn2d import CNN2DClassifier
                classifier = CNN2DClassifier(config)
                classifier.load(str(model_path))
                model_type = 'cnn2d'
                print(f"✓ Loaded CNN2D model from {model_path}")
            else:
                # Logistic regression (BaselineClassifier or LogisticClassifier)
                classifier = BaselineClassifier(config)
                classifier.load(str(model_path))
                model_type = 'logistic'
                print(f"✓ Loaded Logistic model from {model_path}")
            
            use_trained_model = True
        except Exception as e:
            print(f"⚠ Could not load model: {e}")
            print("  Using heuristic classifier instead")
            classifier = BaselineClassifier(config)
            model_type = 'logistic'
    else:
        print(f"⚠ No trained model found at {model_path}")
        print("  Using heuristic classifier (run 'python main.py train' first)")
        classifier = BaselineClassifier(config)
    
    print(f"Model type: {model_type}")
    print("\nDetecting LEFT and RIGHT swipes...")
    print("Press Ctrl+C to exit.\n")
    
    # Gesture callback - uses trained model if available
    def on_gesture(event):
        if use_trained_model and len(event.features) >= 3:
            # Extract D(t) values from gesture
            d_values = np.array([f.d for f in event.features])
            a_values = np.array([f.a for f in event.features])
            
            if model_type == 'cnn1d':
                # CNN expects raw D(t) series - it handles its own preprocessing
                pred = classifier.predict(d_values)
                print(f"\n  [CNN1D] D len={len(d_values)}, pred={pred.label.value}, conf={pred.confidence:.1%}")
                
            elif model_type == 'cnn2d':
                # CNN2D expects spectrogram - not fully supported in detect yet
                print(f"\n  [CNN2D] Not fully supported in detect yet")
                return
                
            else:
                # Logistic expects feature vector
                feat_vec = classifier.extract_features(d_values, a_values, event.duration)
                print(f"\n  [Logistic] slope={feat_vec[3]:.1f}, peak_asym={feat_vec[9]:.3f}")
                
                # Normalize using stored mean/std from training
                feat_vec_norm = classifier.normalize_features(feat_vec)
                
                # Get prediction from trained model
                pred = classifier.predict(feat_vec_norm)
            
            # Update event with trained model's prediction
            event.label = pred.label
            event.confidence = pred.confidence
        
        ui.print_gesture(event)
    
    segmenter.set_gesture_callback(on_gesture)
    
    try:
        with AudioDuplex(config) as audio:
            # Skip baseline calibration since training used use_baseline=False
            # This ensures live features match training features
            print("Starting detection (no baseline - matching training mode)...\n")
            time.sleep(1)  # Brief pause to let audio stabilize
            
            # Detection loop
            while True:
                time.sleep(config.time_resolution)
                
                samples = audio.get_buffer(0.1)
                if len(samples) < config.fft_size:
                    continue
                
                # Extract features WITHOUT baseline (same as training)
                features = dsp.extract_features(samples, use_baseline=False)
                event = segmenter.update(features)
                
                ui.update(features, gesture=event)
    
    except KeyboardInterrupt:
        print("\n\nStopping...")


def cmd_track(args):
    """Motion tracking mode (Part 2 demo)."""
    config = Config(
        carrier_freq=args.freq,
        activity_threshold=args.threshold,
        tone_amplitude=args.amplitude,
        velocity_ema_alpha=args.smoothing
    )
    
    print("\n" + "=" * 60)
    print("  AirSwipe - Motion Tracking (Part 2)")
    print("=" * 60)
    print("\nShowing continuous motion tracking visualization.")
    print("Move hand left/right to see velocity and position tracking!")
    print("\nClose window or Ctrl+C to exit.\n")
    
    dsp = DSP(config)
    
    if args.kalman:
        tracker = KalmanTracker(config)
        print("Using Kalman filter for tracking")
    else:
        tracker = MotionTracker(config)
        print("Using EMA filter for tracking")
    
    with AudioDuplex(config) as audio:
        time.sleep(0.5)
        
        ui = MatplotlibUI(config, audio, dsp, tracker)
        ui.start()


def cmd_collect(args):
    """Dataset collection mode."""
    from airswipe.dataset import DatasetCollector
    
    config = Config(
        carrier_freq=args.freq,
        tone_amplitude=args.amplitude
    )
    
    print("\n" + "=" * 60)
    print("  AirSwipe - Dataset Collection")
    print("=" * 60)
    
    dsp = DSP(config)
    
    with AudioDuplex(config) as audio:
        time.sleep(0.5)
        
        collector = DatasetCollector(config, audio, dsp)
        collector.interactive_collection(target_per_class=args.samples)


def cmd_train(args):
    """Train classifier."""
    from airswipe.dataset import DatasetProcessor
    from pathlib import Path
    import json
    import numpy as np
    
    config = Config()
    dsp = DSP(config)
    processor = DatasetProcessor(config, dsp)
    
    model_type = getattr(args, 'model_type', 'logistic')
    
    print("\n" + "=" * 60)
    print(f"  AirSwipe - Training ({model_type.upper()})")
    print("=" * 60)
    
    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    
    with open(dataset_path) as f:
        data = json.load(f)
    
    from airswipe.dataset import Sample
    samples = [Sample.from_dict(s) for s in data['samples']]
    print(f"\nLoaded {len(samples)} samples")
    
    # Filter for binary classification if requested
    if args.binary:
        samples = [s for s in samples if s.label != 'none']
        print(f"Binary mode: filtered to {len(samples)} samples (LEFT/RIGHT only)")
    
    # Create label mapping
    labels_present = sorted(set(s.label for s in samples))
    label_map = {label: i for i, label in enumerate(labels_present)}
    print(f"Labels: {labels_present} -> {label_map}")
    
    # Different data extraction based on model type
    if model_type == 'logistic':
        # Extract hand-crafted features
        print("\nExtracting hand-crafted features...")
        X, y = processor.extract_features(samples)
        print(f"Feature matrix shape: {X.shape}")
        
        # Show feature statistics
        print("\nFeature statistics (before normalization):")
        feature_names = ['mean_d', 'max_d', 'min_d', 'slope_d', 'signed_area', 'duration', 'max_a', 'std_d']
        for i, name in enumerate(feature_names[:min(8, X.shape[1])]):
            print(f"  {name:12s}: min={X[:, i].min():12.2f}, max={X[:, i].max():12.2f}, mean={X[:, i].mean():12.2f}")
        
    elif model_type == 'cnn1d':
        # Extract D(t) time series
        print("\nExtracting D(t) time series for CNN1D...")
        X, y = processor.extract_d_series(samples, seq_len=50)
        print(f"D(t) series shape: {X.shape}")
        
    elif model_type == 'cnn2d':
        # Extract spectrograms
        print("\nExtracting spectrograms for CNN2D...")
        X, y = processor.extract_spectrograms(samples)
        print(f"Spectrogram shape: {X.shape}")
    
    # Split data
    splits = processor.train_test_split(X, y)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    print(f"\nTrain: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Train model based on type
    if model_type == 'logistic':
        # Normalize features
        feature_mean = X_train.mean(axis=0)
        feature_std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - feature_mean) / feature_std
        X_test_norm = (X_test - feature_mean) / feature_std
        
        print("\nTraining Logistic Regression...")
        classifier = BaselineClassifier(config)
        classifier.fit(X_train_norm, y_train, epochs=500, learning_rate=0.1)
        classifier._feature_mean = feature_mean
        classifier._feature_std = feature_std
        
    elif model_type == 'cnn1d':
        from airswipe.models.cnn1d import CNN1DClassifier
        print("\nTraining 1D CNN...")
        classifier = CNN1DClassifier(config, seq_len=X.shape[1])
        classifier.fit(X_train, y_train, epochs=100, batch_size=8)
        X_test_norm = X_test  # CNN handles its own normalization
        
    elif model_type == 'cnn2d':
        from airswipe.models.cnn2d import CNN2DClassifier
        print("\nTraining 2D CNN...")
        classifier = CNN2DClassifier(config, freq_bins=X.shape[1], time_frames=X.shape[2])
        classifier.fit(X_train, y_train, epochs=100, batch_size=8)
        X_test_norm = X_test  # CNN handles its own normalization
    
    # Get label mapping from classifier
    label_names = [l.value for l in classifier._labels]
    print(f"Model labels: {label_names}")
    
    # Evaluate on normalized test set
    correct = 0
    for x, y_true in zip(X_test_norm, y_test):
        pred = classifier.predict(x)
        if pred.label.value == label_names[y_true]:
            correct += 1
    
    accuracy = correct / len(y_test)
    print(f"\nTest accuracy: {accuracy:.1%}")
    
    # Per-class accuracy
    print("\nPer-class results:")
    for class_idx, class_name in enumerate(label_names):
        mask = y_test == class_idx
        if mask.sum() > 0:
            class_correct = sum(
                1 for x, y_true in zip(X_test_norm[mask], y_test[mask])
                if classifier.predict(x).label.value == class_name
            )
            class_acc = class_correct / mask.sum()
            print(f"  {class_name}: {class_correct}/{mask.sum()} ({class_acc:.1%})")
    
    # Save model (include model type in filename)
    model_path = dataset_path.parent / f"model_{model_type}.pkl"
    classifier.save(str(model_path))
    print(f"\n✓ Model saved to {model_path}")
    
    # Also save as default model.pkl for compatibility
    default_path = dataset_path.parent / "model.pkl"
    classifier.save(str(default_path))
    print(f"✓ Also saved to {default_path} (default)")


def main():
    parser = argparse.ArgumentParser(
        description="AirSwipe - Ultrasonic Doppler Gesture Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py scan                      # Find best carrier frequency
    python main.py visualize                 # See spectrogram
    python main.py visualize --tracking      # With motion tracking
    python main.py detect                    # Detect left/right swipes
    python main.py track                     # Motion tracking demo
    python main.py collect --samples 50      # Collect training data
    python main.py train --dataset data/dataset.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Common arguments - use Config defaults as single source of truth
    default_config = Config()
    def add_common_args(p):
        p.add_argument('--freq', type=float, default=default_config.carrier_freq,
                      help=f'Carrier frequency in Hz (default: {default_config.carrier_freq:.0f})')
        p.add_argument('--threshold', type=float, default=default_config.activity_threshold,
                      help=f'Activity threshold - higher = fewer detections (default: {default_config.activity_threshold})')
        p.add_argument('--amplitude', type=float, default=default_config.tone_amplitude,
                      help=f'Tone amplitude 0-1 (default: {default_config.tone_amplitude})')
    
    # Scan command
    p_scan = subparsers.add_parser('scan', help='Find best carrier frequency')
    add_common_args(p_scan)
    
    # Visualize command
    p_viz = subparsers.add_parser('visualize', help='Real-time visualization')
    add_common_args(p_viz)
    p_viz.add_argument('--tracking', action='store_true',
                      help='Show motion tracking plots')
    p_viz.add_argument('--detect', action='store_true',
                      help='Enable gesture detection with ML predictions')
    p_viz.add_argument('--model', type=str, default=None,
                      help='Path to trained model (default: data/model.pkl)')
    
    # Detect command
    p_detect = subparsers.add_parser('detect', help='Gesture detection')
    add_common_args(p_detect)
    p_detect.add_argument('--cooldown', type=float, default=0.5,
                         help='Cooldown between gestures (default: 0.5s)')
    p_detect.add_argument('--model', type=str, default=None,
                         help='Path to trained model (default: data/model.pkl)')
    
    # Track command
    p_track = subparsers.add_parser('track', help='Motion tracking demo')
    add_common_args(p_track)
    p_track.add_argument('--smoothing', type=float, default=0.85,
                        help='EMA smoothing factor (default: 0.85)')
    p_track.add_argument('--kalman', action='store_true',
                        help='Use Kalman filter instead of EMA')
    
    # Collect command
    p_collect = subparsers.add_parser('collect', help='Collect training data')
    add_common_args(p_collect)
    p_collect.add_argument('--samples', type=int, default=30,
                          help='Target samples per class (default: 30)')
    
    # Train command
    p_train = subparsers.add_parser('train', help='Train classifier')
    p_train.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset JSON file')
    p_train.add_argument('--binary', action='store_true',
                        help='Binary classification (LEFT/RIGHT only, ignore NONE)')
    p_train.add_argument('--model-type', type=str, default='logistic',
                        choices=['logistic', 'cnn1d', 'cnn2d'],
                        help='Model type: logistic, cnn1d, or cnn2d (default: logistic)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch
    commands = {
        'scan': cmd_scan,
        'visualize': cmd_visualize,
        'detect': cmd_detect,
        'track': cmd_track,
        'collect': cmd_collect,
        'train': cmd_train,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
