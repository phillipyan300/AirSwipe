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
    print("\nClose window or Ctrl+C to exit.\n")
    
    dsp = DSP(config)
    tracker = MotionTracker(config) if args.tracking else None
    
    with AudioDuplex(config) as audio:
        time.sleep(0.5)  # Let buffer fill
        
        ui = MatplotlibUI(config, audio, dsp, tracker)
        ui.start()


def cmd_detect(args):
    """Gesture detection mode."""
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
    print("\nDetecting LEFT and RIGHT swipes...")
    print("Press Ctrl+C to exit.\n")
    
    dsp = DSP(config)
    segmenter = GestureSegmenter(config)
    classifier = BaselineClassifier(config)
    ui = ConsoleUI(config)
    
    # Gesture callback
    def on_gesture(event):
        ui.print_gesture(event)
    
    segmenter.set_gesture_callback(on_gesture)
    
    try:
        with AudioDuplex(config) as audio:
            # Calibration phase
            print("Calibrating baseline (keep hands away)...", end=" ", flush=True)
            time.sleep(2)
            
            for _ in range(50):
                samples = audio.get_buffer(0.1)
                if len(samples) >= config.fft_size:
                    spectrum = dsp.compute_spectrum(samples)
                    dsp.update_baseline(spectrum)
                time.sleep(0.02)
            
            print("done!\n")
            
            # Detection loop
            while True:
                time.sleep(config.time_resolution)
                
                samples = audio.get_buffer(0.1)
                if len(samples) < config.fft_size:
                    continue
                
                features = dsp.extract_features(samples)
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
    
    config = Config()
    dsp = DSP(config)
    processor = DatasetProcessor(config, dsp)
    
    print("\n" + "=" * 60)
    print("  AirSwipe - Training")
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
    
    # Extract features
    print("Extracting features...")
    X, y = processor.extract_features(samples)
    print(f"Feature matrix shape: {X.shape}")
    
    # Split
    splits = processor.train_test_split(X, y)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Train baseline classifier
    print("\nTraining baseline classifier...")
    classifier = BaselineClassifier(config)
    classifier.fit(X_train, y_train, epochs=500)
    
    # Evaluate
    correct = 0
    for x, y_true in zip(X_test, y_test):
        pred = classifier.predict(x)
        if pred.label.value == ['none', 'left', 'right'][y_true]:
            correct += 1
    
    accuracy = correct / len(y_test)
    print(f"\nTest accuracy: {accuracy:.1%}")
    
    # Save model
    model_path = dataset_path.parent / "model.pkl"
    classifier.save(str(model_path))
    print(f"\n✓ Model saved to {model_path}")


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
    
    # Common arguments
    def add_common_args(p):
        p.add_argument('--freq', type=float, default=18500,
                      help='Carrier frequency in Hz (default: 18500)')
        p.add_argument('--threshold', type=float, default=0.5,
                      help='Activity threshold (default: 0.5)')
        p.add_argument('--amplitude', type=float, default=0.3,
                      help='Tone amplitude 0-1 (default: 0.3)')
    
    # Scan command
    p_scan = subparsers.add_parser('scan', help='Find best carrier frequency')
    add_common_args(p_scan)
    
    # Visualize command
    p_viz = subparsers.add_parser('visualize', help='Real-time visualization')
    add_common_args(p_viz)
    p_viz.add_argument('--tracking', action='store_true',
                      help='Show motion tracking plots')
    
    # Detect command
    p_detect = subparsers.add_parser('detect', help='Gesture detection')
    add_common_args(p_detect)
    p_detect.add_argument('--cooldown', type=float, default=0.5,
                         help='Cooldown between gestures (default: 0.5s)')
    
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
