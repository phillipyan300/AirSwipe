# AirSwipe 🌊👋

**Ultrasonic Doppler Gesture Recognition for macOS**

Detect hand swipe gestures using only your MacBook's built-in speaker and microphone—no external hardware required.

## How It Works

AirSwipe emits an ultrasonic tone (~18.5 kHz) through your laptop speaker while simultaneously recording with the microphone. When your hand moves, the reflected sound experiences a Doppler shift:

- **Hand moving left** → frequency shifts up (approaching)
- **Hand moving right** → frequency shifts down (receding)

By analyzing the STFT (Short-Time Fourier Transform) of the recorded audio, we detect these shifts and classify gestures in real-time.

## Features

- **Part 1 (MVP):** Robust left/right swipe classification with "none" class
- **Part 2 (Upgrade):** Continuous motion tracking with velocity and position visualization
- Automatic carrier frequency selection for your hardware
- Background noise subtraction
- Configurable sensitivity and thresholds

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AirSwipe.git
cd AirSwipe

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Find the Best Carrier Frequency

First, scan to find the optimal ultrasonic frequency for your Mac's hardware:

```bash
python main.py scan
```

Keep your hands away during the scan. The tool will test frequencies from 16-19.5 kHz and recommend the best one.

### 2. Visualize the Signal

See the spectrogram and Doppler features in real-time:

```bash
python main.py visualize
```

Wave your hand in front of the laptop to see:
- Spectrogram showing energy distribution around the carrier
- D(t) - signed Doppler proxy (positive = left, negative = right)
- A(t) - activity level

### 3. Detect Gestures

Run gesture detection:

```bash
python main.py detect --freq 18500
```

Swipe your hand left or right about 30-50cm in front of the laptop.

### 4. Motion Tracking (Part 2)

See continuous motion tracking visualization:

```bash
python main.py track --tracking
```

This shows:
- Real-time velocity `v̂(t)` strip chart
- Position `x̂(t)` tracking dot

## Commands

| Command | Description |
|---------|-------------|
| `python main.py scan` | Find best carrier frequency |
| `python main.py visualize` | Real-time spectrogram |
| `python main.py visualize --tracking` | With motion tracking plots |
| `python main.py detect` | Gesture detection mode |
| `python main.py track` | Motion tracking demo |
| `python main.py track --kalman` | Kalman filter tracking |
| `python main.py collect` | Collect training data |
| `python main.py train --dataset data/dataset.json` | Train classifier |

## Configuration

Key parameters (adjustable via CLI):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--freq` | 18500 | Carrier frequency (Hz) |
| `--threshold` | 0.5 | Activity detection threshold |
| `--amplitude` | 0.3 | Tone amplitude (0-1) |
| `--cooldown` | 0.5 | Seconds between gestures |
| `--smoothing` | 0.85 | EMA smoothing factor |

## Project Structure

```
AirSwipe/
├── airswipe/
│   ├── __init__.py
│   ├── config.py        # Configuration dataclass
│   ├── audio_tx.py      # Tone generation
│   ├── audio_rx.py      # Microphone capture
│   ├── dsp.py           # Signal processing & features
│   ├── segmentation.py  # Gesture event detection
│   ├── model.py         # Classifiers (logistic + CNN)
│   ├── tracker.py       # Motion tracking (Part 2)
│   ├── ui.py            # Visualization
│   └── dataset.py       # Data collection & training
├── main.py              # CLI entry point
├── requirements.txt
└── README.md
```

## Tips for Best Results

1. **Environment:** Works best in quiet rooms with hard surfaces
2. **Distance:** 20-80cm from laptop is ideal
3. **Speed:** Medium-speed swipes work best initially
4. **Calibration:** Run `scan` first to find optimal frequency
5. **Background:** Keep hands away during baseline calibration

## Troubleshooting

**No carrier peak visible:**
- Try different frequencies with `--freq`
- Run `scan` to find working frequency

**Too many false positives:**
- Increase `--threshold`
- Ensure proper baseline calibration (keep still for first 2 seconds)

**Gestures not detected:**
- Decrease `--threshold`
- Swipe closer to the laptop
- Try faster/slower swipe speeds

## Technical Details

- **Sample Rate:** 48 kHz
- **FFT Size:** 2048 (≈23.4 Hz frequency resolution)
- **Hop Size:** 512 (≈10.7 ms time resolution)
- **Carrier Range:** 16-19.5 kHz (ultrasonic, mostly inaudible)
- **Doppler Bandwidth:** ±500 Hz around carrier

## LicenseMIT License - see [LICENSE](LICENSE)
