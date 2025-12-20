# AirSwipe

**Gesture input on macOS using acoustic sensing.**

AirSwipe explores whether a MacBook can sense hand swipes in the air using nothing but inaudible audio and signal processing.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## How It Works

1. **Emit**: The MacBook speaker plays a near-ultrasonic tone (18–19.5 kHz)
2. **Reflect**: Hand motion causes Doppler shifts in the reflected signal
3. **Detect**: The microphone captures these frequency shifts
4. **Act**: Motion patterns trigger system actions (swipe gestures)

No external hardware. No phone app. Just physics.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/airswipe.git
cd airswipe

# Install dependencies
pip install -r requirements.txt

# Run feasibility test (Phase 0)
python airswipe.py --visualize

# Run with gesture detection (Phase 1+)
python airswipe.py --detect
```

## Requirements

- macOS (tested on M1+ MacBooks)
- Python 3.9+
- Built-in speaker and microphone

## Supported Gestures

| Gesture | Action |
|---------|--------|
| Swipe Right | Next desktop/app |
| Swipe Left | Previous desktop/app |
| Swipe Toward | Mission Control |
| Swipe Away | Lock screen |

## Project Phases

- [x] **Phase 0**: Feasibility — Generate tone, record mic, visualize spectrogram
- [ ] **Phase 1**: Motion Detection — Compute motion scores, detect bursts
- [ ] **Phase 2**: Gesture Classification — Segment windows, classify directions
- [ ] **Phase 3**: System Integration — Map gestures to macOS actions

## Technical Details

### Signal Processing Pipeline

```
Speaker (18.5 kHz) → Air → Hand Motion → Doppler Shift → Microphone
                                              ↓
                              STFT → Feature Extraction → Classification
```

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Carrier Frequency | 18.5 kHz | Near-ultrasonic, mostly inaudible |
| Sample Rate | 48 kHz | Standard audio rate |
| Window Size | 256-512 samples | ~5-10ms for good time resolution |
| FFT Size | 2048 | Good frequency resolution |

## Configuration

Edit `config.py` or pass command-line arguments:

```bash
python airswipe.py --freq 18500 --visualize --threshold 2.0
```

## Troubleshooting

**No audio output?**
- Check System Preferences → Sound → Output
- Ensure volume is not muted

**Microphone not working?**
- Grant microphone permissions: System Preferences → Privacy → Microphone
- Check `python` or `Terminal` has microphone access

**High false positive rate?**
- Increase detection threshold
- Ensure quiet environment
- Check for ultrasonic interference (some electronics emit high frequencies)

## Contributing

This is an experimental research project. Contributions welcome!

1. Fork the repo
2. Create a feature branch
3. Submit a PR

## License

MIT License — See [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by research in acoustic sensing (FingerIO, SoundWave, etc.) and the desire to make gesture input accessible without specialized hardware.

---

*AirSwipe is a research demo, not a production system. Use responsibly.*


