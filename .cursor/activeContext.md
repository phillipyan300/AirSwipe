# Active Context

> Recent state, current focus, and working memory for the session.

---

## Project Summary

**AirSwipe** is a Mac-only gesture recognition system using ultrasonic Doppler sensing:
- Emit ~18.5 kHz tone from laptop speaker
- Record reflections from built-in microphone  
- Detect Doppler shifts when hand moves (toward = higher freq, away = lower freq)
- Classify left/right swipes + continuous motion tracking

---

## Current State

### What's Working
- ✅ All modules implemented and importing correctly
- ✅ Diagnostic passed: SNR = 52.6 dB at 18500 Hz
- ✅ Visualize mode shows Doppler shifts when waving hand
- ✅ Audio I/O stable after tuning (blocksize=2048, amplitude=0.15)

### What's Next
- Test carrier scan to find optimal frequency
- Test gesture detection (`python main.py detect`)
- Tune thresholds based on real usage
- Collect training data if heuristic classifier needs improvement

---

## Key Files

| File | Purpose |
|------|---------|
| `NEWDESIGNDOC.md` | Full specification document |
| `main.py` | CLI entry point |
| `airswipe/config.py` | All tunable parameters |
| `airswipe/dsp.py` | Signal processing, feature extraction |
| `airswipe/segmentation.py` | Gesture detection state machine |

---

## Recent Changes (2026-01-27)

1. Created `.cursor/` folder with structured logging:
   - `progress.md` - Task log
   - `lessons-learned.md` - Insights and pitfalls
   - `scratchpad.md` - Task tracking
   - `activeContext.md` - This file

2. Created `.cursorrules` at project root with logging rules

3. Migrated content from temporary `cursor_log/` folder

---

## Configuration Snapshot

```python
sample_rate = 48000      # Hz
carrier_freq = 18500     # Hz (ultrasonic)
fft_size = 2048          # ~23.4 Hz resolution
hop_size = 512           # ~10.7 ms time resolution
tone_amplitude = 0.15    # Reduced for less distortion
activity_threshold = 0.5 # Event detection
```

---

## Open Questions

1. Should carrier scan run automatically on first launch?
2. Best default carrier frequency for Mac speakers?
3. How to persist optimal carrier after scanning?

---
