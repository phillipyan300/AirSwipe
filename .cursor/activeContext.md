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

## Current State (2026-01-30)

### What's Working
- ✅ All modules implemented and importing correctly
- ✅ Diagnostic passed: SNR = 52.6 dB at 18500 Hz
- ✅ Visualize mode shows Doppler shifts when waving hand
- ✅ Audio I/O stable after tuning (blocksize=2048, amplitude=0.15)
- ✅ Auto data collection mode added (with countdown timers)
- ✅ Modular model system: Logistic, CNN1D, CNN2D (swappable)
- ✅ Confidence gating implemented (reject low-confidence predictions)
- ✅ Activity cap implemented (reject noise spikes)
- ✅ D variance gate implemented (reject flat noise)
- ✅ Combined visualize+detect mode working

### Current Issue: CNN Always Predicts LEFT
- CNN1D trained on binary (LEFT/RIGHT) data
- Model predicts LEFT with 99.9% confidence even on noise
- **Root cause:** D values always positive (asymmetric noise floor), low variance
- **Filters added:** D variance check should reject these

### What's Next
- Tune `min_d_variance` threshold (currently 500, may need higher)
- Verify filters reject ambient noise while passing real gestures
- Consider retraining with more varied data
- May need to investigate why D is always positive at rest

---

## ML Architecture Understanding

### Available Models (Swappable)

| Model | Input | Use Case |
|-------|-------|----------|
| `logistic` | 10 hand-crafted features | Fast, interpretable, needs good features |
| `cnn1d` | Raw D(t) time series | Learns patterns automatically |
| `cnn2d` | Full spectrogram | Most complex, probably overkill |

### Current Approach: Binary + Confidence Gating
- Train only on LEFT and RIGHT (clear gestures)
- Reject predictions with confidence < 60% as NONE
- Avoids the impossible task of defining "nothing"

### Segmentation Pipeline
```
Audio → DSP → Features (D, A) → Segmenter → [Gates] → Classifier → [Confidence Gate] → Output
                                    ↓
                           Activity: 200-5000
                           D variance: > 500
```

### Feature Engineering (for Logistic)
See "Feature Analysis" section below for relevance assessment.

---

## Feature Analysis

| Feature | Formula | Relevance | Notes |
|---------|---------|-----------|-------|
| `peak_asymmetry` | (max_d - \|min_d\|) / (max_d + \|min_d\|) | ⭐⭐⭐ CRITICAL | Which peak is bigger? Best single feature for L/R. |
| `slope_d` | linear fit of D(t) | ⭐⭐⭐ HIGH | Captures temporal trend |
| `peak_order` | (t_max - t_min) / len | ⭐⭐ HIGH | Which peak came first? |
| `max_d` | max(D centered) | ⭐⭐ MEDIUM | Peak positive deviation |
| `min_d` | min(D centered) | ⭐⭐ MEDIUM | Peak negative deviation |
| `max_a` | max(A(t)) | ⭐⭐ MEDIUM | Confirms gesture vs noise |
| `std_d` | std(D centered) | ⭐ LOW | Motion magnitude, not direction |
| `mean_d` | mean(D centered) | ⭐ LOW | ~0 after centering |
| `signed_area` | Σ D centered | ⭐ LOW | ~0 after centering |
| `duration` | seconds | ⭐ LOW | Doesn't help L/R |

### Key Insight: Microphone Location Matters!
- D > 0 means hand moving TOWARD mic (Doppler compression)
- D < 0 means hand moving AWAY from mic (Doppler expansion)
- The actual sign mapping depends on mic location and swipe direction
- **Data showed LEFT has positive peak_asymmetry, RIGHT has negative** (opposite of physics expectation)
- This is fine - model learns whatever pattern exists in the data
- **Each user must train on their own hardware**

### Linear Classifier Limitation (Solved)
- Linear classifiers can't learn "is max_d > |min_d|?"
- **Solution:** `peak_asymmetry` pre-computes this nonlinear comparison
- This allows logistic regression to separate classes that raw features couldn't

---

## Key Files

| File | Purpose |
|------|---------|
| `NEWDESIGNDOC.md` | Full specification document |
| `main.py` | CLI entry point with train/detect/visualize |
| `airswipe/config.py` | All tunable parameters |
| `airswipe/dsp.py` | Signal processing, feature extraction |
| `airswipe/model.py` | BaselineClassifier (logistic reg) |
| `airswipe/models/` | Modular model system |
| `airswipe/models/base.py` | BaseModel abstract class |
| `airswipe/models/logistic.py` | LogisticClassifier |
| `airswipe/models/cnn1d.py` | CNN1DClassifier (PyTorch) |
| `airswipe/models/cnn2d.py` | CNN2DClassifier (PyTorch) |
| `airswipe/dataset.py` | Data collection with auto mode |
| `airswipe/segmentation.py` | GestureSegmenter with activity gates |

---

## Configuration Snapshot

```python
# Audio
sample_rate = 48000      # Hz
carrier_freq = 18500     # Hz (ultrasonic)
tone_amplitude = 0.15    # Reduced for less distortion

# FFT
fft_size = 2048          # ~23.4 Hz resolution
hop_size = 512           # ~10.7 ms time resolution

# Segmentation Gates
activity_threshold = 200.0   # Min activity to trigger
activity_cap = 5000.0        # Max activity (above = noise spike)
min_d_variance = 500.0       # Min D variance (below = flat noise)

# Classification
confidence_threshold = 0.6   # Below this → reject as NONE
```

---

## Open Questions

1. ~~Should carrier scan run automatically on first launch?~~ (not urgent)
2. ~~Which features are actually discriminative?~~ → `peak_asymmetry` is the key feature
3. ~~Should we add `peak_order` feature?~~ → Yes, implemented
4. ~~Does live detection work with the new features?~~ → Partially, needs tuning
5. **Why is D always positive at rest?** — Asymmetric noise floor above/below carrier
6. **What's the right min_d_variance threshold?** — Currently 500, may need 2000+
7. **Should we add D mean check?** — If mean D is always same sign, no direction change occurred

---
