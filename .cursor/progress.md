# AirSwipe Progress Log

> Append new entries at the bottom. Do not overwrite existing entries.

---

## 2026-01-26 — Initial Implementation Session

### Task
Complete implementation of AirSwipe ultrasonic gesture recognition system from NEWDESIGNDOC.md specification.

### Reason
User requested full modular implementation following the design document.

### Changes Made

| File | Summary |
|------|---------|
| `airswipe/__init__.py` | Package initialization |
| `airswipe/config.py` | All tunable parameters (carrier freq, FFT size, thresholds) |
| `airswipe/audio_tx.py` | AudioTx class - tone generation with phase continuity |
| `airswipe/audio_rx.py` | AudioRx, AudioDuplex classes - mic capture with buffer |
| `airswipe/dsp.py` | DSP class, DopplerFeatures, CarrierScanner |
| `airswipe/segmentation.py` | GestureSegmenter, GestureLabel, GestureEvent |
| `airswipe/model.py` | BaselineClassifier (logistic reg), TinyCNN (PyTorch) |
| `airswipe/tracker.py` | MotionTracker (EMA), KalmanTracker |
| `airswipe/ui.py` | ConsoleUI, MatplotlibUI |
| `airswipe/dataset.py` | DatasetCollector, DatasetProcessor |
| `airswipe/diagnostic.py` | Hardware diagnostic tool |
| `main.py` | CLI entry point with subcommands |
| `requirements.txt` | Dependencies |
| `README.md` | Usage documentation |

### Commands Run
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "from airswipe import Config"  # Import test
python main.py --help  # CLI test
python -m airswipe.diagnostic  # Hardware test → SNR=52.6dB ✅
python main.py visualize  # First UI test
```

### Errors Encountered

1. **Input overflow** - Audio callback couldn't keep up
   - Fix: Increased blocksize from 512 to 2048, added `latency='high'`

2. **Wrong frequency display** - Spectrogram showed ~8500 Hz instead of 18500 Hz
   - Fix: Changed from `pcolormesh` to `imshow` with proper extent

3. **Crackling sound** - Tone too loud
   - Fix: Reduced amplitude from 0.3 to 0.15

4. **Legend warning** - matplotlib warning when no data
   - Fix: Only add legend when data exists

5. **Accidental file deletion** - User deleted config.py
   - Fix: Restored file

---

## 2026-01-27 — Cursor Rules Setup

### Task
Create `.cursor` folder with structured logging system.

### Reason
Pivot from temporary `cursor_log/` folder to standardized `.cursor/` structure with progress.md, lessons-learned.md, scratchpad.md, and activeContext.md.

### Changes Made
- Created `.cursorrules` at project root
- Created `.cursor/progress.md` (this file)
- Created `.cursor/lessons-learned.md`
- Created `.cursor/scratchpad.md`
- Created `.cursor/activeContext.md`
- Migrated content from `cursor_log/session_2026-01-26.md`

### Commands Run
None (file creation only)

### Errors Encountered
None

---

## 2026-01-28 — Auto Collection Mode

### Task
Add automatic guided data collection mode to dataset collector.

### Reason
Manual command entry for each sample was tedious. User requested automatic mode with countdown timers.

### Changes Made

| File | Summary |
|------|---------|
| `airswipe/dataset.py` | Added `auto_collection()` method with countdown timers and interleaved sampling |

### Features Added
- Press 'a' in collection menu to start auto mode
- 3-2-1 countdown before each sample
- Clear instructions: "LEFT SWIPE (hand moves →)", "RIGHT SWIPE (hand moves ←)", "NONE (stay still)"
- Interleaved sampling (left, right, none, left, right, none...) for variety
- Progress updates every 5 samples
- Ctrl+C to stop and return to menu (samples preserved)

### Commands Run
None (code change only)

### Errors Encountered
None

---

## 2026-01-28 — ML Training & Detection Fixes

### Task
Fix training accuracy issues and integrate trained model with live detection.

### Reason
1. Initial training got 42% accuracy due to unscaled features
2. Live detection wasn't using the trained model
3. Activity threshold was too low causing constant false positives

### Changes Made

| File | Summary |
|------|---------|
| `main.py` (cmd_train) | Added z-score feature normalization, feature statistics output, per-class accuracy |
| `main.py` (cmd_detect) | Load trained model from model.pkl, use it for classification |
| `airswipe/model.py` | Store/load normalization params (mean, std), added normalize_features() |
| `airswipe/config.py` | Increased activity_threshold from 0.5 to 50.0 |

### Key Findings

1. **Feature scaling critical**: Raw features ranged from -500,000 to +0.6. Z-score normalization fixed training.

2. **Baseline mismatch**: Training used raw spectrum (all negative D), live detection used baseline-subtracted (all positive D). This confused the heuristic classifier.

3. **Threshold too low**: activity_threshold=0.5 but actual A values were 10-400+, causing constant detections.

4. **Training vs inference gap**: The detect command wasn't loading the trained model at all!

### Accuracy Improvement
- Before normalization: 42.9%
- After normalization: 85.7%
- Per-class: none=100%, left=50% (4 samples), right=100%

### Commands Run
```bash
python main.py train --dataset data/dataset_20260128_164617.json
python main.py detect
```

### Errors Encountered
- Constant LEFT detections during idle (fixed by raising threshold)
- Trained model not being used (fixed by loading model.pkl in cmd_detect)

---

## 2026-01-30 — Nonlinear Feature Engineering

### Task
Debug why model always predicts LEFT during live detection despite 100% training accuracy.

### Reason
User observed constant LEFT predictions for all gestures, indicating training/inference mismatch or model limitations.

### Root Cause Analysis

1. **Per-sample centering side effect**: After centering, `mean_d ≈ 0` and `signed_area ≈ 0` for all classes. These features became useless.

2. **Linear classifier limitation discovered**: 
   - User's intuition: LEFT = negative peak bigger than positive, RIGHT = opposite
   - This is `|min_d| > max_d` vs `max_d > |min_d|` — a NONLINEAR comparison
   - Logistic regression can only learn `w1*max_d + w2*min_d > threshold`, not comparisons

3. **Feature separation analysis**:
   - `slope_d`: Some overlap between classes
   - `max_d`, `min_d`: Different ranges per class but linear model couldn't use effectively
   - Data was internally consistent (just with flipped signs from physics expectation)

### Changes Made

| File | Summary |
|------|---------|
| `airswipe/model.py` | Added `peak_order` feature: `(t_max - t_min) / len` — encodes temporal order |
| `airswipe/model.py` | Added `peak_asymmetry` feature: `(max_d - |min_d|) / (max_d + |min_d|)` |
| `.cursor/lessons-learned.md` | Added section on linear classifiers and nonlinear feature engineering |
| `.cursor/activeContext.md` | Updated feature table, current state, open questions |

### Key Insight

**`peak_asymmetry` is the key discriminative feature:**
- LEFT samples: all positive (0.55 to 0.83)
- RIGHT samples: mostly negative (-0.87 to +0.57)
- NONE samples: in between (-0.08 to +0.41)

This pre-computes the nonlinear "which peak is bigger" comparison, allowing logistic regression to separate classes with a simple linear boundary.

### Training Results
- Features: 10 (added peak_order, peak_asymmetry)
- Test accuracy: 100% (4 none, 5 left, 5 right correct)
- Training accuracy: 93.8% after 1000 epochs

### Commands Run
```bash
python main.py train --dataset data/dataset_20260130_132851.json
```

### Next Steps
- Test live detection: `python main.py detect`
- Verify model distinguishes left/right/none correctly in practice

---

## 2026-01-30 — Modular Models & Noise Filtering

### Task
1. Implement swappable model architecture (Logistic, CNN1D, CNN2D)
2. Debug CNN1D always predicting LEFT
3. Add noise filtering gates to segmentation

### Reason
- User wanted to try CNN models instead of just logistic regression
- Live detection had constant false positives from noise
- CNN was predicting LEFT with 99.9% confidence even on nothing

### Changes Made

| File | Summary |
|------|---------|
| `airswipe/models/` | New folder with modular model system |
| `airswipe/models/base.py` | BaseModel abstract class with save/load |
| `airswipe/models/logistic.py` | LogisticClassifier from BaseModel |
| `airswipe/models/cnn1d.py` | CNN1DClassifier (PyTorch) on D(t) series |
| `airswipe/models/cnn2d.py` | CNN2DClassifier (PyTorch) on spectrograms |
| `airswipe/models/__init__.py` | Factory function `get_model()` |
| `airswipe/config.py` | Added `activity_cap`, `min_d_variance`, `confidence_threshold` |
| `airswipe/segmentation.py` | Added activity cap check to reject noise spikes |
| `airswipe/ui.py` | Added D variance check, confidence gating, debug output |
| `main.py` | `--model-type` flag for train/detect, binary mode |

### Key Findings

1. **Binary classification better than 3-class**: The "NONE" class is impossible to define — "nothing" can look like infinite patterns. Better to use confidence gating.

2. **Multiple noise filtering gates needed**:
   - Activity range: `200 < A < 5000` (filter idle and spikes)
   - D variance: `var(D) > 500` (filter flat noise)
   - Confidence: `> 60%` (filter ambiguous predictions)

3. **CNN always predicting LEFT issue**:
   - D values always positive even at rest (asymmetric noise floor)
   - Low D variance → after CNN normalization, flat noise looks like a pattern
   - Solution: D variance gate rejects before classification

4. **Activity (A) is sum of squared FFT magnitudes**:
   - Not frequency difference, but total energy in Doppler bands
   - ~50-150: idle, ~200-2000: gesture, ~5000+: noise spike

### Commands Run
```bash
python main.py train --model-type cnn1d --binary --dataset data/dataset_20260128.json
python main.py visualize --detect
```

### Current Status
- Filtering gates implemented but need tuning
- `min_d_variance = 500` may be too low, consider 2000+
- Real gestures should have D variance in thousands

---
