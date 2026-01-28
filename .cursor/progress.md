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
