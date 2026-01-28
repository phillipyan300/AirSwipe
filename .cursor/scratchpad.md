# Scratchpad

> Task tracking with status markers:
> - `[ ]` To-do
> - `[X]` Done  
> - `[!]` Blocked
> - `[?]` Question

---

## Current Sprint: Day 1 Validation

### Environment Setup
- [X] Create virtual environment
- [X] Install dependencies
- [X] Verify all modules import correctly
- [X] Run diagnostic tool (SNR=52.6dB ✅)

### Core Functionality
- [X] Test visualize mode - can see hand motion in spectrogram
- [ ] Run carrier scan to find optimal frequency
- [ ] Test gesture detection (`python main.py detect`)
- [ ] Tune activity threshold for real-world use

### Data Collection & Training
- [ ] Collect dataset (target: 100+ samples per class)
- [ ] Train baseline classifier
- [ ] Evaluate accuracy
- [ ] Train CNN if baseline insufficient

---

## Future Improvements

### Performance
- [ ] Add pyqtgraph for faster UI updates
- [ ] Profile DSP pipeline for bottlenecks
- [ ] Consider SIMD optimizations if needed

### Features
- [ ] Implement USB mic fallback for debugging
- [ ] Add gesture actions (desktop switching, media control)
- [ ] Add configuration file loading/saving
- [ ] Add proper logging framework (replace print statements)

### Polish
- [ ] Add unit tests for DSP functions
- [ ] Add integration tests for full pipeline
- [ ] Improve error messages for common failures

---

## Questions

- [?] Should carrier scan run automatically on first launch?
- [?] Is 18.5 kHz the best default, or should we default lower (17 kHz) for better speaker response?
- [?] Should we persist the best carrier frequency to a config file?

---

## Dependencies

```
Core functionality:
  sounddevice → audio I/O
  numpy → array operations
  scipy → signal processing (STFT, filters)

Visualization:
  matplotlib → spectrograms, plots

Optional (CNN training):
  torch → neural network
```

---

## Key Commands

```bash
# Activate environment
source venv/bin/activate

# Run diagnostics
python -m airswipe.diagnostic

# Find best carrier frequency
python main.py scan

# Live spectrogram
python main.py visualize
python main.py visualize --tracking  # with motion plots

# Gesture detection
python main.py detect

# Motion tracking demo
python main.py track
python main.py track --kalman

# Collect training data
python main.py collect

# Train classifier
python main.py train --dataset data/dataset_XXXXXX.json
```

---
