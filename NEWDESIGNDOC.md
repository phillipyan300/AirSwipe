# Engineering Doc: Mac Ultrasonic Doppler Gestures (Swipe MVP + Trajectory Upgrade)

## 0) TLDR

Build a Mac-only gesture system using the laptop speaker to emit an ultrasonic carrier and the microphone to record reflections. Use STFT features to detect Doppler motion.

Two deliverables:

1. **Part 1 (MVP):** robust **left vs right swipe** classifier (plus “none”).
2. **Part 2 (Upgrade):** **continuous motion tracking signal** (a smooth direction/velocity trace, optionally a 1D trajectory proxy) that looks impressive in a demo.

This doc is written so a Cursor agent can implement end to end.

---

## 1) Goals and Non-goals

### Goals

* Works on a 2021 Mac (M1 era) using built-in speaker and mic.
* Real-time detection with low latency, target under 150 ms end to end.
* Part 1: classify gestures {left, right, none} reliably in a dorm room.
* Part 2: show a “fancier” real-time movement visualization (direction and velocity over time, optionally an approximate lateral track proxy) that looks good on video.

### Non-goals (for v1)

* Finger-level tracking or centimeter-accurate 2D localization.
* Multi-user separation.
* Guaranteed reliability in extremely loud environments without tuning.
* Full distance/angle estimation with chirps and mic arrays.

---

## 2) System Overview

### Hardware

* Built-in MacBook speakers
* Built-in MacBook microphone
* No external devices required

### Core idea

* Emit a continuous ultrasonic tone at carrier frequency `cf` in roughly 17–19.5 kHz.
* When a hand moves, the reflected tone experiences small Doppler shifts (tens to a few hundred Hz).
* Compute an STFT over microphone input, then analyze energy distribution around `cf` over time.

---

## 3) Deliverables

### Part 1: Swipe classifier (MVP)

* Real-time classification of:

  * `left`
  * `right`
  * `none`
* Display:

  * live spectrogram band around `cf`
  * current predicted label and confidence
* Provide a simple “demo action”, for example moving a cursor overlay, switching slides, or printing output.

### Part 2: Fancy motion tracking (Upgrade)

Produce a smooth “motion trace” suitable for demos:

* A continuous signal for each time step:

  * signed Doppler velocity proxy `v_hat(t)` (positive, negative)
  * optionally a smoothed integrated position proxy `x_hat(t) = ∫ v_hat(t) dt`
* A visualization:

  * scrolling strip chart of `v_hat(t)`
  * and/or a moving dot driven by `x_hat(t)` (bounded, auto-centered)

This is not ground-truth physical tracking. It is a convincing motion visualization driven by Doppler features.

---

## 4) Key Risks and Mitigations

### Risk A: Ultrasonic band not supported by Mac speaker/mic chain

Mitigation:

* Implement an automated carrier sweep test to pick `cf` with best SNR.
* Prefer `fs=48kHz`. Avoid `cf` near Nyquist for 44.1 kHz.

### Risk B: OS audio processing (AGC, noise suppression)

Mitigation:

* Use CoreAudio through a stable Python library.
* Measure carrier stability in a “no motion” test.
* If instability is severe, allow an optional USB mic fallback (only for debugging).

### Risk C: Multipath and reflections cause messy spectrograms

Mitigation:

* Background subtraction: subtract a running median/mean spectrogram baseline.
* Classify with a small CNN plus temporal smoothing.

### Risk D: False positives

Mitigation:

* “none” class
* confidence threshold
* minimum event duration
* cooldown between recognized gestures

---

## 5) Architecture

### Modules

1. `audio_tx.py`

   * Generate and play continuous carrier tone at `cf`
   * Manage output stream

2. `audio_rx.py`

   * Capture microphone audio stream at `fs` (48kHz preferred)
   * Produce frames of size `frame_len` with hop `hop_len`

3. `dsp.py`

   * Bandpass filter around `cf ± B` (optional but recommended)
   * Compute STFT or streaming FFT magnitudes
   * Extract features:

     * band energy above carrier
     * band energy below carrier
     * signed energy difference
     * spectral centroid around carrier
   * Background subtraction

4. `segmentation.py`

   * Detect candidate gesture windows
   * Gating rules:

     * energy above threshold
     * minimum duration
     * cooldown time

5. `model.py`

   * Baseline: logistic regression / linear SVM
   * Main: tiny CNN on spectrogram patches
   * Output label and confidence

6. `tracker.py` (Part 2)

   * Convert features into continuous `v_hat(t)`
   * Smooth via EMA or 1D Kalman filter
   * Integrate to get `x_hat(t)` with drift control

7. `ui.py`

   * Live plots:

     * spectrogram around `cf`
     * classification label, confidence
     * motion trace `v_hat(t)` and dot position `x_hat(t)`
   * Hotkeys: start/stop, recalibrate baseline, record dataset

8. `dataset.py`

   * Record raw audio segments with labels and metadata
   * Store STFT patches for training
   * Train/val/test splits

---

## 6) Signal and DSP Design

### Sampling rate

* Target `fs = 48,000 Hz`
* If only `44,100 Hz` works reliably, keep `cf` ≤ ~19 kHz to avoid edge effects.

### Carrier frequency selection

Implement `carrier_scan()`:

1. Record ambient PSD without transmission for 5–10 seconds.
2. For each candidate `cf` in:

   * `16k, 16.5k, 17k, 17.5k, 18k, 18.5k, 19k, 19.5k`
     do:
   * play tone at `cf` for 1–2 seconds
   * record mic and compute PSD around `cf`
   * score = peak power at `cf` minus median power in `cf ± 1kHz` excluding the peak
3. Choose best `cf` by highest score and stability.

Recommendation:

* Try `cf=18.5 kHz` first, then scan if weak.

### STFT defaults

* Window size `N = 2048` at 48kHz
* Hop `H = 256` or `512`
* Window: Hann or Hamming
* Focus band: `cf ± 500 Hz` initially, tune later

### Feature extraction (Part 1)

From each time frame:

* `E_above(t)`: sum of log-magnitude in `(cf, cf+B]`
* `E_below(t)`: sum of log-magnitude in `[cf-B, cf)`
* `D(t) = E_above(t) - E_below(t)` (signed Doppler proxy)
* `A(t) = E_above(t) + E_below(t)` (activity)

Background subtraction:

* Maintain running median `M(f)` of magnitude spectrum in the band during idle.
* Use `S'(t,f)=max(0, S(t,f)-M(f))` before computing energies.

Segmentation:

* Candidate event if `A(t)` exceeds threshold for at least `T_min` frames.
* Extract a fixed-length patch, for example 0.6 seconds centered on event peak.

---

## 7) Modeling

### Baseline model (implement first)

* Logistic regression on summary features over the segment:

  * mean, max, slope of `D(t)`
  * duration above threshold
  * signed area under `D(t)`
* Classes: left, right, none

This gives quick validation and debugging.

### Primary model (recommended)

Tiny CNN on spectrogram patch:

* Input: `log(|STFT|)` band slice around `cf` with shape `(T, F)`

  * Example: `T=64` frames, `F=64` frequency bins
* Architecture:

  * Conv2D(16, 3x3) + ReLU
  * MaxPool
  * Conv2D(32, 3x3) + ReLU
  * GlobalAvgPool
  * Linear to 3 logits

Post-processing:

* run every hop
* smooth predictions with majority vote over last 5–9 frames
* apply confidence threshold
* enforce cooldown after a gesture fires

---

## 8) Part 2 Tracking Design

### Goal

Show a convincing motion trace in real time.

### Approach

Use the signed Doppler proxy `D(t)` as velocity-like signal:

1. Normalize: `v_raw(t) = clamp(D(t) / scale, -1, 1)`
2. Smooth:

   * EMA: `v_hat(t) = α v_hat(t-1) + (1-α) v_raw(t)`
   * start with `α = 0.8` to `0.95`
3. Integrate:

   * `x_hat(t) = x_hat(t-1) + k * v_hat(t)`
4. Drift control:

   * when idle (low activity `A(t)`), slowly decay toward 0:

     * `x_hat(t) = 0.98 * x_hat(t)`
5. Clamp:

   * keep x in [-1, 1] and auto-center if needed

Visualization:

* plot `v_hat(t)` over time
* show dot position `x_hat(t)` on a horizontal bar

Optional improvement:

* Use a 1D Kalman filter for `v_hat` and `x_hat` smoothing if EMA is too jittery.

---

## 9) Dataset Collection

### Labels

* left swipe: hand moves left-to-right in front of laptop
* right swipe: right-to-left
* none: idle, random motion not matching swipe, background noise

### Collection protocol

* At least 100 examples per class for first training.
* Record across:

  * different distances: 20cm, 50cm, 80cm
  * different speeds: slow, medium, fast
  * different environments: quiet room, moderate noise

Store:

* raw audio chunk
* `cf`, `fs`, STFT params
* timestamp, label
* optional live spectrogram patch

Split:

* train 70%, val 15%, test 15%
* keep sessions separated, do not mix segments from the same continuous recording across splits

---

## 10) Evaluation Metrics

### Part 1

* Accuracy, precision/recall per class
* False positive rate during idle
* Gesture latency: time from motion start to classification
* Robustness across distance and speed

Success criteria for MVP:

* Dorm room: >90% accuracy on left/right when a gesture is present
* Idle false positive: fewer than 1 per minute
* End-to-end latency under 150 ms

### Part 2

* Visual stability: low jitter during steady motion
* No drift during idle (returns to neutral)
* Looks good in a screen recording

---

## 11) Implementation Plan

### Milestone 0: Environment and scaffolding

* Choose Python stack:

  * `sounddevice` for audio I/O
  * `numpy/scipy` for DSP
  * `torch` for CNN
  * `matplotlib` or `pyqtgraph` for UI
* Confirm you can simultaneously play tone and record mic.

### Milestone 1: Carrier viability and selection

* Implement ambient PSD scan.
* Implement carrier sweep scoring.
* Pick `cf` automatically.

### Milestone 2: Live spectrogram around carrier

* Streaming STFT
* Display band-limited spectrogram
* Verify Doppler patterns by hand movement.

### Milestone 3: Baseline classifier

* Implement segmentation and logistic regression classifier.
* Add none class.
* Demo in room.

### Milestone 4: Tiny CNN classifier

* Collect dataset
* Train CNN
* Add temporal smoothing and cooldown
* Demo with higher robustness

### Milestone 5: Fancy tracking

* Implement `v_hat(t)` and `x_hat(t)` from features
* Add UI plot and dot visualization
* Record demo video

---

## 12) Notes for Cursor Agent

When implementing, prioritize:

* correct audio streaming without dropouts
* deterministic STFT parameters
* a tight and fast feature pipeline
* instrumentation:

  * logs for `cf`, peak SNR, baseline energy
  * debug plots for `E_above`, `E_below`, `D(t)`, `A(t)`

Common failure patterns:

* no visible carrier peak means wrong `cf` or hardware filtering
* unstable carrier peak often means OS processing or clipping
* lots of false positives means segmentation thresholds are too low or baseline subtraction is missing

---

## 13) Open Questions (acceptable to decide later)

* Should we use continuous tone or repeated short chirps for Part 2?

  * Tone is easier and good enough for a Doppler-based “trajectory proxy”
  * Chirps enable ranging but increase complexity
* If the built-in mic rolloff is severe above 18 kHz, should we drop `cf` to 16–17 kHz?

  * Yes, Doppler still works, it just has slightly smaller shifts

---

If you want, I can also write a “Cursor tasks list” version that is copy-pastable into Cursor as sequential tickets (each ticket with acceptance criteria).
