# AirSwipe Architecture Overview

> Complete technical documentation of how the code works.

---

## System Overview

AirSwipe uses **ultrasonic Doppler sensing** to detect hand gestures:
1. Emit ~20 kHz tone from laptop speaker
2. Record reflections from microphone
3. Detect frequency shifts when hand moves (Doppler effect)
4. Classify as LEFT / RIGHT / NONE

---

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│                         config.py                                   │
│                    (all parameters)                                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ used by all modules
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  audio_tx.py │    │  audio_rx.py │    │   dsp.py     │
│  (speaker)   │    │    (mic)     │    │  (FFT, etc)  │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       └─────────┬─────────┘                   │
                 ▼                             │
         ┌──────────────┐                      │
         │ AudioDuplex  │◄─────────────────────┘
         │ (TX + RX)    │
         └──────┬───────┘
                │ audio data
        ┌───────┴────────┐
        ▼                ▼
┌──────────────┐  ┌──────────────┐
│    ui.py     │  │segmentation.py│
│ (visualize)  │  │(detect events)│
└──────────────┘  └──────┬───────┘
                         │ events
                         ▼
                  ┌──────────────┐
                  │  model.py    │
                  │ (classify)   │
                  └──────────────┘
```

---

## File-by-File Breakdown

### config.py — Central Configuration

**Purpose:** Single source of truth for all tunable parameters.

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 48000 Hz | Audio sample rate |
| `carrier_freq` | 19000 Hz | Ultrasonic carrier frequency |
| `tone_amplitude` | 0.15 | Speaker volume (0-1) |
| `fft_size` | 2048 | FFT window size |
| `hop_size` | 512 | Samples between FFT frames |
| `doppler_bandwidth` | 500 Hz | Analysis band around carrier |
| `activity_threshold` | 0.5 | Minimum activity to trigger |

**Why Python dataclass?**
- Computed properties (e.g., `freq_resolution = sample_rate / fft_size`)
- Type hints and IDE support
- Default values with validation

---

### audio_tx.py — Tone Generation

**Purpose:** Generate and play continuous ultrasonic sine wave.

**Key Class:** `AudioTx`

**How it works:**
```python
# Every ~10ms, sounddevice calls _output_callback
def _output_callback(self, outdata, frames, time_info, status):
    # Generate sine wave samples
    t = np.arange(frames) / sample_rate
    outdata[:, 0] = amplitude * np.sin(2π * freq * t + phase)
    
    # Update phase for continuity (no clicks between blocks)
    self._phase = (self._phase + phase_increment) % (2π)
```

**Key concept — Phase continuity:**
```
Without phase tracking: ∿∿∿∿│∿∿∿∿│∿∿∿∿  ← Clicks at boundaries
With phase tracking:    ∿∿∿∿∿∿∿∿∿∿∿∿∿∿  ← Smooth continuous wave
```

---

### audio_rx.py — Microphone Recording

**Purpose:** Capture audio from microphone into circular buffer.

**Key Classes:**
- `AudioRx` — Input only
- `AudioDuplex` — Combined TX + RX (recommended)

**Circular Buffer:**
```python
# deque with max length auto-discards old samples
self._buffer = deque(maxlen=240000)  # 5 seconds at 48kHz

# When new samples arrive:
self._buffer.extend(new_samples)  # Old samples auto-removed
```

**Callback System:**
```python
# Register a function to be called when new audio arrives
def set_frame_callback(self, callback):
    self._frame_callback = callback

# In _input_callback:
if self._frame_callback:
    self._frame_callback(samples)  # Notify listener
```

**AudioDuplex — Why use it?**
- Single callback handles both TX and RX
- Tighter synchronization
- Cleaner code

---

### dsp.py — Signal Processing

**Purpose:** FFT computation, feature extraction, background subtraction.

**Key Class:** `DSP`

**Main Operations:**

#### 1. Compute Spectrum (Single FFT)
```python
def compute_spectrum(self, audio):
    segment = audio[-2048:]           # Last 2048 samples
    windowed = segment * hann_window  # Apply window
    spectrum = np.abs(rfft(windowed)) # FFT → magnitudes
    return spectrum                   # 1025 frequency bins
```

#### 2. Background Subtraction (Noise Removal)
```python
# During idle, collect spectra and compute median
self._baseline = np.median(collected_spectra)

# For each new spectrum:
cleaned = max(0, spectrum - baseline)  # Remove static noise
```

**Analogy:** Like zeroing a kitchen scale before measuring flour.

#### 3. Feature Extraction
```python
def extract_features(self, audio):
    spectrum = self.compute_spectrum(audio)
    spectrum = self.subtract_baseline(spectrum)
    
    # Sum energy in bands around carrier
    e_below = sum(spectrum[18000-18450 Hz]²)  # Hand receding
    e_above = sum(spectrum[18550-19000 Hz]²)  # Hand approaching
    
    d = e_above - e_below  # Direction indicator
    a = e_above + e_below  # Activity level
    
    return DopplerFeatures(e_above, e_below, d, a)
```

**Visual:**
```
Spectrum around carrier:

Power
  │    [e_below]    █    [e_above]
  │    ████████    ███    ████████
  └────────────────┴───────────────→ Frequency
        18000    18500    19000

d > 0 → approaching → LEFT swipe
d < 0 → receding → RIGHT swipe
```

**Key Class:** `CarrierScanner`
- Tests multiple frequencies
- Measures SNR at each
- Picks best for hardware

---

### segmentation.py — Gesture Detection

**Purpose:** Detect when gestures start/end from continuous feature stream.

**Key Class:** `GestureSegmenter`

**State Machine:**
```
            A > threshold
     IDLE ─────────────────→ ONSET
       ↑                        │
       │ A dropped              │ duration > 0.15s
       │ too fast               ▼
       └──── A < threshold ─ ACTIVE
                                │
                          A < threshold
                                │
                                ▼
                         _classify_event()
                                │
                                ▼
                        LEFT / RIGHT / NONE
```

**Why ONSET state?**
- Debounces false triggers
- Requires activity to sustain for 0.15s before committing
- Filters noise spikes

**Classification (Baseline):**
```python
def _classify_event(self):
    d_values = [f.d for f in collected_features]
    signed_area = sum(d_values)
    
    if signed_area > threshold:
        return LEFT
    elif signed_area < -threshold:
        return RIGHT
    else:
        return NONE
```

**Callback Pattern:**
```python
segmenter.set_gesture_callback(my_handler)
# my_handler(event) called when gesture detected
```

---

### model.py — Classification Models

**Purpose:** Classify gestures (simple rules or neural network).

**Two Options:**

#### 1. BaselineClassifier (No training needed)
```python
# Hand-crafted features → threshold → label
if signed_area > threshold:
    return LEFT
```

#### 2. TinyCNN (Requires training)
```python
# Spectrogram patch → Conv layers → Softmax
Conv2D(16) → ReLU → Pool
Conv2D(32) → ReLU → Pool
Conv2D(64) → ReLU
GlobalAvgPool → Linear(3)  # [none, left, right]
```

---

### ui.py — Visualization

**Purpose:** Real-time spectrogram and feature display.

**Key Class:** `MatplotlibUI`

**Update Loop (every ~100ms):**
```python
def _update_plot(self, frame):
    # 1. Get audio from duplex
    audio = self.audio.get_buffer(3.0)
    
    # 2. Compute spectrogram via DSP
    times, freqs, Sxx = self.dsp.compute_spectrogram(audio)
    
    # 3. Extract features via DSP
    features = self.dsp.extract_features(audio)
    
    # 4. Update plots
    self._ax_spec.imshow(Sxx)
    self._ax_features.plot(d_history)
```

---

### dataset.py — Training Data Collection

**Purpose:** Record labeled samples for CNN training.

**Key Classes:**
- `Sample` — One recorded gesture (audio + label + metadata)
- `DatasetCollector` — Interactive recording session
- `DatasetProcessor` — Extract features, create splits

**Collection Flow:**
```bash
python main.py collect
# Interactive: press 'l' for left, 'r' for right, 'n' for none
# Saves to data/dataset_XXXXX.json
```

---

### diagnostic.py — Hardware Testing

**Purpose:** Verify audio hardware works before use.

**Tests:**
1. Import sounddevice
2. List audio devices
3. Check sample rate support
4. Test tone output
5. Test mic recording
6. Measure SNR in duplex mode

---

## Data Flow Pipelines

### Real-Time Detection Pipeline
```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  AudioDuplex ──→ DSP.extract_features() ──→ Segmenter.update()     │
│       │                   │                        │                │
│    (audio)          (d, a features)          (state machine)        │
│                                                    │                │
│                                              Gesture detected?      │
│                                                    │                │
│                                             YES    │    NO          │
│                                              ↓     │     ↓          │
│                                        _classify   │   continue     │
│                                              ↓     │                │
│                                        LEFT/RIGHT  │                │
│                                              ↓     │                │
│                                        callback()  │                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Visualization Pipeline
```
AudioDuplex ──→ DSP.compute_spectrogram() ──→ MatplotlibUI
     │                    │
     └─→ DSP.extract_features() ──→ UI (D(t), A(t) plots)
```

### Training Pipeline
```
DatasetCollector ──→ JSON file ──→ DatasetProcessor ──→ TinyCNN.fit()
    (record)           (save)        (extract)           (train)
```

---

## Key Concepts

### Doppler Effect
```
Hand moving TOWARD mic:  reflected freq > transmitted freq (shift UP)
Hand moving AWAY:        reflected freq < transmitted freq (shift DOWN)

Shift amount: Δf = (2 × velocity × carrier_freq) / speed_of_sound
At 1 m/s, 18500 Hz: Δf ≈ 108 Hz
```

### FFT Parameters Trade-off
```
Larger FFT size:  Better frequency resolution, worse time resolution
Smaller FFT size: Better time resolution, worse frequency resolution

Current: 2048 samples → 23 Hz resolution, 43ms time blur
```

### STFT (Short-Time Fourier Transform)
```
Sliding window of FFTs:
- Window size = fft_size (2048)
- Hop = hop_size (512)
- Overlap = 75%
- Each column of spectrogram = one FFT
```

### Background Subtraction
```
Baseline = median spectrum during idle
Clean spectrum = max(0, spectrum - baseline)
Result: Only Doppler changes visible, static noise removed
```

---

## Threading Model

```
Main Thread                    Audio Thread (C, from sounddevice)
────────────                   ─────────────────────────────────
                               
UI updates                     _duplex_callback() runs every ~10ms
segmenter.update()                ├── Generate TX samples
get_buffer()                      ├── Store RX samples in buffer
                                  └── Call frame_callback if registered
        │                                    │
        └─────── SHARED: self._buffer ───────┘
                 (protected by self._lock)
```

**Why locks?**
- Audio callback runs in separate thread
- Main thread reads buffer
- Lock prevents race conditions

---

## CLI Commands

| Command | What it does |
|---------|--------------|
| `python main.py scan` | Find best carrier frequency |
| `python main.py visualize` | Show live spectrogram |
| `python main.py detect` | Run gesture detection |
| `python main.py collect` | Record training data |
| `python main.py train` | Train CNN model |
| `python -m airswipe.diagnostic` | Test hardware |

---

## Quick Reference

### Important Frequencies
```
Sample rate:      48000 Hz
Carrier:          18500-20000 Hz (adjust per hardware)
Doppler range:    ±500 Hz around carrier
Nyquist limit:    24000 Hz (max capturable)
```

### Important Timings
```
FFT window:       2048 samples = 42.7 ms
Hop:              512 samples = 10.7 ms
Min gesture:      150 ms
Max gesture:      1000 ms
Cooldown:         500 ms
UI update:        100 ms
```

### Feature Interpretation
```
d > 0:  Energy above carrier → approaching → LEFT
d < 0:  Energy below carrier → receding → RIGHT
a high: Motion detected
a low:  Idle
```
