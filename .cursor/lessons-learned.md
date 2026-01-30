# Lessons Learned

> Append new insights, pitfalls, and decisions here.

---

## Audio I/O on macOS

### Blocksize and Latency
- **Problem:** Small blocksize (512) causes input overflow errors on Mac
- **Solution:** Use `blocksize=2048` and `latency='high'` for stability over speed
- **Tradeoff:** Adds ~40ms latency but prevents dropouts

### Tone Amplitude
- **Problem:** High amplitude (0.3) causes audible crackling/distortion
- **Solution:** Keep amplitude at 0.15 or lower
- **Note:** Ultrasonic frequencies are less efficient on laptop speakers

### Sample Rate
- **Verified:** Mac M1 supports 48kHz sample rate reliably
- **SNR achieved:** 52.6 dB at 18500 Hz carrier (excellent for Doppler detection)

---

## Matplotlib Real-time Visualization

### Spectrogram Display
- **Problem:** `pcolormesh` doesn't handle frequency axis correctly when band-filtered
- **Solution:** Use `imshow` with explicit `extent` parameter to set correct axis labels
- **Example:**
  ```python
  ax.imshow(data, aspect='auto', origin='lower',
            extent=[t_min, t_max, f_min, f_max])
  ```

### Update Rate
- **Problem:** 50ms update interval causes UI lag
- **Solution:** Use 100ms interval for stability
- **Alternative:** Consider pyqtgraph for faster updates if needed

### Legend Handling
- **Problem:** Adding legend when no data causes matplotlib warning
- **Solution:** Only call `ax.legend()` after confirming data exists

---

## Python Audio with sounddevice

### Duplex Mode
- Using `sd.Stream` for simultaneous play/record is more synchronized than separate streams
- Phase continuity in tone generation is important for clean audio

### Microphone Permissions
- macOS requires explicit microphone permission for Terminal/IDE
- First run will trigger permission dialog

### Device Query
- Always verify devices with `sd.query_devices()` before assuming defaults work

---

## Project Structure

### Modular Design Benefits
- Separating audio_tx, audio_rx, dsp, etc. makes testing individual components easier
- Config dataclass with properties provides derived values (freq resolution, bin indices)

### CLI with argparse
- Subcommands (scan, visualize, detect, track, collect, train) organize functionality
- Each mode has clear entry point in main.py

---

## Machine Learning Fundamentals

### Feature Scaling is Critical
- **Problem:** Training stuck at 42% accuracy, loss ~14 (should be ~1)
- **Cause:** Features had wildly different scales:
  - `signed_area`: -50,000 to +50,000
  - `duration`: 0.3 to 0.8
- **Solution:** Z-score normalization: `(x - mean) / std`
- **Key insight:** Always normalize features before gradient descent. Large features dominate gradients.

### Logistic Regression = 1-Layer Neural Network
- Formula: `P(class) = softmax(W @ features + bias)`
- No hidden layers, decision boundary is linear (hyperplane in feature space)
- Very simple, interpretable, works well when you have good features
- If it fails, either: (1) features need scaling, (2) features aren't discriminative, or (3) problem needs non-linear boundary

### Why "None" Class is Essential
- Without "none", model forced to choose left/right for ANY input
- Sneezing, typing, random noise all trigger false positives
- "None" teaches the model what idle/background looks like

### Model Complexity for Doppler Data
- Doppler gestures are fundamentally simple (1D amplitude change over time)
- Left vs Right: just the sign of the integral of D(t)
- 2D CNN on spectrogram is overkill for this problem
- Logistic regression or 1D CNN is the sweet spot

---

## Feature Engineering for Doppler Gestures

### Most Important Features
1. **signed_area** (Σ D(t)): Integral of Doppler proxy. Sign indicates direction relative to microphone.
2. **slope_d**: Captures temporal structure — which direction came first.
3. **mean_d**: Correlated with signed_area but still useful.

### Less Useful Features
- `duration` — Does NOT help distinguish left from right. A fast left swipe and slow left swipe are both left. Only helps separate "none" (too short) from gesture (long enough).
- `std_d` — Indicates motion occurred but not direction.

### Features to Consider Adding
- **peak_order**: Did max_d occur before or after min_d? Directly encodes swipe direction.
- **first_half_area vs second_half_area**: Temporal asymmetry

---

## CRITICAL: Microphone Location Affects Label Mapping

### The Physics
- Doppler shift depends on motion **relative to microphone position**
- Hand moving **toward** mic = frequency compression = energy **above** carrier = D > 0
- Hand moving **away** from mic = frequency expansion = energy **below** carrier = D < 0

### Hardware-Dependent Mapping
- On MacBooks, the microphone is typically on the **left side** (near the keyboard)
- Swiping LEFT (→) means hand first approaches mic, then recedes
  - D(t): positive then negative
  - signed_area: positive
- Swiping RIGHT (←) means hand first recedes, then approaches
  - D(t): negative then positive  
  - signed_area: negative

### Implication: Each User Must Train Their Own Model
- The mapping "positive D = left swipe" depends on mic location
- Different laptops may have mics on different sides
- A model trained on one laptop may have inverted labels on another
- **Each person should collect their own training data on their specific hardware**

### Alternative Approach
- Could make the system hardware-agnostic by using **relative** features (slope, peak_order) rather than absolute sign
- Or: detect mic location during calibration and adjust mapping

---

### Data Collection Tips
- Vary distance (20cm, 50cm, 80cm)
- Vary speed (slow, medium, fast)
- **Train on YOUR specific laptop — mic position matters**

### Critical: None Class Must Have LOWER Activity Than Gestures
- **Bad "none"**: Random vigorous movements → higher activity than swipes
- **Good "none"**: Sitting still, typing, mouse movements (lateral, not toward/away)
- Swipes should have the HIGHEST peaks because they involve deliberate toward/away motion
- If none has higher activity than gestures, no threshold can separate them

### Carrier Frequency Must Match Between Training and Inference
- Config default and argparse default must match
- A carrier mismatch means wrong frequency bins are analyzed
- Always verify carrier frequency in output matches what you trained with

### Peak vs Total Activity
- Model correctly uses `max_a` (peak activity per sample)
- Segmenter uses instantaneous A(t) for triggering
- Peak activity better distinguishes sharp gestures from accumulated fidgeting
- A sharp swipe has a high peak; fidgeting has many small values

---

## Linear Classifiers and Nonlinear Relationships

### Linear Classifiers Can't Learn Comparisons
- **Problem:** Logistic regression learns `w1*feature1 + w2*feature2 > threshold`
- **Limitation:** It CANNOT learn "is feature1 > feature2?" or "|feature1| vs |feature2|"
- **Example:** For left/right, the intuition is:
  - LEFT: negative peak (|min_d|) is bigger than positive peak (max_d)
  - RIGHT: positive peak (max_d) is bigger than negative peak (|min_d|)
- A linear classifier cannot express `max_d > |min_d|` with just raw max_d and min_d features

### Solution: Engineer Nonlinear Features
- **peak_asymmetry** = `(max_d - |min_d|) / (max_d + |min_d|)`
  - Positive when positive peak dominates → one direction
  - Negative when negative peak dominates → other direction
  - Near zero when balanced → idle
- This pre-computes the nonlinear comparison so the linear model can use it directly
- **Key insight:** If you suspect a nonlinear relationship matters, explicitly encode it as a feature

### Per-Sample Centering
- Subtract mean D from each gesture window before computing features
- **Why:** Removes constant offset (ambient bias), forces model to focus on the *shape* of D(t)
- **Effect:** After centering, `mean_d ≈ 0` and `signed_area ≈ 0` for all samples
- Features like `slope_d`, `max_d`, `min_d` become relative to the gesture's own mean

### Data Can Be Internally Consistent Even If Flipped
- Analysis showed LEFT always had positive peak_asymmetry (opposite of physics prediction)
- This is fine! The model learns whatever pattern exists
- What matters is consistency: all left swipes should have similar patterns
- Physics intuition helps design features, but data determines the actual mapping

---

## Binary Classification vs 3-Class (NONE Problem)

### The "None" Class is Problematic
- **Issue:** "Nothing" can look like infinite different patterns:
  - True silence
  - Typing, mouse movements
  - Random noise spikes, crackles
  - Partial/weak gestures
  - Any movement that isn't a deliberate swipe
- **Impossible to capture** all variations in training data
- Model learns a *specific* pattern of "none" that won't generalize

### Better Approach: Binary + Confidence Gating
- Train only on LEFT and RIGHT (clear, deliberate gestures)
- Use model's **uncertainty** as a natural "I don't know" signal
- If confidence < threshold (e.g., 60%), reject as NONE
- **Why this works:**
  - Model only learns what gestures look like
  - Anything that doesn't match either → low confidence → rejected
  - Handles novel noise patterns model has never seen

### Implementation
```python
if pred.confidence < config.confidence_threshold:
    final_label = GestureLabel.NONE  # Low confidence = reject
```

---

## Segmentation Quality Gates

### Problem: False Positives from Noise
- Segmenter triggers on `activity > threshold`
- But noise spikes and ambient fluctuations also exceed threshold
- Model forced to classify garbage → false detections

### Solution: Multi-Stage Filtering

| Gate | Parameter | Purpose |
|------|-----------|---------|
| 1. Activity min | `activity_threshold = 200` | Filter out idle/quiet |
| 2. Activity max | `activity_cap = 5000` | Filter out noise spikes (crackles fill all frequencies) |
| 3. D variance | `min_d_variance = 500` | Filter out flat noise (no Doppler shift) |
| 4. Confidence | `confidence_threshold = 0.6` | Filter out ambiguous classifications |

### Understanding Activity (A) Values
- `A = e_above + e_below` (sum of squared FFT magnitudes in Doppler bands)
- **Units:** Arbitrary energy measure (squared amplitude)
- **Typical ranges:**
  - ~50-150: Ambient noise, idle
  - ~200-2000: Real hand gesture
  - ~5000+: Loud noise spike (crackle/pop)

### Why D Variance Matters
- Real gestures: D swings from positive to negative (or big swings in one direction)
- Noise/idle: D is flat-ish, constant offset
- After CNN normalization, flat noise can accidentally match learned patterns
- **Fix:** Require `var(D) > threshold` to confirm actual Doppler shift occurred

---

## CNN Models for Doppler Data

### 1D CNN on D(t) Time Series
- **Input:** Raw D(t) values, padded/truncated to fixed length
- **Architecture:** Conv1D → MaxPool → Conv1D → MaxPool → Dense → Softmax
- **Preprocessing:** Zero-mean, unit-variance normalization per sample
- **Benefit:** Learns temporal patterns automatically, translation invariant

### Gotcha: Normalization Makes Flat Noise Look Like Patterns
- When D is flat (constant), after normalization: `(D - mean) / std` → amplified noise
- The shape of this noise might accidentally match a learned class
- **Solution:** Add variance gate before classification

### Model Architecture Choices
| Model | Good For | Complexity |
|-------|----------|------------|
| Logistic Regression | Well-engineered features | Lowest |
| 1D CNN | Raw D(t) time series | Medium |
| 2D CNN | Full spectrogram | Highest (overkill) |

---
