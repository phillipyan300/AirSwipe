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
