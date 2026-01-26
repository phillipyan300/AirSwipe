#!/usr/bin/env python3
"""
AirSwipe Diagnostic Tool

Quick checks to verify audio hardware and basic functionality.
Run this first to make sure everything works.
"""

import sys
import time
import numpy as np

def check_imports():
    """Check all required imports."""
    print("=" * 50)
    print("1. CHECKING IMPORTS")
    print("=" * 50)
    
    checks = []
    
    try:
        import sounddevice as sd
        checks.append(("sounddevice", "✅"))
    except ImportError as e:
        checks.append(("sounddevice", f"❌ {e}"))
    
    try:
        import numpy as np
        checks.append(("numpy", "✅"))
    except ImportError as e:
        checks.append(("numpy", f"❌ {e}"))
    
    try:
        import scipy
        checks.append(("scipy", "✅"))
    except ImportError as e:
        checks.append(("scipy", f"❌ {e}"))
    
    try:
        import matplotlib
        checks.append(("matplotlib", "✅"))
    except ImportError as e:
        checks.append(("matplotlib", f"❌ {e}"))
    
    for name, status in checks:
        print(f"  {name}: {status}")
    
    return all("✅" in s for _, s in checks)


def check_audio_devices():
    """List available audio devices."""
    print("\n" + "=" * 50)
    print("2. AUDIO DEVICES")
    print("=" * 50)
    
    import sounddevice as sd
    
    print("\nDefault devices:")
    defaults = sd.default.device
    print(f"  Input:  {defaults[0]}")
    print(f"  Output: {defaults[1]}")
    
    print("\nAll devices:")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        marker = ""
        if i == defaults[0]:
            marker += " [DEFAULT INPUT]"
        if i == defaults[1]:
            marker += " [DEFAULT OUTPUT]"
        
        channels = f"in={d['max_input_channels']}, out={d['max_output_channels']}"
        print(f"  [{i}] {d['name'][:40]:<40} ({channels}){marker}")
    
    return True


def check_sample_rates():
    """Check supported sample rates."""
    print("\n" + "=" * 50)
    print("3. SAMPLE RATE SUPPORT")
    print("=" * 50)
    
    import sounddevice as sd
    
    rates_to_test = [44100, 48000, 96000]
    
    for rate in rates_to_test:
        try:
            sd.check_input_settings(samplerate=rate)
            sd.check_output_settings(samplerate=rate)
            print(f"  {rate} Hz: ✅ Supported")
        except Exception as e:
            print(f"  {rate} Hz: ❌ {e}")
    
    return True


def test_tone_output(duration=1.0, freq=18500, amplitude=0.3):
    """Test playing a tone."""
    print("\n" + "=" * 50)
    print("4. TONE OUTPUT TEST")
    print("=" * 50)
    
    import sounddevice as sd
    
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    tone = amplitude * np.sin(2 * np.pi * freq * t)
    
    print(f"  Playing {freq} Hz tone for {duration}s...")
    print(f"  (This is ultrasonic - you may not hear it clearly)")
    
    try:
        sd.play(tone, samplerate=sample_rate)
        sd.wait()
        print("  ✅ Tone played successfully")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_microphone(duration=1.0):
    """Test recording from microphone."""
    print("\n" + "=" * 50)
    print("5. MICROPHONE TEST")
    print("=" * 50)
    
    import sounddevice as sd
    
    sample_rate = 48000
    
    print(f"  Recording for {duration}s...")
    
    try:
        recording = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype=np.float32)
        sd.wait()
        
        # Analyze recording
        rms = np.sqrt(np.mean(recording ** 2))
        peak = np.max(np.abs(recording))
        
        print(f"  ✅ Recorded {len(recording)} samples")
        print(f"  RMS level: {rms:.6f}")
        print(f"  Peak level: {peak:.6f}")
        
        if rms < 0.0001:
            print("  ⚠️  Very low signal - check microphone permissions!")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_duplex(duration=1.0, freq=18500):
    """Test simultaneous play and record."""
    print("\n" + "=" * 50)
    print("6. DUPLEX TEST (Play + Record)")
    print("=" * 50)
    
    import sounddevice as sd
    
    sample_rate = 48000
    
    print(f"  Playing {freq} Hz while recording for {duration}s...")
    
    # Generate tone
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * freq * t)
    
    try:
        recording = sd.playrec(tone, samplerate=sample_rate, 
                               channels=1, dtype=np.float32)
        sd.wait()
        
        # Check for carrier in recording
        from scipy.fft import rfft, rfftfreq
        
        spectrum = np.abs(rfft(recording[:, 0]))
        freqs = rfftfreq(len(recording), 1/sample_rate)
        
        # Find peak near carrier
        carrier_idx = np.argmin(np.abs(freqs - freq))
        peak_region = slice(max(0, carrier_idx-10), carrier_idx+10)
        peak_power = np.max(spectrum[peak_region])
        noise_floor = np.median(spectrum[1000:2000])  # Mid-frequency noise
        
        snr = 20 * np.log10(peak_power / (noise_floor + 1e-10))
        
        print(f"  ✅ Duplex working")
        print(f"  Carrier detected at ~{freqs[carrier_idx]:.0f} Hz")
        print(f"  SNR: {snr:.1f} dB")
        
        if snr > 20:
            print("  ✅ Good signal - ready for gesture detection!")
        elif snr > 10:
            print("  ⚠️  Weak signal - may work with tuning")
        else:
            print("  ❌ Poor signal - try different carrier frequency")
        
        return True, snr
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False, 0


def run_all_diagnostics():
    """Run all diagnostic tests."""
    print("\n" + "🔊 " * 20)
    print("   AIRSWIPE DIAGNOSTIC")
    print("🔊 " * 20)
    
    results = []
    
    # 1. Imports
    results.append(("Imports", check_imports()))
    
    # 2. Audio devices
    results.append(("Audio Devices", check_audio_devices()))
    
    # 3. Sample rates
    results.append(("Sample Rates", check_sample_rates()))
    
    # 4. Tone output
    results.append(("Tone Output", test_tone_output(duration=0.5)))
    
    # 5. Microphone
    results.append(("Microphone", test_microphone(duration=0.5)))
    
    # 6. Duplex
    duplex_ok, snr = test_duplex(duration=1.0)
    results.append(("Duplex", duplex_ok))
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    all_ok = True
    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False
    
    print("\n" + "-" * 50)
    if all_ok:
        print("🎉 All tests passed! Ready for gesture detection.")
        print("\nNext steps:")
        print("  1. python main.py scan        # Find best carrier")
        print("  2. python main.py visualize   # See spectrogram")
        print("  3. python main.py detect      # Detect gestures")
    else:
        print("⚠️  Some tests failed. Check errors above.")
    
    return all_ok


if __name__ == "__main__":
    run_all_diagnostics()
