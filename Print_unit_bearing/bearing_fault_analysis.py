# bearing_fault_analysis.py

"""
A complete Python script to:
1. Load vibration data from CSV (single column, no timestamp)
2. Compute FFT
3. Identify amplitudes at bearing fault frequencies (FIP, FEP, FRP)
4. Visualize fault zones clearly
5. Fully documented for educational and production use
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# --------------------------- CONFIGURATION ---------------------------

FS = 50000  # Sampling frequency in Hz (50 kHz)
RPM = 400  # Shaft speed
SHAFT_FREQ = RPM / 60  # Hz

FREQ_FACTORS = {
    "FIP": 14.271,   # Inner race fault
    "FEP": 11.729,   # Outer race fault
    "FRP": 10.133    # Rolling element fault
}

WINDOW_PERCENT = 2.0  # Fault zone window tolerance in percent
CSV_PATH = "vibration_data.csv"  # Input CSV path

# --------------------------- LOAD RAW CSV ---------------------------

print("Loading data from CSV...")
df = pd.read_csv(CSV_PATH, header=None, names=['accel'])
accel = df['accel'].values
N = len(accel)
print(f"Loaded {N} samples at {FS} Hz → {N / FS:.2f} seconds of data.")

# --------------------------- FFT COMPUTATION ---------------------------

print("Computing FFT...")
yf = np.abs(fft(accel)[:N // 2])  # FFT magnitude (one-sided)
xf = fftfreq(N, 1 / FS)[:N // 2]  # Frequency axis (Hz)

# --------------------------- FAULT BAND ANALYSIS ---------------------------

def extract_fault_band_energy(xf, yf, center_freq, tol_percent):
    tol = center_freq * (tol_percent / 100)
    lower, upper = center_freq - tol, center_freq + tol
    band_mask = (xf >= lower) & (xf <= upper)
    if not np.any(band_mask):
        return None, None
    peak_idx = np.argmax(yf[band_mask])
    peak_freq = xf[band_mask][peak_idx]
    peak_amp = yf[band_mask][peak_idx]
    return peak_freq, peak_amp

fault_results = {}

for fault, factor in FREQ_FACTORS.items():
    fault_freq = factor * SHAFT_FREQ
    peak_freq, peak_amp = extract_fault_band_energy(xf, yf, fault_freq, WINDOW_PERCENT)
    fault_results[fault] = {
        'center': fault_freq,
        'peak_freq': peak_freq,
        'peak_amp': peak_amp
    }
    print(f"{fault}: Expected @ {fault_freq:.2f} Hz → Peak @ {peak_freq:.2f} Hz, Amplitude = {peak_amp:.2f}")

# --------------------------- VISUALIZATION ---------------------------

plt.figure(figsize=(16, 8))
plt.plot(xf, yf, label="Vibration Spectrum", color='steelblue', linewidth=1.2)

colors = {'FIP': 'crimson', 'FEP': 'seagreen', 'FRP': 'darkorange'}

# Highlight zones and peaks
for fault, result in fault_results.items():
    center = result['center']
    tol = center * (WINDOW_PERCENT / 100)
    peak_amp = result['peak_amp']
    peak_freq = result['peak_freq']

    # Shaded zone
    plt.axvspan(center - tol, center + tol, color=colors[fault], alpha=0.25, label=f"{fault} zone")
    
    # Peak line
    plt.axvline(peak_freq, color=colors[fault], linestyle='--', linewidth=1.5)
    plt.text(peak_freq + 1, peak_amp, f"{fault} peak\n{peak_freq:.2f} Hz\n{peak_amp:.2f}", 
             color=colors[fault], fontsize=10, va='bottom', ha='left', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[fault]))

plt.title("Vibration Spectrum Print Unit ", fontsize=16, fontweight='bold')
plt.xlabel("Frequency (Hz)", fontsize=14)
plt.ylabel("Amplitude (|FFT|)", fontsize=14)
plt.xlim(0, 1000)  # Focus on low-frequency faults
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
