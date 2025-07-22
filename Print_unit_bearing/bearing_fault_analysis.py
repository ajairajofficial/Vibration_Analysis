# bearing_fault_analysis.py

"""
##########################################################################################################
############ A Product developed by: Ajai Raj for stuff Print unit bearing fault analysis ################
##########################################################################################################
Objective:
Finding the fault frequencies (peak Amplitude) of the bearing in the vibration data.
##########################################################################################################
This analysis is for Print Unit 1 Print Couple 3. The hardware setup includes the VSE903 data acquisition system
paired with a VSA001 vibration sensor. Data was accessed and captured via VES004 IFM software, 
with the raw data exported in CSV format for further analysis.

The shaft speed was determined to be 416 RPM, calculated using a separate digital pulse signal. 
The sampling frequency during data acquisition was 50 kHz.

##########################################################################################################

1. Load vibration data from CSV 
2. Compute FFT
3. Identify amplitudes at bearing fault frequencies (FIP, FEP, FRP) 
4. Visualize fault zones clearly

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# --------------------------- CONFIGURATION ---------------------------

FS = 50000  # Sampling frequency in Hz (50 kHz) (how many data points per second)
RPM = 416  # Shaft speed
SHAFT_FREQ = RPM / 60  # Hz (how many revolutions per second)

FREQ_FACTORS = {
    "FIP": 14.271,   # Inner race fault
    "FEP": 11.729,   # Outer race fault
    "FRP": 10.133    # Rolling element fault
}
# Dictionary mapping fault types to their frequency multiplication factors (used to calculate expected fault frequencies).

WINDOW_PERCENT = 2.0  # Fault zone window tolerance in percent (how much to the left and right of the fault frequency to look for the peak)
CSV_PATH = "vibration_data.csv"  # Input CSV path

SHOW_IN_G = True  # Set to True to display/export/plot in g, False for m/s²
G_CONV = 9.81

# --------------------------- LOAD RAW CSV ---------------------------

print("Loading data from CSV...")
df = pd.read_csv(CSV_PATH, header=None, names=['accel']) # load the single column accelaration data into a data frame
accel = df['accel'].values # convert the data frame to a numpy array
N = len(accel) # number of data points
print(f"Loaded {N} samples at {FS} Hz → {N / FS:.2f} seconds of data.")

# --------------------------- FFT COMPUTATION ---------------------------

print("Computing FFT...")
# Normalized FFT (amplitude spectrum)
yf = (2.0 / N) * np.abs(fft(accel)[:N // 2])  # m/s²
if SHOW_IN_G:
    yf = yf / G_CONV  # Convert to g
    unit_label = 'g'
else:
    unit_label = 'm/s²'
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
    print(f"{fault}: Expected @ {fault_freq:.2f} Hz → Peak @ {peak_freq:.2f} Hz, Amplitude = {peak_amp:.5f} {unit_label}")

# --------------------------- HIGH-FREQ PEAK SEARCH ---------------------------

# Find peaks in the 2000-5000 Hz range with amplitude > 1000 (in current unit)
freq_min = 2000
freq_max = 5000
amp_thresh = .01 / G_CONV if SHOW_IN_G else 1000  # Adjust threshold if in g

# Mask for the frequency range
range_mask = (xf >= freq_min) & (xf <= freq_max)
xf_range = xf[range_mask]
yf_range = yf[range_mask]

# Find peaks above threshold
peaks, properties = find_peaks(yf_range, height=amp_thresh)

# Export peaks to Excel if any found
if len(peaks) > 0:
    peak_freqs = xf_range[peaks]
    peak_amps = yf_range[peaks]
    peaks_df = pd.DataFrame({'Frequency (Hz)': peak_freqs, f'Amplitude ({unit_label})': peak_amps})
    excel_path = f'high_freq_peaks_{unit_label.replace("/", "_")}.xlsx'
    peaks_df.to_excel(excel_path, index=False)
    print(f"\nExported {len(peaks)} peaks above {amp_thresh:.5f} {unit_label} between {freq_min} Hz and {freq_max} Hz to {excel_path}.")
else:
    print(f"\nNo peaks above {amp_thresh:.5f} {unit_label} found between {freq_min} Hz and {freq_max} Hz.")

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
    plt.text(peak_freq + 1, peak_amp, f"{fault} peak\n{peak_freq:.2f} Hz\n{peak_amp:.5f} {unit_label}", 
             color=colors[fault], fontsize=10, va='bottom', ha='left', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[fault]))

plt.title("Vibration Spectrum Print Unit ", fontsize=16, fontweight='bold')
plt.xlabel("Frequency (Hz)", fontsize=14)
plt.ylabel(f"Amplitude (|FFT|) [{unit_label}]", fontsize=14)
plt.xlim(0, 1000)  # Focus on low-frequency faults
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
