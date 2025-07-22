import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# ===== 1. Load CSV 
file_path = "vibration_data.csv"  # Change this if using another path
df = pd.read_csv(file_path, header=None)
df.columns = ["Acceleration_m_s2"]

# ===== 2. Extract Signal =====
accel = df["Acceleration_m_s2"].values
FS = 50000  # Sampling rate in Hz
N = len(accel)
T = 1 / FS
time = np.linspace(0, N*T, N, endpoint=False)

# ===== 3. Plot Time-Domain Waveform =====
plt.figure(figsize=(14, 5))
plt.plot(time[:5000], accel[:5000], color='orange', linewidth=0.8)
plt.title("Time-Domain Vibration Signal (First 0.1 sec)")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== 4. FFT =====
yf = np.abs(fft(accel)[:N // 2]) * (2 / N)  # Normalize
xf = fftfreq(N, T)[:N // 2]

# ===== 5. Fault Frequencies for 416 RPM =====
RPM = 416
fault_freqs = {
    "Inner Race": 14.271 * RPM / 60,
    "Outer Race": 11.729 * RPM / 60,
    "Rolling Element": 10.133 * RPM / 60,
}

# ===== 6. Plot FFT with Fault Frequencies =====
plt.figure(figsize=(14, 6))
plt.plot(xf, yf, label="FFT Amplitude", color='blue', linewidth=0.7)

for label, freq in fault_freqs.items():
    plt.axvline(x=freq, color='red', linestyle='--', label=f"{label}: {freq:.2f} Hz")

plt.xlim(0, 800)  
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (m/s²)")
plt.title("Frequency Spectrum with Bearing Fault Overlays (RPM = 416)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ===== 7. Zoomed-In Plot with Fault Frequency Windows and Peak Annotations =====
from scipy.signal import find_peaks
plt.figure(figsize=(14, 6))
plt.plot(xf, yf, color='blue', linewidth=0.7, label="FFT Amplitude")

for label, freq in fault_freqs.items():
    # ±2% window
    f_low = freq * 0.98
    f_high = freq * 1.02

    # Highlight the fault band
    plt.axvspan(f_low, f_high, color='red', alpha=0.2)
    plt.axvline(freq, color='red', linestyle='--', linewidth=1.2)

    # Find index range in xf corresponding to the band
    idx_range = np.where((xf >= f_low) & (xf <= f_high))[0]
    if len(idx_range) == 0:
        continue

    # Local FFT data
    local_x = xf[idx_range]
    local_y = yf[idx_range]

    # Find peaks in that band
    peaks, _ = find_peaks(local_y)
    if peaks.size > 0:
        peak_idx = peaks[np.argmax(local_y[peaks])]
        peak_freq = local_x[peak_idx]
        peak_amp = local_y[peak_idx]

        # Plot black dot and label on the peak
        plt.plot(peak_freq, peak_amp, "ko")
        plt.text(peak_freq + 0.5, peak_amp, f"{label}\n{peak_freq:.2f} Hz\n{peak_amp:.3f}", color='black')

plt.xlim(0, 120)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (m/s²)")
plt.title("Zoomed-In FFT with Fault Bands and Peak Annotations")
plt.grid(True)
plt.tight_layout()
plt.show()

from scipy.signal import find_peaks

# ===== 8. Plot Frequency Band 2000–4000 Hz with Peaks > 0.01 m/s² =====
# Find indices for the 2000–4000 Hz range
from scipy.signal import find_peaks
idx_band = np.where((xf >= 2000) & (xf <= 4000))[0]
band_x = xf[idx_band]
band_y = yf[idx_band]

# Find peaks above 0.01 amplitude
peaks, _ = find_peaks(band_y, height=0.01)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(band_x, band_y, color='purple', linewidth=0.7, label="FFT Amplitude (2000–4000 Hz)")
plt.scatter(band_x[peaks], band_y[peaks], color='black', zorder=5, label="Peaks > 0.01")

# Annotate peaks
for i in peaks:
    freq = band_x[i]
    amp = band_y[i]
    plt.text(freq + 5, amp, f"{freq:.1f} Hz\n{amp:.3f}", color='black', fontsize=8)

plt.title("Step 8: Peaks in 2000–4000 Hz Range (> 0.01 m/s²)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (m/s²)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


from collections import defaultdict

# ===== 9. Detect and Group Harmonics in 2000–4000 Hz Band =====

# Threshold for spacing match
spacing_tolerance = 2.0

# Get peak frequencies from Step 8
peak_freqs = band_x[peaks]
peak_freqs = np.sort(peak_freqs)

# Group harmonics by checking spacing patterns
groups = []
used = set()

for i, base_freq in enumerate(peak_freqs):
    if base_freq in used:
        continue

    group = [base_freq]
    last = base_freq
    for test_freq in peak_freqs[i+1:]:
        spacing = test_freq - last
        if spacing < 10:  # ignore very close noise peaks
            continue

        # Check if it's close to the original spacing
        expected_spacing = group[1] - group[0] if len(group) > 1 else spacing
        if abs(spacing - expected_spacing) <= spacing_tolerance:
            group.append(test_freq)
            last = test_freq

    if len(group) >= 3:  # Only keep groups with at least 3 harmonics
        groups.append(group)
        used.update(group)

# ===== Plot Harmonic Groups =====
plt.figure(figsize=(14, 6))
plt.plot(band_x, band_y, color='gray', linewidth=0.6, label="FFT Amplitude")

colors = ['red', 'green', 'blue', 'orange', 'magenta', 'cyan', 'purple']

for i, group in enumerate(groups):
    color = colors[i % len(colors)]
    amps = [band_y[np.where(band_x == f)[0][0]] for f in group]
    plt.scatter(group, amps, color=color, s=50, label=f"Harmonic Group {i+1}")
    for f, a in zip(group, amps):
        plt.text(f + 5, a, f"{f:.1f} Hz", fontsize=8, color=color)

plt.title("Step 9: Harmonic Groups Detected in 2000–4000 Hz")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (m/s²)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


import csv

# ===== 10. Export Harmonic Groups and Detect Sidebands =====

# 1. Export Harmonic Groups
harmonic_csv = "harmonic_groups.csv"
with open(harmonic_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Group", "Frequency_Hz", "Amplitude_m_s2"])
    
    for i, group in enumerate(groups):
        for freq in group:
            idx = np.where(band_x == freq)[0]
            if idx.size == 0:
                continue
            amp = band_y[idx[0]]
            writer.writerow([f"Group {i+1}", round(freq, 2), round(amp, 4)])

# 2. Sideband Detection Function
def detect_sidebands(freqs, tolerance=5):
    """
    Simple sideband detector.
    Looks for frequency spacing around a carrier that repeats.
    Returns list of sidebands if spacing is consistent.
    """
    freqs = np.sort(freqs)
    spacing_list = np.diff(freqs)
    if len(spacing_list) < 2:
        return []

    # Round spacings to see if they're nearly equal
    rounded_spacings = np.round(spacing_list, 1)
    most_common_spacing = max(set(rounded_spacings), key=list(rounded_spacings).count)

    # Build sideband list from most common spacing
    sidebands = [freqs[0]]
    for i in range(1, len(freqs)):
        if abs(freqs[i] - sidebands[-1] - most_common_spacing) <= tolerance:
            sidebands.append(freqs[i])

    return sidebands if len(sidebands) >= 3 else []

# 3. Detect and Save Sidebands per Group
sideband_csv = "sidebands.csv"
with open(sideband_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Group", "Sideband_Frequency_Hz", "Amplitude_m_s2"])
    
    for i, group in enumerate(groups):
        sb = detect_sidebands(group)
        if sb:
            for freq in sb:
                idx = np.where(band_x == freq)[0]
                if idx.size == 0:
                    continue
                amp = band_y[idx[0]]
                writer.writerow([f"Group {i+1}", round(freq, 2), round(amp, 4)])



# ===== 11. Plot Detected Sidebands per Group =====

# Load sidebands CSV
sideband_df = pd.read_csv("sidebands.csv")

# Group by harmonic group
sideband_groups = sideband_df.groupby("Group")

# Plot
plt.figure(figsize=(14, 6))
plt.plot(band_x, band_y, color='lightgray', linewidth=0.7, label="FFT Amplitude")

colors = ['red', 'green', 'blue', 'orange', 'magenta', 'cyan', 'purple']

for i, (group_name, group_data) in enumerate(sideband_groups):
    color = colors[i % len(colors)]
    freqs = group_data["Sideband_Frequency_Hz"].values
    amps = group_data["Amplitude_m_s2"].values

    plt.scatter(freqs, amps, color=color, s=60, label=f"{group_name} Sidebands", zorder=5)

    for freq, amp in zip(freqs, amps):
        plt.text(freq + 5, amp, f"{freq:.1f} Hz", fontsize=8, color=color)

plt.title("Step 11: Detected Sidebands in 2000–4000 Hz")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (m/s²)")
plt.grid(True)
plt.xlim(2000, 4000)
plt.legend()
plt.tight_layout()
plt.show()



# ===== 12. Overlay Harmonics, Sidebands, Carrier & AM Pattern Detection =====

# Reload harmonic and sideband CSVs
harmonic_df = pd.read_csv("harmonic_groups.csv")
sideband_df = pd.read_csv("sidebands.csv")

# Group by harmonic set
harmonic_groups = harmonic_df.groupby("Group")
sideband_groups = sideband_df.groupby("Group")

# Plot base FFT
plt.figure(figsize=(14, 6))
plt.plot(band_x, band_y, color='lightgray', linewidth=0.6, label="FFT Amplitude")

colors = ['red', 'green', 'blue', 'orange', 'magenta', 'cyan', 'purple']

for i, (group_name, group_data) in enumerate(harmonic_groups):
    color = colors[i % len(colors)]
    
    # Harmonics
    h_freqs = group_data["Frequency_Hz"].values
    h_amps = group_data["Amplitude_m_s2"].values
    plt.scatter(h_freqs, h_amps, color=color, s=50, label=f"{group_name} Harmonics")

    # Overlay sidebands (if any)
    if group_name in sideband_groups.groups:
        sb_data = sideband_groups.get_group(group_name)
        sb_freqs = sb_data["Sideband_Frequency_Hz"].values
        sb_amps = sb_data["Amplitude_m_s2"].values

        plt.scatter(sb_freqs, sb_amps, marker='x', s=70, color=color, label=f"{group_name} Sidebands")

        # Estimate carrier frequency (midpoint)
        carrier_freq = np.mean(sb_freqs)
        plt.axvline(carrier_freq, color=color, linestyle='--', alpha=0.5)
        plt.text(carrier_freq + 3, max(sb_amps), f"Carrier ≈ {carrier_freq:.1f} Hz", color=color, fontsize=9)

        # AM pattern: check symmetric sidebands
        spacings = np.diff(np.sort(sb_freqs))
        spacing_mean = np.mean(spacings)
        if 2 <= spacing_mean <= 150:
            plt.annotate("AM Pattern",
                         xy=(carrier_freq, max(sb_amps)*1.05),
                         xytext=(carrier_freq + 20, max(sb_amps)*1.1),
                         arrowprops=dict(facecolor=color, shrink=0.05, width=2, headwidth=8),
                         fontsize=10, color=color)

plt.title("Step 12: Full Modulation Analysis with Harmonics, Sidebands & AM")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (m/s²)")
plt.xlim(2000, 4000)
plt.grid(True)
plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.show()


from scipy.signal import spectrogram

# ===== 13. Spectrogram for Visualizing Time-Frequency Energy =====
# Horizontal lines = constant frequency (e.g., harmonic tones)
# Wiggling lines = modulated faults
# Bursts of energy = impacts, rubs, looseness
# Repeated bands = sidebands = AM

# Use a time slice (optional) if dataset is very large
# accel = accel[:100000]  # first 2 sec at 50kHz

# Parameters
window_length = 2048  # Higher = better freq resolution, slower time
noverlap = 1024
nfft = 4096  # Zero-padding to improve frequency resolution
fmin, fmax = 0, 4500

# Compute spectrogram
f, t, Sxx = spectrogram(accel, fs=FS, window='hann', nperseg=window_length,
                        noverlap=noverlap, nfft=nfft, scaling='spectrum', mode='magnitude')

# Trim frequency range for zoom
freq_mask = (f >= fmin) & (f <= fmax)
f = f[freq_mask]
Sxx = Sxx[freq_mask, :]

# Plot
plt.figure(figsize=(14, 6))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno')  # dB scale
plt.colorbar(label='Amplitude (dB)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Step 13: Spectrogram (0–4500 Hz)')
plt.ylim(fmin, fmax)
plt.tight_layout()
plt.show()


from scipy.stats import kurtosis

# ===== 14. Moving RMS & Kurtosis Detector =====

# Parameters
window_size = 2048  # about 0.041s at 50kHz
step_size = 512     # overlap amount
num_windows = (len(accel) - window_size) // step_size

rms_vals = []
kurt_vals = []
time_vals = []

# Slide window across signal
for i in range(num_windows):
    start = i * step_size
    end = start + window_size
    segment = accel[start:end]
    
    rms = np.sqrt(np.mean(segment**2))
    krt = kurtosis(segment, fisher=False)  # use "True" for excess kurtosis

    rms_vals.append(rms)
    kurt_vals.append(krt)
    time_vals.append(start / FS)  # convert to seconds

# Plot
fig, ax1 = plt.subplots(figsize=(14, 6))

color_rms = 'tab:blue'
color_kurt = 'tab:red'

ax1.set_title("Step 14: Moving RMS and Kurtosis Over Time")
ax1.set_xlabel("Time (s)")

ax1.plot(time_vals, rms_vals, color=color_rms, label="RMS", linewidth=1.2)
ax1.set_ylabel("RMS (m/s²)", color=color_rms)
ax1.tick_params(axis='y', labelcolor=color_rms)

ax2 = ax1.twinx()
ax2.plot(time_vals, kurt_vals, color=color_kurt, label="Kurtosis", linewidth=1.2)
ax2.set_ylabel("Kurtosis", color=color_kurt)
ax2.tick_params(axis='y', labelcolor=color_kurt)

fig.tight_layout()
plt.grid(True)
plt.show()


# ===== 15. Spectrogram with RMS and Kurtosis Overlay =====
# RMS rising steadily	             --->     Looseness, imbalance, or alignment issue
# Kurtosis spikes randomly         --->	  Impacts — could be bearing or gear tooth crack
# Both spike together	             --->     Sudden transient — check for machine events
# Spikes in spectrogram AND RMS	 --->     Sustained fault energy

# Reuse the spectrogram data from Step 13
# f, t, Sxx already calculated

# Interpolate RMS and Kurtosis to match spectrogram time bins
from scipy.interpolate import interp1d

# Interpolators
rms_interp = interp1d(time_vals, rms_vals, bounds_error=False, fill_value="extrapolate")
kurt_interp = interp1d(time_vals, kurt_vals, bounds_error=False, fill_value="extrapolate")

# Resampled values to match spectrogram timeline
rms_on_spec = rms_interp(t)
kurt_on_spec = kurt_interp(t)

# Normalize for plotting
rms_norm = (rms_on_spec - np.min(rms_on_spec)) / (np.max(rms_on_spec) - np.min(rms_on_spec))
kurt_norm = (kurt_on_spec - np.min(kurt_on_spec)) / (np.max(kurt_on_spec) - np.min(kurt_on_spec))

# Scale to fit on top of spectrogram (pick frequency range)
overlay_min = 4200
overlay_max = 4500

rms_freq = overlay_min + (overlay_max - overlay_min) * rms_norm
kurt_freq = overlay_min + (overlay_max - overlay_min) * kurt_norm

# Plot spectrogram
plt.figure(figsize=(14, 6))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno')
plt.colorbar(label='Amplitude (dB)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title("Step 15: Spectrogram with RMS & Kurtosis Overlaid")
plt.ylim(0, 4500)

# Overlay RMS and Kurtosis
plt.plot(t, rms_freq, label="RMS", color='cyan', linewidth=1.5)
plt.plot(t, kurt_freq, label="Kurtosis", color='lime', linewidth=1.5)

# Labels
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


