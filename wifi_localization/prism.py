import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==========================================
# 1. DSP FILTER FUNCTIONS
# ==========================================

def hampel_filter(data_series, window_size=15, n_sigmas=3):
    """Removes random RF spikes caused by microwave/Bluetooth noise."""
    # Calculate rolling median and MAD (Median Absolute Deviation)
    rolling_median = data_series.rolling(window=window_size, center=True).median()
    rolling_mad = data_series.rolling(window=window_size, center=True).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    
    threshold = n_sigmas * 1.4826 * rolling_mad
    difference = np.abs(data_series - rolling_median)
    
    # Replace outliers with the median
    outlier_idx = difference > threshold
    data_series[outlier_idx] = rolling_median[outlier_idx]
    
    # Fill any NaN values created at the edges
    return data_series.bfill().ffill()

def dynamic_background_subtraction(data_series, window=100):
    """Zeroes out the static room (desks, walls) by subtracting the moving average."""
    rolling_avg = data_series.rolling(window=window, min_periods=1).mean()
    return data_series - rolling_avg

def apply_bandpass(data, lowcut=0.1, highcut=3.0, fs=100, order=3):
    """Strictly isolates the Doppler frequencies of human walking (0.1 Hz to 3.0 Hz)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# ==========================================
# 2. RAW DATA PARSER
# ==========================================

def parse_esp32_file(filepath, target_subcarrier=40):
    """Safely extracts CSI arrays from messy terminal logs."""
    print(f"🔍 Reading raw data from: {filepath}")
    amplitudes = []
    
    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            if "CSI_DATA" in line:
                # Find everything inside the brackets [ ]
                match = re.search(r'\[(.*?)\]', line)
                if match:
                    num_str = match.group(1)
                    # Convert string to numpy array (handles spaces or commas)
                    vals = np.fromstring(num_str.replace(',', ' '), sep=' ')
                    
                    if len(vals) > target_subcarrier * 2:
                        # ESP32 outputs [Real, Imaginary, Real, Imaginary...]
                        # Calculate Amplitude = sqrt(I^2 + Q^2)
                        real = vals[target_subcarrier * 2]
                        imag = vals[(target_subcarrier * 2) + 1]
                        amp = np.sqrt(real**2 + imag**2)
                        amplitudes.append(amp)
                        
    print(f"✅ Successfully extracted {len(amplitudes)} data points.")
    return pd.Series(amplitudes)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    target_file = "zone_a_mov_2.csv"
    
    if not os.path.exists(target_file):
        print(f"❌ ERROR: Cannot find '{target_file}'. Make sure it is in the same folder as this script.")
        exit()

    # 1. Parse the Data
    raw_signal = parse_esp32_file(target_file)
    if len(raw_signal) < 200:
        print("❌ ERROR: Not enough data. Did you record for at least 10 seconds?")
        exit()

    # 2. Apply Pipeline
    print("⚙️ Applying DSP Filters...")
    clean_signal = hampel_filter(raw_signal.copy())
    dynamic_signal = dynamic_background_subtraction(clean_signal)
    
    # We assume roughly 100 packets per second from the active_sta heartbeat
    final_human_signal = apply_bandpass(dynamic_signal, fs=100) 

    # 3. Plotting the Logic
    print("📊 Generating PRISM Radar Plots...")
    plt.style.use('dark_background')
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Project PRISM: Wi-Fi CSI Signal Processing Pipeline', fontsize=16, fontweight='bold')

    # Plot 1: Raw
    axs[0].plot(raw_signal.values, color='gray', alpha=0.8)
    axs[0].set_title('1. Raw Subcarrier Amplitude (Noisy)')
    axs[0].grid(True, alpha=0.2)

    # Plot 2: Hampel
    axs[1].plot(clean_signal.values, color='cyan')
    axs[1].set_title('2. After Hampel Filter (Outlier Spikes Removed)')
    axs[1].grid(True, alpha=0.2)

    # Plot 3: Subtraction
    axs[2].plot(dynamic_signal.values, color='orange')
    axs[2].set_title('3. Dynamic Background Subtraction (Static Room Deleted)')
    axs[2].grid(True, alpha=0.2)

    # Plot 4: Final Bandpass
    axs[3].plot(final_human_signal, color='red', linewidth=1.5)
    axs[3].set_title('4. Butterworth Bandpass (Human Movement Isolated: 0.1 - 3.0 Hz)')
    axs[3].set_xlabel('Time (Packets)')
    axs[3].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()
