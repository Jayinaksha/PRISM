"""
Generate Visualizations for PRISM Project Report
Produces Heatmaps, PCA plots, and DSP comparison plots.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# No seaborn

ROOT_DIR = os.path.expanduser('~/sem6/RF/wifi_localization')
IMG_DIR = os.path.join(ROOT_DIR, 'images')
os.makedirs(IMG_DIR, exist_ok=True)
NULL_SC = list(range(27, 38))

def get_active_cols():
    return [i for i in range(64) if i not in NULL_SC]

def apply_dsp(data, fs=100):
    df = pd.DataFrame(data)
    dyn = (df - df.rolling(window=100, min_periods=1).mean()).values
    nyq = 0.5 * fs
    b, a = butter(3, [0.1/nyq, 3.0/nyq], btype='band')
    if dyn.shape[0] < 15: return dyn
    with np.errstate(all='ignore'):
        clean = filtfilt(b, a, dyn, axis=0)
    return clean

def plot_heatmap(data_dict, title, out_filename, vmin=None, vmax=None):
    fig, axes = plt.subplots(1, len(data_dict), figsize=(5 * len(data_dict), 6))
    if len(data_dict) == 1: axes = [axes]
    
    for ax, (label, data) in zip(axes, data_dict.items()):
        im = ax.imshow(data.T, aspect='auto', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (Packets)')
        ax.set_ylabel('Subcarrier Index')
        
    fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.04)
    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(IMG_DIR, out_filename), dpi=200, bbox_inches='tight')
    plt.close()

def plot_dsp_comparison(raw_data, clean_data, out_filename):
    sc_idx = 10 # Check subcarrier 10
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(raw_data[:, sc_idx], color='salmon', alpha=0.8)
    plt.title('Raw CSI Amplitude (Subcarrier 10)', fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    plt.subplot(1, 2, 2)
    plt.plot(clean_data[:, sc_idx], color='teal')
    plt.title('After Hampel + Dynamic + Bandpass', fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Amplitude Range')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, out_filename), dpi=200)
    plt.close()

def generate_pca(features_dict, out_filename, title):
    # Flatten dict into X and y
    X = []
    y = []
    labels_map = list(features_dict.keys())
    
    for i, label in enumerate(labels_map):
        f = features_dict[label]
        X.append(f)
        y.extend([i] * len(f))
        
    X = np.vstack(X)
    y = np.array(y)
    
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    colors = ['gray', 'cyan', 'orange', 'red']
    for i, label in enumerate(labels_map):
        plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1], label=label, color=colors[i], alpha=0.6, edgecolors='w', s=50)
        
    plt.title(title, fontweight='bold', fontsize=16)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(IMG_DIR, out_filename), dpi=200)
    plt.close()

# Quick feature extraction for PCA (just basic variance and energy, no overlap needed for viz)
def get_basic_features(data):
    features = []
    win_size = 100
    for i in range(0, len(data) - win_size, win_size):
        w = data[i:i+win_size]
        f = np.concatenate([np.var(w, axis=0), np.mean(w**2, axis=0)])
        features.append(f)
    return np.array(features)

if __name__ == "__main__":
    print("Generating visualizations...")
    active_cols = get_active_cols()
    
    # 1. Load Corridor Data
    corridor_files = {'Empty': 'empty_corridor.csv', 'Zone A': 'zone_a.csv', 'Zone B': 'zone_b.csv'}
    corridor_raw = {}
    corridor_clean = {}
    corridor_feat = {}
    
    for label, file in corridor_files.items():
        path = os.path.join(ROOT_DIR, 'data', file)
        if os.path.exists(path):
            raw = pd.read_csv(path).values[:, active_cols]
            clean = apply_dsp(raw)
            corridor_raw[label] = raw[:1000] # Take first 1000 for heatmap
            corridor_clean[label] = clean[:1000]
            corridor_feat[label] = get_basic_features(clean)
            
    # 2. Load Room Data
    room_files = {'Empty': 'empty_room.csv', 'Zone A': 'zone_a.csv', 'Zone B': 'zone_b.csv', 'Zone C': 'zone_c.csv'}
    room_raw = {}
    room_clean = {}
    room_feat = {}
    
    for label, file in room_files.items():
        path = os.path.join(ROOT_DIR, 'data_room', file)
        if os.path.exists(path):
            raw = pd.read_csv(path).values[:, active_cols]
            clean = apply_dsp(raw)
            room_raw[label] = raw[:1000]
            room_clean[label] = clean[:1000]
            room_feat[label] = get_basic_features(clean)
            
    # GENERATE PLOTS
    if corridor_raw:
        print("  -> Creating Corridor Plots")
        plot_heatmap(corridor_raw, "Corridor: Raw CSI Amplitude Heatmaps", "heatmap_corridor_raw.png")
        plot_heatmap(corridor_clean, "Corridor: Filtered CSI Heatmaps", "heatmap_corridor_clean.png", vmin=-5, vmax=5)
        generate_pca(corridor_feat, "pca_corridor.png", "Corridor Data: PCA Scatter (Basic Features)")
        # Pick one for DSP comparison
        plot_dsp_comparison(corridor_raw['Zone A'], corridor_clean['Zone A'], "dsp_comparison.png")
        
    if room_raw:
        print("  -> Creating Room Plots")
        plot_heatmap(room_raw, "Room: Raw CSI Amplitude Heatmaps", "heatmap_room_raw.png")
        plot_heatmap(room_clean, "Room: Filtered CSI Heatmaps", "heatmap_room_clean.png", vmin=-5, vmax=5)
        generate_pca(room_feat, "pca_room.png", "Room Data: PCA Scatter (Basic Features)")
        
    print("Done! Visualizations saved to images/ directory.")
