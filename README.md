<div align="center">

# 📡 Project PRISM

### **P**assive **R**F-based **I**ndoor **S**patial **M**apping

**Real-Time Wi-Fi Zonal Localization Using ESP32 Channel State Information**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8+-orange.svg)](https://scikit-learn.org/)
[![ESP32](https://img.shields.io/badge/hardware-ESP32-green.svg)](https://www.espressif.com/)
[![License](https://img.shields.io/badge/license-Academic-lightgrey.svg)]()

---

*High-accuracy, privacy-preserving human tracking through walls and in total darkness — using a single $4 microcontroller and invisible Wi-Fi waves.*

</div>

---

## 🧠 What Is PRISM?

Project PRISM is a **passive indoor localization system** that detects human presence and predicts spatial position in real-time without cameras, microphones, or wearable devices. It exploits **Channel State Information (CSI)** — the fine-grained amplitude and phase data embedded in every Wi-Fi packet — to sense how human bodies perturb the electromagnetic field in a room.

By deploying a custom Digital Signal Processing (DSP) pipeline and a 135-dimensional machine learning feature engine on data from a **single ESP32 antenna**, PRISM divides indoor spaces into discrete zones and classifies a person's location at ~100 Hz.

### Key Capabilities

| Capability | Detail |
|:---|:---|
| **Zones** | Up to 4 (Empty, Zone A, Zone B, Zone C) |
| **Accuracy** | 73.3% generalized (4-zone room), 81.4% CV (2-zone corridor) |
| **Latency** | Real-time (~10ms per inference cycle) |
| **Hardware** | Single ESP32 NodeMCU ($4) |
| **Privacy** | Zero visual/audio data captured |
| **Conditions** | Works through walls, in complete darkness |

---

## 📡 Hardware Requirements

| Component | Purpose |
|:---|:---|
| **1× ESP32 NodeMCU** | Flashed with ESP-IDF CSI extraction firmware — operates as a Wi-Fi sniffer |
| **1× Laptop/PC** | Runs the Python ML backend; connected via Serial USB (`/dev/ttyUSB0`) |
| **Ambient Wi-Fi** | Any standard 2.4GHz 802.11n router or device within range |
| **USB Cable** | Micro-USB for ESP32 serial communication at 115200 baud |

> **No additional sensors, cameras, or wearable devices are required.**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRISM Architecture                           │
│                                                                     │
│  ┌──────────┐    Serial     ┌──────────────┐    ┌───────────────┐  │
│  │  ESP32   │───115200bd──→ │  CSI Parser  │──→ │  Ring Buffer  │  │
│  │ (Sniffer)│   /dev/USB0   │ (I/Q → Amp)  │    │  (100 pkts)   │  │
│  └──────────┘               └──────────────┘    └───────┬───────┘  │
│                                                         │          │
│                              ┌──────────────────────────▼────────┐ │
│                              │        DSP Pipeline               │ │
│                              │  1. Hampel Filter (outlier kill)  │ │
│                              │  2. Background Subtraction        │ │
│                              │  3. Butterworth Bandpass (0.1-3Hz)│ │
│                              └──────────────────┬────────────────┘ │
│                                                 │                  │
│                              ┌──────────────────▼────────────────┐ │
│                              │    Feature Engine (135-dim)        │ │
│                              │  • Multi-Lag Autocorrelation       │ │
│                              │  • Variance Ratios                 │ │
│                              │  • Spectral Band Energy            │ │
│                              │  • Covariance Eigenvalues          │ │
│                              │  • Subcarrier Profile Gradients    │ │
│                              └──────────────────┬────────────────┘ │
│                                                 │                  │
│  ┌───────────────┐    ┌─────────────────────────▼────────────────┐ │
│  │  Live GUI     │←── │   Random Forest Classifier               │ │
│  │ (Matplotlib)  │    │   + Confidence Thresholding (>50%)       │ │
│  │  Zone Display │    │   + 3-Vote Exponential Smoothing Queue   │ │
│  └───────────────┘    └──────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```text
wifi_localization/
│
├── README.md                      # This file
├── walkthrough.md                 # Detailed technical report
├── final_submission_materials.md  # Presentation script, slide layout, writeup
│
├── data/                          # CSI amplitude logs — Corridor environment
│   ├── empty_area.csv             #   Empty corridor (3 recordings)
│   ├── zone_a.csv                 #   Zone A occupancy (3 recordings)
│   └── zone_b.csv                 #   Zone B occupancy (3 recordings)
│
├── data_room/                     # CSI amplitude logs — Room environment
│   ├── empty_room.csv             #   Empty room
│   ├── zone_a.csv                 #   Zone A occupancy
│   ├── zone_b.csv                 #   Zone B occupancy
│   └── zone_c.csv                 #   Zone C occupancy
│
├── exp_data/                      # Experimental data from multiple environments
│   ├── sparkonics_lab_*.csv       #   Sparkonics Lab captures
│   ├── stc_*.csv                  #   STC building captures
│   ├── stairs_*.csv               #   Stairwell captures
│   └── tl_*.csv                   #   TL environment captures
│
├── images/                        # Generated visualizations
│   ├── heatmap_room_raw.png       #   Raw CSI amplitude heatmaps
│   ├── heatmap_room_clean.png     #   Filtered CSI heatmaps
│   ├── dsp_comparison.png         #   Raw vs cleaned signal comparison
│   ├── pca_room.png               #   PCA scatter plot (room features)
│   ├── pca_corridor.png           #   PCA scatter plot (corridor features)
│   ├── feature_importance_*.png   #   Random Forest feature importances
│   └── zone_*.png                 #   Per-zone signal plots
│
├── wifi_localization/             # Source code
│   ├── pyproject.toml             #   Python dependencies (uv managed)
│   ├── uv.lock                    #   Locked dependency versions
│   │
│   ├── prism.py                   # ⚡ Core DSP filter library
│   │                              #    Hampel, Background Sub, Butterworth
│   │
│   ├── prism_debug.py             # 🔧 ESP32 serial debug tool
│   │                              #    Raw packet inspection for 10 seconds
│   │
│   ├── csi_logger.py              # 📝 Data harvesting script
│   │                              #    Records CSI from live ESP32 to CSV
│   │
│   ├── prism_ai.py                # 🤖 v1 ML training (SVM-RBF, 4-class)
│   ├── prism_ai_prev.py           # 🤖 v0 ML training (RF, basic variance)
│   ├── prism_ai_v2.py             # 🤖 v2 ML training (corridor, 87-dim)
│   ├── prism_ai_room.py           # 🤖 v3 ML training (room, 135-dim) ← BEST
│   │
│   ├── prism_live_room.py         # 🔴 Live inference (v1 model, 4-zone)
│   ├── prism_live_room_v2.py      # 🔴 Live inference (v2 corridor, 2-zone)
│   ├── prism_live_room_room.py    # 🔴 Live inference (room model, 4-zone) ← BEST
│   │
│   ├── prism_model.pkl            # Serialized v1 model
│   ├── prism_model_v2.pkl         # Serialized corridor model
│   ├── prism_model_room.pkl       # Serialized room model ← BEST
│   │
│   ├── generate_visualizations.py # 📊 Heatmap, PCA, DSP comparison generator
│   ├── create_pptx.py             # 📑 Auto-generates presentation slides
│   │
│   ├── corridor/                  # Organized corridor environment copies
│   │   ├── prism_ai_v2.py
│   │   ├── prism_live_room_v2.py
│   │   └── prism_model_v2.pkl
│   │
│   └── room/                      # Organized room environment copies
│       ├── prism_ai_room.py
│       ├── prism_live_room_room.py
│       └── prism_model_room.pkl
│
├── PRISM_Zonal_Localization.pptx  # Generated presentation
└── RF_PRISM*.mp4                  # Demo videos
```

---

## ⚙️ The DSP Pipeline

The raw CSI amplitude from the ESP32 is devastatingly noisy. PRISM applies a three-stage Digital Signal Processing pipeline (implemented in [`prism.py`](wifi_localization/prism.py)) to isolate the human-induced perturbations:

### Stage 1 — Hampel Filter (Outlier Removal)

Bluetooth, microwaves, and other RF sources cause massive random spikes. A rolling median window (size=15) replaces any value exceeding 3σ (via Median Absolute Deviation) with the local median.

### Stage 2 — Dynamic Background Subtraction

Static room geometry (walls, desks) dominates the raw signal. A trailing 100-packet moving average is subtracted to zero out the static environment, isolating only dynamic (human-induced) changes.

### Stage 3 — Butterworth Bandpass Filter

A 3rd-order Butterworth bandpass at **0.1–3.0 Hz** eliminates low-frequency drift and high-frequency electronic noise, isolating the Doppler frequencies of human breathing (~0.1–0.5 Hz) and walking (~1.0–3.0 Hz).

### Signal Extraction

- The ESP32 outputs a 128-element array per packet: `[Real₁, Imag₁, Real₂, Imag₂, ...]`
- Amplitude is computed as: **A = √(I² + Q²)** (phase discarded due to single-antenna clock drift)
- Null subcarriers 27–37 are dropped per IEEE 802.11n → **53 active subcarriers**

---

## 🔬 Feature Engineering (135 Dimensions)

The critical breakthrough was moving from naive per-subcarrier statistics to **domain-aware time-frequency features**. For each 100-packet (1-second) window, we extract:

| Feature Group | Dimensions | Scientific Justification |
|:---|:---:|:---|
| **Basic Statistics** | 40 | Variance, std, energy, diff-variance, skewness, kurtosis, IQR, range (5-number summary each) |
| **Multi-Lag Autocorrelation** | 15 | Lags 1, 5, 10 capture signal persistence — separates erratic noise from rhythmic walking |
| **Temporal Derivatives** | 10 | 1st and 2nd order temporal diff-variance detects acceleration patterns |
| **Multi-Scale Variance Ratios** | 10 | Half/quarter window variance ratios detect subjects crossing zone boundaries |
| **Spectral Features** | 20 | FFT peak frequency, spectral centroid, spectral bandwidth, band energy ratios (breathing vs walking) |
| **Subcarrier Profile Gradients** | 10 | 1st and 2nd derivatives of the mean amplitude profile capture frequency-selective fading |
| **Covariance Eigenvalues** | 5 | Top-5 eigenvalues of the 53×53 subcarrier covariance matrix map multipath complexity |
| **Correlation Statistics** | 3 | Mean, std, median of upper-triangle cross-subcarrier correlations |
| **Global Metrics** | 2 | Total energy, subcarrier entropy |
| **Top-10 Subcarrier Features** | 20 | Variance and energy of the 10 most variable subcarriers |

---

## 🤖 Machine Learning Models

### Model Evolution

| Version | Script | Model | Features | Classes | Accuracy | Notes |
|:---|:---|:---|:---:|:---:|:---:|:---|
| v0 | `prism_ai_prev.py` | Random Forest | ~53 (variance only) | 4 | ~65% | Basic per-subcarrier variance |
| v1 | `prism_ai.py` | SVM-RBF | ~212 (var+std+energy+diff) | 4 | Variable | Leave-One-Chunk-Out CV |
| v2 | `prism_ai_v2.py` | GradientBoosting | 87 | 3 (corridor) | **81.4%** | Best corridor model |
| **v3** | **`prism_ai_room.py`** | **RandomForest** | **135** | **4 (room)** | **73.3%** | **Production model** |

### Why Random Forest Over SVM?

- **SVM-RBF** requires `StandardScaling` which destroys the relative magnitude physics between subcarriers
- **SVMs** scale poorly in high-dimensional (135+), highly-correlated feature spaces
- **Random Forest** implicitly feature-selects, carves non-linear decision boundaries, and needs no normalization

### The Overfitting Trap (Temporal Data Leakage)

> **The Bug:** Initial models reported 96.3% accuracy but failed completely in live inference.
>
> **Root Cause:** A sliding window step of 10 (on a 100-packet window) created 90% overlap. K-Fold CV leaked near-identical frames across train/test splits. The model memorized local noise patterns, not physical zone signatures.
>
> **The Fix:**
> 1. Reduced overlap to 50% (`step=50`) for truly independent windows
> 2. 8× Gaussian noise augmentation (scaled per-subcarrier std) simulating dynamic multipath changes
> 3. Cross-validation runs only on real (non-augmented) samples

### Final Validated Performance (Room Model)

| Metric | Score |
|:---|:---:|
| **Overall Accuracy** | 73.3% |
| **Empty Detection Recall** | >83% |
| **Zone B Recall** | >86% |
| **False Positive Rate** | Low — confusions largely between neighboring physical zones |

---

## 🔴 Live Inference Architecture

The real-time system ([`prism_live_room_room.py`](wifi_localization/prism_live_room_room.py)) streams from the ESP32 at 115200 baud and runs inference on every incoming packet:

### Ring Buffer Processing

A circular NumPy buffer maintains the latest 100 packets. Old data rolls out, new data rolls in — `numpy` matrix operations execute without memory reallocation.

### Two-Layer UI Stabilization

Even an 86% accurate model will misclassify ~1/10 packets, causing UI flicker. PRISM solves this with:

1. **Confidence Thresholding:** `predict_proba()` must exceed **50%** for the dominant class. Below threshold → fallback to previous stable state.

2. **Exponential Vote Queue:** A 3-vote sliding window requires unanimous agreement before switching zones. A **1.0-second release timeout** prevents zone "sticking" when the target leaves.

### Live GUI

The Matplotlib-based dashboard renders zone rectangles that light up in real-time as the model classifies human position:

```
┌──────────────────────────────────────────────────┐
│          STATUS: TARGET IN ZONE B                │
│  ┌──────────┐  ┌──████████┐  ┌──────────────┐   │
│  │          │  │ ████████ │  │              │   │
│  │  Zone A  │  │  ZONE B  │  │    Zone C    │   │
│  │          │  │ ████████ │  │              │   │
│  └──────────┘  └──████████┘  └──────────────┘   │
└──────────────────────────────────────────────────┘
```

---

## 🚀 Usage

We use [`uv`](https://docs.astral.sh/uv/) to manage Python dependencies.

### 1. Install Dependencies

```bash
cd wifi_localization/
uv sync
```

### 2. Debug ESP32 Connection

Verify your ESP32 is streaming CSI data correctly:

```bash
uv run prism_debug.py
```

This prints raw serial lines for 10 seconds and reports packet rate.

### 3. Collect Training Data

Record CSI data from the live ESP32 to CSV files:

```bash
uv run csi_logger.py
```

### 4. Train the Room Model

Retrain the model with new data or modified hyperparameters:

```bash
uv run prism_ai_room.py
```

This automatically handles:
- 50% non-overlapping sliding windows
- 8× Gaussian noise augmentation
- 5-fold Stratified CV on real data only
- Model comparison (SVM-RBF, RandomForest, HistGBM)
- Feature importance plots

### 5. Train the Corridor Model

```bash
uv run prism_ai_v2.py
```

### 6. Run Live Inference (Room Radar)

Fire up the real-time dashboard with the room model:

```bash
uv run prism_live_room_room.py
```

> **⚠️ Note:** Ensure your ESP32 is plugged into `/dev/ttyUSB0` and `idf.py monitor` is **not** running. If your port differs (e.g., `COM3` on Windows), modify the `SERIAL_PORT` variable at the top of the script.

### 7. Run Live Inference (Corridor Radar)

```bash
uv run prism_live_room_v2.py
```

### 8. Generate Visualizations

Create heatmaps, PCA plots, and DSP comparison images:

```bash
uv run generate_visualizations.py
```

### 9. Generate Presentation Slides

Auto-generate the PowerPoint deck:

```bash
uv run --with python-pptx create_pptx.py
```

---

## 📊 Data Collection Environments

### Environment 1: Corridor (2 Zones)

- **Classes:** Empty, Zone A, Zone B
- **Challenge:** Symmetric geometry created near-identical multipath signatures
- **Fisher Separability Score:** 0.070 (extremely low)
- **Data:** 9 CSV files across 3 recording sessions per class

### Environment 2: Enclosed Room (4 Zones)

- **Classes:** Empty, Zone A, Zone B, Zone C
- **Advantage:** Enclosed walls create distinct multipath reflections per zone
- **Data:** 1,500 continuous packets (~15 seconds steady recording) per zone
- **Result:** Substantially better spatial discrimination

### Experimental Data

Additional captures from diverse environments (STC building, Sparkonics Lab, stairwells) are stored in `exp_data/` for extended analysis.

---

## 🧪 Dependencies

| Package | Version | Purpose |
|:---|:---|:---|
| `numpy` | ≥2.4.4 | Matrix operations, ring buffers |
| `pandas` | ≥3.0.2 | Data loading, rolling window calculations |
| `scipy` | ≥1.17.1 | Butterworth filter, signal processing |
| `scikit-learn` | ≥1.8.0 | Random Forest, SVM, PCA, cross-validation |
| `matplotlib` | ≥3.10.8 | Live GUI dashboard, visualization generation |
| `pyserial` | ≥3.5 | ESP32 serial communication |
| `python-pptx` | (optional) | PowerPoint slide generation |

**Python:** ≥ 3.11

---

## 📚 Technical References

- **CSI Extraction:** ESP-IDF Wi-Fi CSI firmware for ESP32
- **Hampel Filter:** Friedrich R. Hampel's robust outlier detection via MAD
- **Butterworth Filter:** 3rd-order Infinite Impulse Response (IIR) bandpass
- **Feature Engineering:** Inspired by radar micro-Doppler signature analysis
- **Validation:** Stratified K-Fold with temporal de-correlation (50% non-overlapping windows)

---

## 📄 Files Reference

| Script | Role | Input | Output |
|:---|:---|:---|:---|
| `prism.py` | Core DSP library | Raw CSV | Cleaned signal + plots |
| `prism_debug.py` | ESP32 serial debugger | `/dev/ttyUSB0` | Terminal diagnostics |
| `prism_ai_room.py` | Room model trainer | `data_room/*.csv` | `prism_model_room.pkl` |
| `prism_ai_v2.py` | Corridor model trainer | `data/*.csv` | `prism_model_v2.pkl` |
| `prism_live_room_room.py` | Live radar (room) | Serial + `.pkl` | Real-time GUI |
| `prism_live_room_v2.py` | Live radar (corridor) | Serial + `.pkl` | Real-time GUI |
| `generate_visualizations.py` | Plot generator | `data/`, `data_room/` | `images/*.png` |
| `create_pptx.py` | Slide generator | `images/` | `.pptx` |

---

<div align="center">

*Built as part of the Vinayabrhami AI OS Architecture.*

**Project PRISM** — Seeing through walls with invisible waves. 📡

</div>
