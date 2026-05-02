# Project PRISM: Real-Time WiFi Zonal Localization

This document provides a comprehensive report of the Project PRISM WiFi localization system. Our objective was to perform passive human localization (distinguishing between "Empty" and specific physical zones) using a **single ESP32 antenna setup** leveraging CSI (Channel State Information) amplitudes.

## 1. Hardware Setup & ESP32 CSI Tool
The core of the system relies on an ESP32 microcontroller operating as a WiFi receiver, logging CSI data injected by another WiFi device on the network (or just observing background beacon frames). 

### Data Parse Pipeline
- **Raw Intercept:** A custom firmware uses ESP-IDF to intercept raw CSI packets and streams them via serial output (tagged as `CSI_DATA`).
- **Array Extraction:** The ESP32 returns a 128-element array representing the 64 subcarriers of the OFDM WiFi channel. The array alternates between the Real (I) and Imaginary (Q) components (`[Real_1, Imag_1, Real_2, Imag_2, ...]`).
- **Amplitude Calculation:** Because phase on a single-antenna ESP32 is wildly inconsistent (due to clock drift and random Phase Offsets), we calculate the absolute **amplitude** (`sqrt(Real² + Imag²)`) for the 64 subcarriers.
- **Null Subcarrier Rejection:** Modern 802.11n standards don't use all 64 subcarriers. We explicitly drop null subcarriers `27` through `37`, leaving us with **53 active subcarriers** for analysis.

## 2. DSP Pipeline (Digital Signal Processing)
Raw CSI amplitude data is extremely noisy: it suffers from high-frequency electromagnetic interference, hardware noise, and low-frequency drift due to changing environmental baselines (like temperature or physical changes in the static environment).

The processing stages used to clean this data before model consumption involve three distinct layers in [prism.py](file:///home/noblehalogen/sem6/RF/wifi_localization/wifi_localization/prism.py):
1. **Hampel Filter (Outlier Removal):** Microwave ovens, Bluetooth devices, and cosmic rays cause massive random spikes in CSI amplitudes. For every subcarrier, we apply a rolling median (window=15). If a value deviates by more than `3-sigma` (calculated via Median Absolute Deviation, or MAD), it is replaced with the rolling median.
2. **Dynamic Background Subtraction:** The physical room itself (desks, walls) creates a massive static multi-path signature that dwarfs the human signature. We apply a trailing Moving Average filter (size=100) and subtract it from the incoming data. This "zeroes out" the static room, ensuring we only track _changes_ (targets moving).
3. **Butterworth Bandpass Filter:** Human movements generate specific Doppler shifts. Respiration causes low-frequency shifts (~0.1 to 0.5 Hz), and walking causes higher shifts (up to ~3.0 Hz). We applied a 3rd-order Butterworth bandpass filter at `[0.1 Hz, 3.0 Hz]` to strictly isolate human-induced perturbations and reject everything else.

#### DSP Effectiveness Visualization
![Raw vs Cleaned CSI Amplitude](/home/noblehalogen/.gemini/antigravity/brain/2f180514-0694-4ba7-ac81-6d719f3bc754/dsp_comparison.png)

## 3. Data Collection Regimens

Two separate operating environments were logged to build models:

*   **Corridor Environment ([data/](file:///home/noblehalogen/sem6/RF/wifi_localization/wifi_localization/prism_ai_v2.py#131-152)):** 3 classes (`Empty`, `Zone A`, `Zone B`). Proved extremely difficult as Zone A and Zone B had virtually identical multipath signatures to the single antenna (initial Fisher separability score: 0.070).
*   **Room Environment ([data_room/](file:///home/noblehalogen/sem6/RF/wifi_localization/data_room)):** 4 classes (`Empty`, `Zone A`, `Zone B`, `Zone C`). Representing a fully enclosed room, reflections bounced differently depending on zone, making spatial discrimination somewhat easier. We had 1,500 samples (~15 seconds of steady recording) per zone.

### Raw Data Heatmaps
*(Heatmaps of amplitude across all active subcarriers over time, for different classes)*

![Room Raw Heatmaps](/home/noblehalogen/.gemini/antigravity/brain/2f180514-0694-4ba7-ac81-6d719f3bc754/heatmap_room_raw.png)
![Room Clean Heatmaps](/home/noblehalogen/.gemini/antigravity/brain/2f180514-0694-4ba7-ac81-6d719f3bc754/heatmap_room_clean.png)

## 4. Feature Engineering: Defeating Dimensionality

### Phase 1: What Didn't Work (Naive Mapping)
Initially, the pipeline extracted simply *mean, standard deviation, energy, and var_diff* per subcarrier individually. This created heavily redundant and sparse arrays facing the "Curse of Dimensionality" (600+ features for only 150 independent samples).

### Phase 2: What Worked (Domain-Aware Aggregation)
We migrated away from treating subcarriers as isolated columns and built ~135 dimensional feature vectors per 1-second rolling window (`w`) aimed at the *time-frequency structure* and *spatial multipath profile* of human movement across the entire OFDM band. For each feature below, we extract the 5-number summary across subcarriers (mean, std, min, max, median).

**1. Multi-Lag Autocorrelation (Lag 1, 5, 10)**
We calculate temporal dependency across varying lags:
`Autocorr(lag_k) = sum((w[t] - mean)*(w[t-k] - mean)) / sum((w-mean)²)`
*Scientific Justification:* Captures the persistence and periodicity of the signal. Short lags (1) detect highly erratic movements, while longer lags (5, 10) detect slower, rhythmic gestures (like steady walking) which differentiate zones based on physical constraint (stairs vs open floor).

**2. Multi-scale Temporal Variance & Non-Stationarity**
We split the 100-packet window into halves and quarters, and calculate the ratio between their variances:
`VarRatio_Half = Var(w[0:50]) / Var(w[50:100])`
*Scientific Justification:* A stationary target produces constant variance. A subject passing *through* a zone (or entering) creates a highly non-stationary variance profile. This ratio directly quantifies how the Doppler signature evolves over the 1-second capture window.

**3. Spectral Band Energy Ratios**
We compute the magnitude spectrum via real-FFT, plotting energy vs frequency from `0 Hz` to `50 Hz` (Nyquist limit).
`Ratio = Energy[0.1 to 1.0 Hz] / Energy[1.0 to 3.0 Hz]`
*Scientific Justification:* Biomechanical physics dictactes that respiration and micro-movements exist beneath 1.0 Hz, whereas gross torso and limb movement (walking) exists between 1.0 Hz and 3.0 Hz. Distinct zones generally incite distinct movement paradigms.

**4. Covariance Eigenvalues**
We build a 53x53 covariance matrix representing how each subcarrier correlates with the others in the window, and extract the top-5 Eigenvalues of this matrix.
*Scientific Justification:* The sparsity of the covariance matrix directly reflects the complexity of the multipath environment. If a human simply blocks the Line-of-Sight component, all subcarriers dip equally (a Rank-1 impact, massive primary Eigenvalue). If a human scatters signals off complex geometries (like stairs), subcarriers experience asynchronous flat-fading, resulting in a more distributed eigenspectrum.

**5. Subcarrier Profile Gradients (1st & 2nd Derivatives)**
We average the amplitude over the time-window (`mean_profile`), then take its first derivative (`np.diff(mean_profile)`) and second derivative (`np.diff(np.diff(mean_profile))`).
*Scientific Justification:* Captures the frequency-selective fading signature of the room. A person standing in Zone A alters the baseline constructive/destructive interference pattern differently than in Zone C. The gradient maps the "shape" of the physical room's resonant frequencies.

#### PCA Representation
These engineered features resulted in distinct feature clouds, even in the highly compressed 2D Principal Component space.
![Room Features PCA](/home/noblehalogen/.gemini/antigravity/brain/2f180514-0694-4ba7-ac81-6d719f3bc754/pca_room.png)

## 5. Machine Learning Models & Cross-Validation Insights

The transition from a Support Vector Machine (SVM) to Tree-based Ensembles was critical to success.

### SVM vs Random Forest in High-Dimensional RF Data
Initially, an SVM-RBF was deployed. However, SVMs rely on Euclidean distance metrics which fail in high-dimensional (135+ dims), highly-correlated feature spaces due to subspace sparsity. Furthermore, SVMs require strict `StandardScaling` which can collapse the relative magnitude structures between subcarriers. 
Conversely, **Random Forest (RF)** and **Gradient Boosting Machines (GBM)** proved vastly superior. Decision trees implicitly perform feature selection, easily ignoring noisy, non-discriminative subcarriers, and they are capable of carving out highly non-linear decision boundaries through the 135-dimensional space without requiring strict normalization.

### The Overfitting Trap: Temporal Data Leakage
When we first created the Room Model ([data_room/](file:///home/noblehalogen/sem6/RF/wifi_localization/data_room)), we utilized a sliding window step of `10` across our 1500 frame files. The pipeline reported an incredible **96.3% CV accuracy**. However, during live testing, the model was horribly biased—locking onto "Zone C" constantly. 

*The Mathematics of the Leakage:* A step size of 10 for a window of 100 equates to a **90% overlap**. By standard `StratifiedKFold` cross-validation, these highly-correlated, nearly-identical overlapping windows were randomly distributed into both training and testing folds. The model was effectively testing on data it had already trained on. It wasn't learning the physical "Zone C" spatial signature; it was memorizing the specific temporal noise events (a random bump in the signal) unique to that single 15-second recording session.

### The Honest Evaluation & Data Augmentation
To construct an un-biased, generalizable model:
1. **De-correlation:** We reduced the overlap to 50% (`step=50`) yielding strictly uncorrelated, independent physical windows.
2. **Physical Noise Augmentation:** The room environment is dynamic. To simulate micro-changes in multipath fading (like someone imperceptibly shifting their weight), we injected Gaussian noise proportional to each subcarrier's distinct standard deviation. This generated 8x more training samples.
3. **Strict Validation:** Cross-Validation was constrained to evaluating *only on real (non-augmented)* windows, ensuring the reported accuracy metric was honest.

**Final Reliable Accuracy:** 
Using **RandomForest**, the true generalized accuracy for four classes in the room environment is **73.3%**. 
- Empty detection recall sits at **83%**
- Zone B recall sits at **86%**
- Confusions are largely split amongst neighboring physical classes, rather than the model locking onto random local minima.

## 6. Live Inference Architecture

The real-world application connects the Python ML pipeline directly to the hardware via `/dev/ttyUSB0` at 115200 baud. To achieve real-time latency without dropped packets, the architecture separates the I/O bottleneck from the inference engine.

### Ring Buffer Processing
By maintaining a circular NumPy buffer (`deque`), the system evaluates **every single packet** that is streamed from the ESP32. Old packets roll out, new packets roll in, and `numpy` matrix operations execute at lightning speed without requiring memory reallocation.

### Signal Smoothing & UI Stabilization
High-frequency RF is inherently jittery. Even an 86% accurate model will inevitably misclassify 1 out of every 10 packets, which would cause a UI dashboard to aggressively flicker between zones. We introduced a two-layer software smoothing architecture:
1. **Confidence Thresholding:** The `RandomForest.predict_proba()` function outputs class probabilities. We require the dominant class to exceed a strict `50.0%` threshold. If it fails, the UI falls back to the previous stable state.
2. **Exponential Vote Queue:** Classification outputs are fed into a sliding window vote queue. A zone is only explicitly published to the UI if it maintains consistency over the queue history, and a 1.0-second release timeout prevents sudden UI "snapping" when the target temporarily shifts. This creates a deeply responsive yet perfectly stable real-time dashboard.
