# Final Project Submission Materials

This document contains everything you need to copy/paste and submit for tomorrow's 5 PM - 7 PM deadline. 

---

## Part 1: Project Writeup (200 Words)

**General Purpose & Specific Usecase Application:**
Project PRISM (Passive RF-based Indoor Spatial Mapping) is an advanced, privacy-preserving localization system designed to detect human presence and precise spatial positioning without relying on cameras or wearable sensors. The general purpose of this project is to enable smart-home automation, elder-care monitoring, and physical intrusion detection using existing Wi-Fi infrastructure. By analyzing the Channel State Information (CSI) embedded in standard Wi-Fi packets, our system captures the minute electromagnetic phase and amplitude shifts caused by human bodies reflecting signals in a room. We specifically target indoor zonal localization, dividing a room into distinct sectors (Empty, Zone A, Zone B, Zone C) and predicting a user's location in real-time. This eliminates the massive privacy concerns associated with CCTV cameras while functioning effortlessly through walls and in total darkness.

**Component Description (50 words):**
The hardware utilizes a single ESP32 microcontroller operating as a Wi-Fi sniffer, intercepting IEEE 802.11n packets. The software backend features a custom Python DSP pipeline (Hampel filtering, Butterworth bandpass) and an advanced 135-dimensional feature-extraction engine running a Random Forest classifier to power a real-time, zero-latency location dashboard.

---

## Part 2: Project Demonstration Video Script (3 Minutes)

*Goal: Make the video exactly 2 minutes and 45 seconds to be safe.*

**[0:00 - 0:30] Introduction & Setup (Voiceover & Camera on Setup)**
"Hello, we are Team [Name/Roll Number]. This is our final project: Real-Time Wi-Fi Zonal Localization. Traditional indoor tracking requires cameras or wearables. Our solution uses invisible Wi-Fi waves. Here is our setup: A single ESP32 microcontroller sniffing Wi-Fi packets, connected to a laptop running our custom Machine Learning pipeline."

**[0:30 - 1:15] The DSP Pipeline (Screen Recording of Code & Dashboard)**
"Standard Wi-Fi signal amplitude is incredibly noisy due to microwave interference and static wall reflections. To fix this, our pipeline applies a rolling Hampel filter to crush outlier spikes, and dynamic background subtraction to mathematically 'delete' the empty room. Finally, a Butterworth bandpass filter precisely isolates the 0.1 to 3.0 Hertz Doppler shifts caused by human walking and breathing."

**[1:15 - 2:00] Feature Engineering & Model (Show the PCA Plot or Heatmaps on screen)**
"We extract 135 complex features per second, including Multi-Lag Autocorrelation and Subcarrier Covariance Eigenvalues, to capture the distinct 'shape' of reflections in different zones. We tested Support Vector Machines, but due to high dimensionality, we ultimately deployed a Random Forest ensemble. To prevent temporal data leakage, we trained with 50% non-overlapping windows and injected Gaussian noise, achieving an honest 73.3% generalized accuracy."

**[2:00 - 2:45] Live Demonstration (Camera showing a person walking + Screen showing Live GUI)**
"Here is the live demonstration. [Person walks into Zone B]. As you can see, the moment they enter Zone B, the Ring-Buffer pipeline evaluates the packets in real-time, the Random Forest passes a 50% confidence threshold, and our exponential smoothing algorithm dynamically updates the UI to Zone B. When they leave, the system immediately recognizes the room is Empty. Thank you."

---

## Part 3: 15-Slide PPT Layout

*Copy these headers and bullet points directly into PowerPoint. Insert the referenced images from your `images/` directory.*

### Slide 1: Title Slide
- **Title:** Real-Time Passive Wi-Fi Zonal Localization
- **Subtitle:** High-Accuracy Spatial Mapping using ESP32 Channel State Information
- **Team Details:** [Your Name], Roll Number: [Your Roll Number]
- **Team Details:** [Partner 2 Name], Roll Number: [Partner 2 Roll]
- **Team Details:** [Partner 3 Name], Roll Number: [Partner 3 Roll]

### Slide 2: Motivation & Problem Statement
- **Problem:** Indoor tracking relies on intrusive cameras or battery-powered wearables. 
- **Privacy:** Cameras compromise privacy (bedrooms, hospitals).
- **Solution:** Passive Wi-Fi sensing. Humans are mostly water and reflect 2.4GHz RF signals. We can track people using the ambient Wi-Fi waves already bouncing around the room.

### Slide 3: What is CSI? (Channel State Information)
- Standard Wi-Fi routers use OFDM (Orthogonal Frequency-Division Multiplexing).
- The 2.4GHz band is split into 64 thinner subcarriers.
- CSI gives us the Amplitude and Phase of *each individual subcarrier*, effectively acting as a 64-pixel RF camera.

### Slide 4: System Architecture
- **Hardware:** 1x ESP32 NodeMCU.
- **Capture:** Sniffing Wi-Fi packets via ESP-IDF firmware.
- **Processing Unit:** Python backend (NumPy, SciPy, Scikit-Learn) directly attached via Serial over USB.
- **Output:** Live updating matched-zone UI terminal.

### Slide 5: Raw Signal Extraction
- We intercept the raw 128-element array from the ESP32.
- Due to single-antenna clock drift, raw Phase is unusable. 
- We compute Absolute Amplitude using Euclidean distance: $Amplitude = \sqrt{Real^2 + Imaginary^2}$.
- We discard null subcarriers (27-37), leaving 53 active RF streams.

### Slide 6: DSP Pipeline Stage 1 (Noise Reduction)
- **Problem:** Environmental interference (Bluetooth, Microwaves, Cosmic rays) causes massive spikes.
- **Solution - Hampel Filter:** We apply a rolling median window. Any sudden spike exceeding 3 standard deviations (MAD) is mathematically replaced by the median, stabilizing the signal.

### Slide 7: DSP Pipeline Stage 2 & 3 (Human Isolation)
- **Dynamic Background Subtraction:** The static room (walls, desks) dominates the signal. We subtract a trailing 100-packet moving average to zero-out the geometry and only track *changes*.
- **Butterworth Bandpass Filter:** Humans induce specific Doppler shifts. We apply a 3rd-order Butterworth bandpass at `[0.1 Hz, 3.0 Hz]` to isolate breathing and walking velocities.
- *[INSERT IMAGE]*: [dsp_comparison.png](file:///home/noblehalogen/sem6/RF/wifi_localization/images/dsp_comparison.png)

### Slide 8: Data Collection Environments
- **Environment 1 (Corridor):** Zones A & B. Proved geometrically challenging due to symmetrical multipath behavior.
- **Environment 2 (Enclosed Room):** Empty, Zone A, Zone B, Zone C. Complex multipath geometry made spatial resolution much better. Extracted 1,500 continuous packets per zone.
- *[INSERT IMAGE]*: [heatmap_room_clean.png](file:///home/noblehalogen/sem6/RF/wifi_localization/images/heatmap_room_clean.png)

### Slide 9: Feature Engineering: The Dimensionality Problem
- Standard Machine Learning methods (Mean, Variance per subcarrier) left us with sparse matrices and poor separation.
- **The Upgrade:** We engineered 135-dimensional vectors focused on *Time-Frequency Structure*.
- Key Feature: **Multi-Lag Autocorrelation** (Lags 1, 5, 10) to determine rhythmic motion vs sharp transients.

### Slide 10: Advanced Spectral Features
- **Temporal Non-Stationarity:** Calculating the ratio of variance between the first-half and second-half of a 1-second window to detect subjects moving *through* the RF boundary rather than standing still.
- **Covariance Eigenvalues:** We pull the top-5 Eigenvalues of the subcarrier covariance matrix to map the sparsity of the multipath fading (how complex the reflections are in a given zone).
- *[INSERT IMAGE]*: [pca_room.png](file:///home/noblehalogen/sem6/RF/wifi_localization/images/pca_room.png)

### Slide 11: Machine Learning Selection
- **Discarded SVM:** Support Vector Machines struggled. They scale poorly in high correlation spaces and strict `StandardScaling` ruined our relative subcarrier magnitude physics.
- **Chosen Model:** Random Forest / HistGradientBoosting.
- Decision trees implicitly feature-select, cutting through noisy subcarriers and finding non-linear decision boundaries through the 135 variables.

### Slide 12: Overcoming The Leakage Trap
- **The Bug:** Our initial model reached 96.3% accuracy, but failed entirely in live inference. 
- **The Cause:** A sliding window `step` of 10 created 90% overlap. K-Fold Cross-Validation leaked near-identical training frames into the testing set. The model memorized local noise, not the physical zones.

### Slide 13: Honest Validation & Results
- We corrected the overlap to an honest 50% (`step=50`).
- Generated 8x augmented data by injecting physical Gaussian noise scaled to specific subcarrier variance to simulate dynamic multipath changes.
- **Final Generalized Accuracy:** 73.3% across 4 zones in an un-seen environment, with Empty Detection recall exceeding 83%.

### Slide 14: Live Inference Architecture
- Serial I/O bottleneck solved using unallocated NumPy `deque` Ring Buffers for continuous streaming.
- **UI Stabilization Stage 1:** The Random Forest must pass a strict >50.0% confidence probability to trigger a zone shift.
- **UI Stabilization Stage 2:** Handled by a 3-vote Exponential Queue with a 1.0s release timeout, preventing screen flickering.

### Slide 15: Conclusion & Future Work
- **Conclusion:** Single-antenna ESP32s *can* perform spatial mapping and zoning with heavy DSP and advanced time-frequency feature extraction.
- **Future Integration:** Fusing this RF data with our `Vinayabrhami` AI OS for robotics.
- **Next Steps:** Upgrading to multi-antenna setups (MIMO) Phase-difference processing (CFO calibration) for Angle of Arrival (AoA) tracking.
