"""
PRISM Live Room — Real-time zonal radar for room environment
Uses prism_model_room.pkl (HistGBM, 135-dim features, 4 classes)
"""
import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

SERIAL_PORT  = '/dev/ttyUSB0'
BAUD_RATE    = 115200
WINDOW_SIZE  = 100
SMOOTHING_VOTES = 3
CONFIDENCE_THRESHOLD = 0.40
ZONE_RELEASE_TIMEOUT = 1.0

# Load room model
try:
    clf = joblib.load("prism_model_room.pkl")
    print("🧠 PRISM Room Model Loaded Successfully!")
except FileNotFoundError:
    print("❌ ERROR: 'prism_model_room.pkl' not found. Run prism_ai_room.py first!")
    exit()

NULL_SUBCARRIERS = list(range(27, 38))
ACTIVE_COLS = [i for i in range(64) if i not in NULL_SUBCARRIERS]

# ==========================================
# DSP
# ==========================================

def apply_live_dsp(data_buffer, fs=100):
    df = pd.DataFrame(data_buffer)
    dyn_data = (df - df.rolling(window=100, min_periods=1).mean()).values
    nyq = 0.5 * fs
    b, a = butter(3, [0.1 / nyq, 3.0 / nyq], btype='band')
    if dyn_data.shape[0] <= 15:
        return dyn_data
    return filtfilt(b, a, dyn_data, axis=0)

# ==========================================
# FEATURE EXTRACTION — must match prism_ai_room.py EXACTLY
# ==========================================

def _agg(v):
    return [np.mean(v), np.std(v), np.min(v), np.max(v), np.median(v)]

def _lag_autocorr(w, lag):
    w0, w1 = w[:-lag, :], w[lag:, :]
    w0m, w1m = w0 - w0.mean(0), w1 - w1.mean(0)
    return np.sum(w0m*w1m,0)/(np.sqrt(np.sum(w0m**2,0)*np.sum(w1m**2,0))+1e-10)

def extract_features(w):
    f = []

    # Basic stats
    f.extend(_agg(np.var(w, axis=0)))
    f.extend(_agg(np.std(w, axis=0)))
    f.extend(_agg(np.mean(w**2, axis=0)))
    f.extend(_agg(np.var(np.diff(w, axis=1), axis=0)))
    f.extend(_agg(skew(w, axis=0)))
    f.extend(_agg(kurtosis(w, axis=0)))
    f.extend(_agg(np.percentile(w,75,0)-np.percentile(w,25,0)))
    f.extend(_agg(np.max(w,0)-np.min(w,0)))

    # Temporal: multi-lag autocorrelation
    f.extend(_agg(_lag_autocorr(w, 1)))
    f.extend(_agg(_lag_autocorr(w, 5)))
    f.extend(_agg(_lag_autocorr(w, 10)))
    f.extend(_agg(np.var(np.diff(w,axis=0),axis=0)))
    f.extend(_agg(np.var(np.diff(np.diff(w,axis=0),axis=0),axis=0)))

    # Multi-scale variance ratios
    half = WINDOW_SIZE // 2
    quarter = WINDOW_SIZE // 4
    var_half1 = np.var(w[:half,:], axis=0)
    var_half2 = np.var(w[half:,:], axis=0)
    var_q1 = np.var(w[:quarter,:], axis=0)
    var_q4 = np.var(w[-quarter:,:], axis=0)
    f.extend(_agg((var_half1+1e-10)/(var_half2+1e-10)))
    f.extend(_agg((var_q1+1e-10)/(var_q4+1e-10)))

    # Frequency domain
    fft_m = np.abs(np.fft.rfft(w, axis=0))
    freqs = np.fft.rfftfreq(w.shape[0], d=0.01)
    f.extend(_agg(freqs[np.argmax(fft_m[1:,:],0)+1]))
    fft_p = fft_m[1:,:]**2; fa = freqs[1:]
    sc = np.sum(fft_p*fa[:,None],0)/(np.sum(fft_p,0)+1e-10)
    f.extend(_agg(sc))
    sb = np.sqrt(np.sum(fft_p*(fa[:,None]-sc[None,:])**2, 0)/(np.sum(fft_p,0)+1e-10))
    f.extend(_agg(sb))
    low_mask = (fa >= 0.1) & (fa <= 1.0)
    high_mask = (fa > 1.0) & (fa <= 3.0)
    low_e = np.sum(fft_p[low_mask,:], axis=0) + 1e-10
    high_e = np.sum(fft_p[high_mask,:], axis=0) + 1e-10
    f.extend(_agg(low_e / high_e))

    # Subcarrier profile
    mean_profile = np.mean(w, axis=0)
    f.extend(_agg(np.diff(mean_profile)))
    f.extend(_agg(np.diff(np.diff(mean_profile))))

    # Cross-subcarrier correlation
    with np.errstate(divide='ignore', invalid='ignore'):
        cm = np.corrcoef(w.T)
    cm = np.nan_to_num(cm, nan=0.0)
    try:
        eig = np.sort(np.linalg.eigvalsh(cm))[::-1][:5]
    except:
        eig = np.zeros(5)
    f.extend(eig.tolist())
    upper = cm[np.triu_indices_from(cm, k=1)]
    f.extend([np.mean(upper), np.std(upper), np.median(upper)])

    # Global
    f.append(np.sum(w**2))
    sp = np.mean(w**2,0); sp = sp/(sp.sum()+1e-10)
    f.append(-np.sum(sp*np.log(sp+1e-10)))

    # Top-10 per-subcarrier
    var_per_sc = np.var(w, axis=0)
    top10 = np.argsort(var_per_sc)[::-1][:10]
    f.extend(var_per_sc[top10].tolist())
    f.extend(np.mean(w**2, axis=0)[top10].tolist())

    return np.nan_to_num(np.array(f), nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)

# ==========================================
# CSI PARSER
# ==========================================

def parse_csi_line(line):
    if not line.startswith("CSI_DATA"):
        return None
    parts = line.split(",[")
    if len(parts) != 2:
        return None
    raw_csi_string = parts[1].replace("]", "").strip()
    try:
        csi_integers = list(map(int, raw_csi_string.split()))
    except ValueError:
        return None
    if len(csi_integers) < 128:
        return None
    real_parts = np.array(csi_integers[0::2], dtype=float)
    imag_parts = np.array(csi_integers[1::2], dtype=float)
    amps = np.sqrt(real_parts ** 2 + imag_parts ** 2)
    return amps[ACTIVE_COLS]

# ==========================================
# GUI SETUP (4 zones)
# ==========================================

plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
fig.canvas.manager.set_window_title('Project PRISM: Room Radar')
plt.style.use('dark_background')
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.set_title("Project PRISM: Room Zonal Tracking", fontsize=18, fontweight='bold', color='white')
ax.axis('off')

zones = {
    0: [0,   0,  12, 6, "EMPTY ROOM",    "gray"],
    1: [0.5, 0.5, 3, 5, "Zone A",        "cyan"],
    2: [4.0, 0.5, 3, 5, "Zone B",        "orange"],
    3: [7.5, 0.5, 4, 5, "Zone C",        "#ff4444"],
}

rect_patches = {}
text_patches = {}
for zone_id, (x, y, w, h, label, color) in zones.items():
    if zone_id == 0:
        continue
    rect = Rectangle((x, y), w, h, linewidth=3, edgecolor="#555555", facecolor="#222222", alpha=0.3)
    ax.add_patch(rect)
    rect_patches[zone_id] = rect
    txt = ax.text(x + w/2, y + h/2, label, color="white", fontsize=14, weight="normal", ha='center', va='center')
    text_patches[zone_id] = txt

status_banner = ax.text(6, 5.5, "STATUS: WARMING UP...", color="orange", fontsize=16, weight='bold',
    ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))
fig.canvas.draw()

def update_gui(current_zone):
    for zone_id in [1, 2, 3]:
        is_active = (current_zone == zone_id)
        color = zones[zone_id][5]
        rect_patches[zone_id].set_facecolor(color if is_active else "#222222")
        rect_patches[zone_id].set_edgecolor(color if is_active else "#555555")
        rect_patches[zone_id].set_alpha(0.8 if is_active else 0.3)
        text_patches[zone_id].set_color('black' if is_active else 'white')
        text_patches[zone_id].set_weight('bold' if is_active else 'normal')
    if current_zone == 0:
        status_banner.set_text("STATUS: NO TARGET DETECTED")
        status_banner.set_color("lightgreen")
    else:
        status_banner.set_text(f"STATUS: TARGET IN {zones[current_zone][4].upper()}")
        status_banner.set_color("yellow")
    fig.canvas.flush_events()

# ==========================================
# MAIN LOOP
# ==========================================

print(f"📡 Opening {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("✅ Connected. Waiting for CSI packets...\n")
except Exception:
    print(f"❌ Cannot open {SERIAL_PORT}. Close idf.py monitor first!")
    exit()

data_buffer         = []
vote_history        = []
displayed_zone      = 0
packets_seen        = 0
last_packet_time    = time.time()
last_confident_time = time.time()
update_gui(0)

try:
    while True:
        raw = ser.readline()
        if not raw:
            if packets_seen > 0 and (time.time() - last_packet_time) > 5:
                status_banner.set_text("STATUS: NO SIGNAL — CHECK ESP32")
                status_banner.set_color("red")
                fig.canvas.flush_events()
            continue

        line = raw.decode('utf-8', errors='ignore').strip()
        if not line or not line.startswith("CSI_DATA"):
            if "connected with" in line or "waiting" in line.lower():
                print(f"[ESP32] {line}")
            continue

        amps = parse_csi_line(line)
        if amps is None:
            continue

        packets_seen += 1
        last_packet_time = time.time()
        data_buffer.append(amps)

        if len(data_buffer) < WINDOW_SIZE:
            pct = int(len(data_buffer) / WINDOW_SIZE * 100)
            status_banner.set_text(f"STATUS: WARMING UP... {pct}%")
            fig.canvas.flush_events()
            continue

        # Inference on every packet
        window = np.array(data_buffer[-WINDOW_SIZE:])
        clean  = apply_live_dsp(window)
        fp     = extract_features(clean)

        probabilities  = clf.predict_proba(fp)[0]
        predicted_zone = int(np.argmax(probabilities))
        top_confidence = probabilities[predicted_zone]

        if top_confidence >= CONFIDENCE_THRESHOLD:
            vote = predicted_zone
            last_confident_time = time.time()
        else:
            vote = 0

        vote_history.append(vote)
        if len(vote_history) > SMOOTHING_VOTES:
            vote_history.pop(0)

        if len(vote_history) == SMOOTHING_VOTES and len(set(vote_history)) == 1:
            new_zone = vote_history[0]
            if new_zone != displayed_zone:
                displayed_zone = new_zone

        if displayed_zone != 0 and (time.time() - last_confident_time) > ZONE_RELEASE_TIMEOUT:
            displayed_zone = 0
            vote_history.clear()
            print("  ⏱️  Zone released (timeout)")

        update_gui(displayed_zone)

        conf_str = "✅" if top_confidence >= CONFIDENCE_THRESHOLD else "❓"
        print(
            f"[{packets_seen:5d} pkts] {conf_str} "
            f"Raw:{zones[predicted_zone][4]:<12} conf:{top_confidence*100:4.1f}% | "
            f"E:{probabilities[0]*100:4.1f}% "
            f"A:{probabilities[1]*100:4.1f}% "
            f"B:{probabilities[2]*100:4.1f}% "
            f"C:{probabilities[3]*100:4.1f}% | "
            f"Display:{zones[displayed_zone][4]}"
        )

        if len(data_buffer) > WINDOW_SIZE * 4:
            data_buffer = data_buffer[-WINDOW_SIZE:]

except KeyboardInterrupt:
    print(f"\n🛑 Shut down. Total packets: {packets_seen}")
    ser.close()
    plt.close()
