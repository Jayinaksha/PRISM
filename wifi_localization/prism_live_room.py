import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt
import joblib
import time

SERIAL_PORT  = '/dev/ttyUSB0'
BAUD_RATE    = 115200
WINDOW_SIZE  = 100
SLIDE_STEP   = 20
SMOOTHING_VOTES = 3

# Confidence threshold: model must be THIS sure to switch zones.
# If the top prediction is below this, treat it as "uncertain" (show empty).
# Your logs show Zone A winning at ~35% — raise this to force the model
# to only commit when it's clearly dominant.
CONFIDENCE_THRESHOLD = 0.40   # 40% — tune up if too jumpy, down if too sticky

# Timeout: if we haven't seen a confident prediction for this many seconds,
# release the current zone back to "empty" instead of freezing.
ZONE_RELEASE_TIMEOUT = 4.0    # seconds

try:
    clf = joblib.load("prism_model.pkl")
    print("🧠 PRISM AI Model Loaded Successfully!")
except FileNotFoundError:
    print("❌ ERROR: 'prism_model.pkl' not found. Run prism_ai.py first!")
    exit()

def apply_live_dsp(data_buffer, fs=100):
    df = pd.DataFrame(data_buffer)
    rolling_mean = df.rolling(window=100, min_periods=1).mean()
    dyn_data = (df - rolling_mean).values
    nyq = 0.5 * fs
    b, a = butter(3, [0.1 / nyq, 3.0 / nyq], btype='band')
    if dyn_data.shape[0] <= 15:
        return dyn_data
    return filtfilt(b, a, dyn_data, axis=0)

def extract_features(window):
    var      = np.var(window, axis=0)
    std      = np.std(window, axis=0)
    energy   = np.mean(window ** 2, axis=0)
    diff_var = np.var(np.diff(window, axis=1), axis=0)
    return np.nan_to_num(
        np.concatenate([var, std, energy, diff_var]),
        nan=0.0, posinf=0.0, neginf=0.0
    ).reshape(1, -1)

# Must match NULL_SUBCARRIERS in prism_ai.py exactly
NULL_SUBCARRIERS = list(range(27, 38))
ACTIVE_COLS = [i for i in range(64) if i not in NULL_SUBCARRIERS]  # 53 active

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
    return amps[ACTIVE_COLS]   # drop null subcarriers — must match training

plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.manager.set_window_title('Project PRISM: Live Radar')
plt.style.use('dark_background')
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_title("Project PRISM: Live Wi-Fi Zonal Tracking", fontsize=18, fontweight='bold', color='white')
ax.axis('off')

zones = {
    0: [0,   0,   10, 6, "EMPTY ROOM",      "gray"],
    1: [0.5, 0.5,  3, 5, "Zone A (Door)",   "cyan"],
    2: [4.0, 0.5,  2, 5, "Zone B (Center)", "orange"],
    3: [6.5, 0.5,  3, 5, "Zone C (Window)", "red"],
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

status_banner = ax.text(5, 5.5, "STATUS: WARMING UP...", color="orange", fontsize=16, weight='bold',
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

print(f"📡 Opening {SERIAL_PORT}...")
print("⚠️  Make sure idf.py monitor is NOT running (Ctrl+] to close it)")
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

        if len(data_buffer) % SLIDE_STEP != 0:
            continue

        window = np.array(data_buffer[-WINDOW_SIZE:])
        clean  = apply_live_dsp(window)
        fp     = extract_features(clean)

        probabilities  = clf.predict_proba(fp)[0]
        predicted_zone = int(np.argmax(probabilities))
        top_confidence = probabilities[predicted_zone]

        # Only vote if the model is confident enough — below threshold
        # counts as "uncertain" which nudges the display back toward empty
        if top_confidence >= CONFIDENCE_THRESHOLD:
            vote = predicted_zone
            last_confident_time = time.time()
        else:
            vote = 0   # uncertain → treat as empty

        vote_history.append(vote)
        if len(vote_history) > SMOOTHING_VOTES:
            vote_history.pop(0)

        # Switch display only when last N votes all agree
        if len(vote_history) == SMOOTHING_VOTES and len(set(vote_history)) == 1:
            new_zone = vote_history[0]
            if new_zone != displayed_zone:
                displayed_zone = new_zone
                update_gui(displayed_zone)

        # Timeout: release zone if nothing confident for a while
        if displayed_zone != 0 and (time.time() - last_confident_time) > ZONE_RELEASE_TIMEOUT:
            displayed_zone = 0
            vote_history.clear()
            update_gui(0)
            print("  ⏱️  Zone released (timeout — no confident prediction)")

        conf_str = "✅" if top_confidence >= CONFIDENCE_THRESHOLD else "❓"
        print(
            f"[{packets_seen:5d} pkts] {conf_str} "
            f"Raw:{zones[predicted_zone][4]:<18} conf:{top_confidence*100:4.1f}% | "
            f"Empty:{probabilities[0]*100:4.1f}% "
            f"ZoneA:{probabilities[1]*100:4.1f}% "
            f"ZoneB:{probabilities[2]*100:4.1f}% "
            f"ZoneC:{probabilities[3]*100:4.1f}% | "
            f"Display:{zones[displayed_zone][4]}"
        )

        if len(data_buffer) > WINDOW_SIZE * 4:
            data_buffer = data_buffer[-WINDOW_SIZE:]

except KeyboardInterrupt:
    print(f"\n🛑 Shut down. Total packets: {packets_seen}")
    ser.close()
    plt.close()
