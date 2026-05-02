import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.expanduser('~/sem6/RF/wifi_localization/data')

DATASET_FILES = {
    "empty_room.csv": 0,
    "zone_a.csv":     1,
    "zone_b.csv":     2,
    "zone_c.csv":     3,
}
CLASS_NAMES = ["Empty Room", "Zone A", "Zone B", "Zone C"]

WINDOW_SIZE = 100
STEP_SIZE   = 10

# Subcarriers 27-37 are null/pilot in LLTF mode — always zero, drop them
NULL_SUBCARRIERS = list(range(27, 38))

# Data quality gates
EMPTY_MAX_VAR = 10.0   # empty chunks above this = someone walked in, drop
ZONE_MIN_VAR  = 5.0    # zone chunks below this = nobody was there, drop

def apply_dsp(data, fs=100):
    df = pd.DataFrame(data)
    dyn_data = (df - df.rolling(window=100, min_periods=1).mean()).values
    nyq = 0.5 * fs
    b, a = butter(3, [0.1 / nyq, 3.0 / nyq], btype='band')
    if dyn_data.shape[0] < 15:
        return dyn_data
    return filtfilt(b, a, dyn_data, axis=0)

def load_and_clean(filepath, label):
    if not os.path.exists(filepath):
        print(f"   ❌ File not found: {filepath}")
        return []
    raw = pd.read_csv(filepath).values
    active_cols = [i for i in range(raw.shape[1]) if i not in NULL_SUBCARRIERS]
    raw = raw[:, active_cols]
    kept, dropped = [], 0
    for i in range(0, len(raw) - WINDOW_SIZE + 1, WINDOW_SIZE):
        chunk = raw[i:i+WINDOW_SIZE]
        if len(chunk) < WINDOW_SIZE:
            continue
        v = np.var(apply_dsp(chunk), axis=0).mean()
        if label == 0:
            if v <= EMPTY_MAX_VAR:
                kept.append(chunk)
            else:
                dropped += 1
        else:
            if v >= ZONE_MIN_VAR:
                kept.append(chunk)
            else:
                dropped += 1
    print(f"   {os.path.basename(filepath)}: {len(kept)} clean chunks kept, {dropped} dropped")
    return kept

def extract_features(window):
    var      = np.var(window, axis=0)
    std      = np.std(window, axis=0)
    energy   = np.mean(window ** 2, axis=0)
    diff_var = np.var(np.diff(window, axis=1), axis=0)
    return np.nan_to_num(np.concatenate([var, std, energy, diff_var]),
                         nan=0.0, posinf=0.0, neginf=0.0)

def build_dataset(chunks_per_class):
    X, y, groups = [], [], []
    for label, chunks in chunks_per_class.items():
        for chunk_id, chunk in enumerate(chunks):
            clean = apply_dsp(chunk)
            for i in range(0, len(clean) - WINDOW_SIZE + 1, STEP_SIZE):
                w = clean[i:i+WINDOW_SIZE]
                if len(w) == WINDOW_SIZE:
                    X.append(extract_features(w))
                    y.append(label)
                    groups.append(chunk_id * 10 + label)
    return np.array(X), np.array(y), np.array(groups)

def build_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=50, gamma='scale',
                    class_weight='balanced', probability=True, random_state=42))
    ])

if __name__ == "__main__":
    print("=" * 57)
    print("  🚀 PRISM AI — Training Pipeline")
    print("=" * 57)

    print(f"\n📂 Loading and cleaning datasets...")
    print(f"   Null subcarriers dropped  : {NULL_SUBCARRIERS}")
    print(f"   Empty room max variance   : {EMPTY_MAX_VAR}")
    print(f"   Zone min variance         : {ZONE_MIN_VAR}\n")

    chunks_per_class = {}
    for filename, label in DATASET_FILES.items():
        chunks = load_and_clean(os.path.join(DATA_DIR, filename), label)
        if chunks:
            chunks_per_class[label] = chunks

    if len(chunks_per_class) < 4:
        print("\n❌ CRITICAL: Missing classes. Cannot train.")
        exit()

    print(f"\n⚙️  Extracting features (window={WINDOW_SIZE}, step={STEP_SIZE})...")
    X, y, groups = build_dataset(chunks_per_class)
    print(f"   Dataset shape  : {X.shape}")
    print(f"   Class balance  : {np.bincount(y)} windows per class")

    print("\n🔍 Leave-One-Chunk-Out CV (honest evaluation)...")
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    for tr_idx, te_idx in logo.split(X, y, groups):
        clf = build_model()
        clf.fit(X[tr_idx], y[tr_idx])
        all_pred.extend(clf.predict(X[te_idx]))
        all_true.extend(y[te_idx])

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    fp  = ((all_true == 0) & (all_pred != 0)).sum()
    fn  = ((all_true != 0) & (all_pred == 0)).sum()
    t_e = (all_true == 0).sum()
    t_o = (all_true != 0).sum()

    print("\n" + "=" * 57)
    print(f"  🎯 CV Accuracy         : {accuracy_score(all_true, all_pred)*100:.1f}%")
    print(f"  🚨 False Positives     : {fp}/{t_e} ({fp/max(t_e,1)*100:.1f}%)  [empty→occupied]")
    print(f"  👻 Missed Detections   : {fn}/{t_o} ({fn/max(t_o,1)*100:.1f}%)  [occupied→empty]")
    print("=" * 57)

    print("\n📊 Per-class Report:")
    print(classification_report(all_true, all_pred, target_names=CLASS_NAMES))

    print("🗺️  Confusion Matrix:")
    cm = confusion_matrix(all_true, all_pred)
    df_cm = pd.DataFrame(cm,
                         index=[f"Actual: {n}" for n in CLASS_NAMES],
                         columns=[f"Pred: {n[:8]}" for n in CLASS_NAMES])
    print(df_cm.to_string())

    print("\n🧠 Training final model on full dataset...")
    final_clf = build_model()
    final_clf.fit(X, y)
    joblib.dump(final_clf, "prism_model.pkl")
    print(f"💾 Model saved → prism_model.pkl")
    print("\n✅ Done. Run prism_live_room.py to start the radar.")
    print("=" * 57)
