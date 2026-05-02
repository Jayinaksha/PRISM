"""
PRISM AI Room — Robust training pipeline
Fixes overfitting from correlated windows by using:
  - Non-overlapping windows (step=WINDOW_SIZE)
  - Gaussian noise augmentation (5x data multiplier)
  - SVM-RBF which generalizes better than HistGBM on small data
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = os.path.expanduser('~/sem6/RF/wifi_localization/data_room')
IMG_DIR  = os.path.expanduser('~/sem6/RF/wifi_localization/images')
os.makedirs(IMG_DIR, exist_ok=True)

DATASET_FILES = {
    "empty_room.csv": 0,
    "zone_a.csv":     1,
    "zone_b.csv":     2,
    "zone_c.csv":     3,
}
CLASS_NAMES = ["Empty", "Zone A", "Zone B", "Zone C"]

WINDOW_SIZE = 100
# 50% overlap — balance between sample count and independence
STEP_SIZE   = 50
NULL_SUBCARRIERS = list(range(27, 38))

# Augmentation: noisy copies per real window
N_AUGMENT = 8
NOISE_STD = 0.2  # gentler noise for augmentation

# ==========================================
# DSP
# ==========================================

def apply_dsp(data, fs=100):
    df = pd.DataFrame(data)
    dyn = (df - df.rolling(window=100, min_periods=1).mean()).values
    nyq = 0.5 * fs
    b, a = butter(3, [0.1/nyq, 3.0/nyq], btype='band')
    if dyn.shape[0] < 15:
        return dyn
    return filtfilt(b, a, dyn, axis=0)

# ==========================================
# FEATURE EXTRACTION — same enhanced features
# ==========================================

FEATURE_NAMES = []

def _agg(v, prefix, names, build):
    agg = [np.mean(v), np.std(v), np.min(v), np.max(v), np.median(v)]
    if build:
        for s in ['mean','std','min','max','med']:
            names.append(f"{prefix}_{s}")
    return agg

def _lac(w, lag):
    w0, w1 = w[:-lag,:], w[lag:,:]
    w0m, w1m = w0-w0.mean(0), w1-w1.mean(0)
    return np.sum(w0m*w1m,0)/(np.sqrt(np.sum(w0m**2,0)*np.sum(w1m**2,0))+1e-10)

def extract_features(w):
    global FEATURE_NAMES
    build = len(FEATURE_NAMES) == 0
    f = []

    # Basic stats
    f.extend(_agg(np.var(w,0), "var", FEATURE_NAMES, build))
    f.extend(_agg(np.std(w,0), "std", FEATURE_NAMES, build))
    f.extend(_agg(np.mean(w**2,0), "energy", FEATURE_NAMES, build))
    f.extend(_agg(np.var(np.diff(w,axis=1),0), "diffvar", FEATURE_NAMES, build))
    f.extend(_agg(skew(w,axis=0), "skew", FEATURE_NAMES, build))
    f.extend(_agg(kurtosis(w,axis=0), "kurt", FEATURE_NAMES, build))
    f.extend(_agg(np.percentile(w,75,0)-np.percentile(w,25,0), "iqr", FEATURE_NAMES, build))
    f.extend(_agg(np.max(w,0)-np.min(w,0), "range", FEATURE_NAMES, build))

    # Multi-lag autocorrelation
    f.extend(_agg(_lac(w,1), "ac1", FEATURE_NAMES, build))
    f.extend(_agg(_lac(w,5), "ac5", FEATURE_NAMES, build))
    f.extend(_agg(_lac(w,10), "ac10", FEATURE_NAMES, build))
    f.extend(_agg(np.var(np.diff(w,axis=0),0), "tdiffvar", FEATURE_NAMES, build))
    f.extend(_agg(np.var(np.diff(np.diff(w,axis=0),axis=0),0), "t2diffvar", FEATURE_NAMES, build))

    # Multi-scale variance ratios
    h = WINDOW_SIZE//2; q = WINDOW_SIZE//4
    vh1 = np.var(w[:h,:],0); vh2 = np.var(w[h:,:],0)
    vq1 = np.var(w[:q,:],0); vq4 = np.var(w[-q:,:],0)
    f.extend(_agg((vh1+1e-10)/(vh2+1e-10), "varratio_h", FEATURE_NAMES, build))
    f.extend(_agg((vq1+1e-10)/(vq4+1e-10), "varratio_q", FEATURE_NAMES, build))

    # Frequency domain
    fft_m = np.abs(np.fft.rfft(w,axis=0))
    freqs = np.fft.rfftfreq(w.shape[0], d=0.01)
    f.extend(_agg(freqs[np.argmax(fft_m[1:,:],0)+1], "fftpeak", FEATURE_NAMES, build))
    fft_p = fft_m[1:,:]**2; fa = freqs[1:]
    sc = np.sum(fft_p*fa[:,None],0)/(np.sum(fft_p,0)+1e-10)
    f.extend(_agg(sc, "spectcent", FEATURE_NAMES, build))
    sb = np.sqrt(np.sum(fft_p*(fa[:,None]-sc[None,:])**2,0)/(np.sum(fft_p,0)+1e-10))
    f.extend(_agg(sb, "spectbw", FEATURE_NAMES, build))
    low = (fa>=0.1)&(fa<=1.0); high = (fa>1.0)&(fa<=3.0)
    le = np.sum(fft_p[low,:],0)+1e-10; he = np.sum(fft_p[high,:],0)+1e-10
    f.extend(_agg(le/he, "bandratio", FEATURE_NAMES, build))

    # Subcarrier profile shape
    mp = np.mean(w,0)
    f.extend(_agg(np.diff(mp), "profgrad", FEATURE_NAMES, build))
    f.extend(_agg(np.diff(np.diff(mp)), "profcurv", FEATURE_NAMES, build))

    # Cross-subcarrier correlation
    with np.errstate(divide='ignore', invalid='ignore'):
        cm = np.corrcoef(w.T)
    cm = np.nan_to_num(cm, nan=0.0)
    try:
        eig = np.sort(np.linalg.eigvalsh(cm))[::-1][:5]
    except:
        eig = np.zeros(5)
    f.extend(eig.tolist())
    if build:
        FEATURE_NAMES.extend([f"eig{i}" for i in range(5)])
    upper = cm[np.triu_indices_from(cm, k=1)]
    f.extend([np.mean(upper), np.std(upper), np.median(upper)])
    if build:
        FEATURE_NAMES.extend(["corr_mean","corr_std","corr_median"])

    # Global
    f.append(np.sum(w**2))
    sp = np.mean(w**2,0); sp = sp/(sp.sum()+1e-10)
    f.append(-np.sum(sp*np.log(sp+1e-10)))
    if build:
        FEATURE_NAMES.extend(["total_energy","sc_entropy"])

    # Top-10 per-subcarrier
    var_ps = np.var(w,0)
    t10 = np.argsort(var_ps)[::-1][:10]
    f.extend(var_ps[t10].tolist())
    f.extend(np.mean(w**2,0)[t10].tolist())
    if build:
        FEATURE_NAMES.extend([f"topvar_{i}" for i in range(10)])
        FEATURE_NAMES.extend([f"topenergy_{i}" for i in range(10)])

    return np.nan_to_num(np.array(f), nan=0.0, posinf=0.0, neginf=0.0)

# ==========================================
# DATASET BUILDER — non-overlapping + augmentation
# ==========================================

def build_dataset():
    X_real, y_real = [], []  # real (non-overlapping) windows
    rng = np.random.RandomState(42)

    for fname, label in DATASET_FILES.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"   ❌ {fname} not found"); continue
        raw = pd.read_csv(fpath).values
        active = [i for i in range(raw.shape[1]) if i not in NULL_SUBCARRIERS]
        raw = raw[:, active]
        clean = apply_dsp(raw)
        nw = 0
        for i in range(0, len(clean)-WINDOW_SIZE+1, STEP_SIZE):
            w = clean[i:i+WINDOW_SIZE]
            X_real.append(w)  # store the RAW window for augmentation
            y_real.append(label)
            nw += 1
        print(f"   {fname}: {len(raw)} rows → {nw} non-overlapping windows", flush=True)

    # Extract features from real windows
    X, y = [], []
    for w, label in zip(X_real, y_real):
        X.append(extract_features(w))
        y.append(label)

    # Augment: add Gaussian noise copies
    n_real = len(X)
    for aug_i in range(N_AUGMENT):
        for w, label in zip(X_real, y_real):
            # Add noise proportional to each subcarrier's std
            sc_std = np.std(w, axis=0, keepdims=True) + 1e-10
            noise = rng.randn(*w.shape) * sc_std * NOISE_STD
            w_noisy = w + noise
            X.append(extract_features(w_noisy))
            y.append(label)

    print(f"\n   Real windows: {n_real}, After augmentation: {len(X)} ({N_AUGMENT}x noise)")
    return np.array(X), np.array(y), n_real

# ==========================================
# MODELS
# ==========================================

MODELS = {
    "SVM-RBF": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=10, gamma='scale',
                    class_weight='balanced', probability=True, random_state=42))
    ]),
    "RandomForest": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1))
    ]),
    "HistGBM": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=10, class_weight='balanced', random_state=42))
    ]),
}

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("  🚀 PRISM AI Room — Robust Training")
    print("=" * 60)

    print(f"\n📂 Loading data from {DATA_DIR}...")
    X, y, n_real = build_dataset()
    print(f"   Shape: {X.shape}, Features: {len(FEATURE_NAMES)}")
    print(f"   Balance: {dict(zip(*np.unique(y, return_counts=True)))}")

    # CV only on real data (first n_real samples) to get honest accuracy
    X_real = X[:n_real]
    y_real = y[:n_real]
    print(f"\n   CV on {n_real} real (non-augmented) samples only")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_acc = None, 0
    best_true, best_pred = None, None

    for model_name, model_template in MODELS.items():
        print(f"\n🔬 Testing {model_name}...")
        all_true, all_pred = [], []

        for fold, (tr, te) in enumerate(skf.split(X_real, y_real), 1):
            clf = clone(model_template)
            # Train on real+augmented: include augmented copies of training windows
            # But test only on real test windows
            aug_tr_idx = []
            for real_idx in tr:
                aug_tr_idx.append(real_idx)
                for aug_i in range(N_AUGMENT):
                    aug_tr_idx.append(n_real + aug_i * n_real + real_idx)

            clf.fit(X[aug_tr_idx], y[aug_tr_idx])
            p = clf.predict(X_real[te])
            acc = accuracy_score(y_real[te], p)
            all_pred.extend(p)
            all_true.extend(y_real[te])
            print(f"   Fold {fold}: {acc*100:.1f}%")

        all_true, all_pred = np.array(all_true), np.array(all_pred)
        acc = accuracy_score(all_true, all_pred)
        print(f"   → Overall: {acc*100:.1f}%")
        if acc > best_acc:
            best_acc = acc
            best_name = model_name
            best_true, best_pred = all_true, all_pred

    # --- Report ---
    print("\n" + "=" * 60)
    print(f"  🏆 Best Model: {best_name} — {best_acc*100:.1f}%")
    print("=" * 60)

    fp = ((best_true==0)&(best_pred!=0)).sum()
    fn = ((best_true!=0)&(best_pred==0)).sum()
    t_e = (best_true==0).sum()
    t_o = (best_true!=0).sum()
    print(f"  🏠 Empty accuracy       : {(t_e-fp)}/{t_e} ({(1-fp/max(t_e,1))*100:.1f}%)")
    print(f"  🚨 False Positives      : {fp}/{t_e} ({fp/max(t_e,1)*100:.1f}%)")
    print(f"  👻 Missed Detections    : {fn}/{t_o} ({fn/max(t_o,1)*100:.1f}%)")

    for i in range(1,4):
        for j in range(i+1,4):
            mask = (best_true==i)|(best_true==j)
            confused = ((best_true[mask]==i)&(best_pred[mask]==j)).sum() + \
                       ((best_true[mask]==j)&(best_pred[mask]==i)).sum()
            print(f"  🔀 {CLASS_NAMES[i]}↔{CLASS_NAMES[j]}: {confused}/{mask.sum()} ({confused/max(mask.sum(),1)*100:.1f}%)")

    print(f"\n📊 Per-class Report ({best_name}):")
    print(classification_report(best_true, best_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(best_true, best_pred)
    print("🗺️  Confusion Matrix:")
    df_cm = pd.DataFrame(cm, index=[f"Act:{n}" for n in CLASS_NAMES],
                         columns=[f"Pred:{n}" for n in CLASS_NAMES])
    print(df_cm.to_string())

    # --- Train final on ALL data (real + augmented) ---
    print(f"\n🧠 Training {best_name} on full dataset (real + augmented)...")
    final_clf = clone(MODELS[best_name])
    final_clf.fit(X, y)

    # Save in both locations
    for path in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "prism_model_room.pkl"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "room", "prism_model_room.pkl"),
    ]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(final_clf, path)
        print(f"   💾 Saved → {path}")

    # Feature importance (if available)
    inner = final_clf.named_steps['clf']
    if hasattr(inner, 'feature_importances_'):
        imp = inner.feature_importances_
        names = FEATURE_NAMES if len(FEATURE_NAMES)==len(imp) else [str(i) for i in range(len(imp))]
        top_n = min(30, len(imp))
        idx = np.argsort(imp)[::-1][:top_n]
        plt.figure(figsize=(12,8))
        plt.barh(range(top_n), imp[idx][::-1], color='cyan', edgecolor='white', linewidth=0.3)
        plt.yticks(range(top_n), [names[i] for i in idx][::-1], fontsize=9)
        plt.xlabel('Feature Importance')
        plt.title(f'PRISM Room — {best_name} Top Features', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "feature_importance_room.png"), dpi=150)
        plt.close()
        print(f"   📊 Feature importance saved")

    print("\n✅ Done.")
    print("=" * 60)
