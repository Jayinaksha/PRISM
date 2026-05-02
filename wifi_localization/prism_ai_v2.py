import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

DATA_DIR = os.path.expanduser('~/sem6/RF/wifi_localization/data')
IMG_DIR  = os.path.expanduser('~/sem6/RF/wifi_localization/images')
os.makedirs(IMG_DIR, exist_ok=True)

DATASET_FILES = {
    "empty_area.csv":  0, "empty_area2.csv": 0, "empty_area3.csv": 0,
    "zone_a.csv":      1, "zone_a2.csv":     1, "zone_a3.csv":     1,
    "zone_b.csv":      2, "zone_b2.csv":     2, "zone_b3.csv":     2,
}
CLASS_NAMES = ["Empty", "Zone A", "Zone B"]

WINDOW_SIZE = 100
STEP_SIZE   = 20
NULL_SUBCARRIERS = list(range(27, 38))

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
# FEATURE EXTRACTION
# ==========================================

FEATURE_NAMES = []

def _agg(v, prefix, names, build):
    agg = [np.mean(v), np.std(v), np.min(v), np.max(v), np.median(v)]
    if build:
        for s in ['mean','std','min','max','med']:
            names.append(f"{prefix}_{s}")
    return agg

def extract_features(w):
    global FEATURE_NAMES
    build = len(FEATURE_NAMES) == 0
    f = []

    # Aggregated features (low-dim, robust)
    f.extend(_agg(np.var(w, axis=0), "var", FEATURE_NAMES, build))
    f.extend(_agg(np.std(w, axis=0), "std", FEATURE_NAMES, build))
    f.extend(_agg(np.mean(w**2, axis=0), "energy", FEATURE_NAMES, build))
    f.extend(_agg(np.var(np.diff(w, axis=1), axis=0), "diffvar", FEATURE_NAMES, build))
    f.extend(_agg(skew(w, axis=0), "skew", FEATURE_NAMES, build))
    f.extend(_agg(kurtosis(w, axis=0), "kurt", FEATURE_NAMES, build))
    f.extend(_agg(np.percentile(w,75,0)-np.percentile(w,25,0), "iqr", FEATURE_NAMES, build))
    f.extend(_agg(np.max(w,0)-np.min(w,0), "range", FEATURE_NAMES, build))

    # Temporal
    w0, w1 = w[:-1,:], w[1:,:]
    w0m, w1m = w0-w0.mean(0), w1-w1.mean(0)
    ac = np.sum(w0m*w1m,0)/(np.sqrt(np.sum(w0m**2,0)*np.sum(w1m**2,0))+1e-10)
    f.extend(_agg(ac, "autocorr", FEATURE_NAMES, build))
    f.extend(_agg(np.var(np.diff(w,axis=0),axis=0), "tdiffvar", FEATURE_NAMES, build))

    # Frequency
    fft_m = np.abs(np.fft.rfft(w, axis=0))
    freqs = np.fft.rfftfreq(w.shape[0], d=0.01)
    f.extend(_agg(freqs[np.argmax(fft_m[1:,:],0)+1], "fftpeak", FEATURE_NAMES, build))
    fft_p = fft_m[1:,:]**2; fa = freqs[1:]
    f.extend(_agg(np.sum(fft_p*fa[:,None],0)/(np.sum(fft_p,0)+1e-10), "spectcent", FEATURE_NAMES, build))

    # Eigenvalues
    try:
        cm = np.nan_to_num(np.corrcoef(w.T), nan=0.0)
        eig = np.sort(np.linalg.eigvalsh(cm))[::-1][:5]
    except:
        eig = np.zeros(5)
    f.extend(eig.tolist())
    if build:
        FEATURE_NAMES.extend([f"eig{i}" for i in range(5)])

    # Global
    f.append(np.sum(w**2))
    sp = np.mean(w**2,0); sp = sp/(sp.sum()+1e-10)
    f.append(-np.sum(sp*np.log(sp+1e-10)))
    if build:
        FEATURE_NAMES.extend(["total_energy","sc_entropy"])

    # --- KEY ADDITION: per-subcarrier variance (the original v1 feature) ---
    # This was the ONLY feature in the original prism_ai_prev.py and it worked at 65%
    # Include variance of top-10 most variable subcarriers
    var_per_sc = np.var(w, axis=0)
    top10_idx = np.argsort(var_per_sc)[::-1][:10]
    f.extend(var_per_sc[top10_idx].tolist())
    if build:
        FEATURE_NAMES.extend([f"topvar_{i}" for i in range(10)])

    # Per-subcarrier energy of top-10
    energy_per_sc = np.mean(w**2, axis=0)
    f.extend(energy_per_sc[top10_idx].tolist())
    if build:
        FEATURE_NAMES.extend([f"topenergy_{i}" for i in range(10)])

    return np.nan_to_num(np.array(f), nan=0.0, posinf=0.0, neginf=0.0)

# ==========================================
# DATASET BUILDER
# ==========================================

def build_dataset():
    X, y, file_ids = [], [], []
    fid = 0
    for fname, label in DATASET_FILES.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"   ❌ {fname} not found"); fid += 1; continue
        raw = pd.read_csv(fpath).values
        active = [i for i in range(raw.shape[1]) if i not in NULL_SUBCARRIERS]
        raw = raw[:, active]
        clean = apply_dsp(raw)
        nw = 0
        for i in range(0, len(clean)-WINDOW_SIZE+1, STEP_SIZE):
            w = clean[i:i+WINDOW_SIZE]
            X.append(extract_features(w))
            y.append(label)
            file_ids.append(fid)
            nw += 1
        print(f"   {fname}: {len(raw)} rows, {nw} windows (class={label})", flush=True)
        fid += 1
    return np.array(X), np.array(y), np.array(file_ids)

# ==========================================
# MODELS TO TRY
# ==========================================

MODELS = {
    "RandomForest": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1))
    ]),
    "SVM-RBF": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=50, gamma='scale',
                    class_weight='balanced', probability=True, random_state=42))
    ]),
    "GradientBoosting": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=10, random_state=42))
    ]),
}

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("  🚀 PRISM AI v2 — Model Comparison")
    print("=" * 60)

    print(f"\n📂 Loading data...")
    X, y, file_ids = build_dataset()
    print(f"\n   Shape: {X.shape}, Features: {len(FEATURE_NAMES)}")
    print(f"   Balance: {dict(zip(*np.unique(y, return_counts=True)))}")

    # --- Stratified 5-Fold CV (fair: every fold has all classes) ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_name, best_acc = None, 0
    best_true, best_pred = None, None

    for model_name, model_template in MODELS.items():
        print(f"\n🔬 Testing {model_name}...")
        all_true, all_pred = [], []

        for fold, (tr, te) in enumerate(skf.split(X, y), 1):
            clf = Pipeline([(n, s) for n, s in model_template.steps])  # fresh copy
            # Have to clone properly
            from sklearn.base import clone
            clf = clone(model_template)
            clf.fit(X[tr], y[tr])
            p = clf.predict(X[te])
            all_pred.extend(p)
            all_true.extend(y[te])

        all_true, all_pred = np.array(all_true), np.array(all_pred)
        acc = accuracy_score(all_true, all_pred)

        # Zone A↔B confusion
        mask = (all_true==1)|(all_true==2)
        ab_confused = ((all_true[mask]==1)&(all_pred[mask]==2)).sum() + \
                      ((all_true[mask]==2)&(all_pred[mask]==1)).sum()

        print(f"   Accuracy: {acc*100:.1f}%  |  A↔B confusion: {ab_confused}/{mask.sum()} ({ab_confused/max(mask.sum(),1)*100:.1f}%)")

        if acc > best_acc:
            best_acc = acc
            best_name = model_name
            best_true, best_pred = all_true, all_pred

    # --- Report best model ---
    print("\n" + "=" * 60)
    print(f"  🏆 Best Model: {best_name} — {best_acc*100:.1f}%")
    print("=" * 60)

    fp = ((best_true==0)&(best_pred!=0)).sum()
    fn = ((best_true!=0)&(best_pred==0)).sum()
    t_e = (best_true==0).sum()
    t_o = (best_true!=0).sum()
    mask = (best_true==1)|(best_true==2)
    ab_confused = ((best_true[mask]==1)&(best_pred[mask]==2)).sum() + \
                  ((best_true[mask]==2)&(best_pred[mask]==1)).sum()

    print(f"  🏠 Empty accuracy       : {(t_e-fp)}/{t_e} ({(1-fp/max(t_e,1))*100:.1f}%)")
    print(f"  🚨 False Positives      : {fp}/{t_e} ({fp/max(t_e,1)*100:.1f}%)")
    print(f"  👻 Missed Detections    : {fn}/{t_o} ({fn/max(t_o,1)*100:.1f}%)")
    print(f"  🔀 Zone A↔B Confusion   : {ab_confused}/{mask.sum()} ({ab_confused/max(mask.sum(),1)*100:.1f}%)")

    print(f"\n📊 Per-class Report ({best_name}):")
    print(classification_report(best_true, best_pred, target_names=CLASS_NAMES))

    print("🗺️  Confusion Matrix:")
    cm = confusion_matrix(best_true, best_pred)
    df_cm = pd.DataFrame(cm, index=[f"Act:{n}" for n in CLASS_NAMES],
                         columns=[f"Pred:{n}" for n in CLASS_NAMES])
    print(df_cm.to_string())

    # --- Train + save best model ---
    print(f"\n🧠 Training {best_name} on full dataset...")
    from sklearn.base import clone
    final_clf = clone(MODELS[best_name])
    final_clf.fit(X, y)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prism_model_v2.pkl")
    joblib.dump(final_clf, model_path)
    print(f"   💾 Saved → {model_path}")

    # --- Feature importance ---
    print("\n📈 Feature importance...")
    inner_clf = final_clf.named_steps['clf']
    if hasattr(inner_clf, 'feature_importances_'):
        imp = inner_clf.feature_importances_
        names = FEATURE_NAMES if len(FEATURE_NAMES)==len(imp) else [str(i) for i in range(len(imp))]
        top_n = min(30, len(imp))
        idx = np.argsort(imp)[::-1][:top_n]
        plt.figure(figsize=(12,7))
        plt.barh(range(top_n), imp[idx][::-1], color='cyan', edgecolor='white', linewidth=0.3)
        plt.yticks(range(top_n), [names[i] for i in idx][::-1], fontsize=9)
        plt.xlabel('Feature Importance')
        plt.title(f'PRISM v2 — {best_name} Feature Importance', fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(IMG_DIR, "feature_importance_v2.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 {save_path}")

    print("\n✅ Done.")
    print("=" * 60)
