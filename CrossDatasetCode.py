# =============================================================================
# cross_dataset_final.py
# =============================================================================
# CROSS‑DATASET EVALUATION: Train on CIC-IDS-2017, Test on CIC-IDS-2018
#
# Solves:
#   1. Concept drift in network intrusion detection
#   2. Feature stability across years (top-k selection)
#   3. Explainability via per‑sample SHAP
# =============================================================================

import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠ Install imbalanced-learn: pip install imbalanced-learn")

warnings.filterwarnings("ignore")

# =============================================================================
# ── CONFIG – EDIT THESE PATHS ──────────────────────────────────────────────
# =============================================================================
CIC2017_CSV = "/content/D:\ml\CIC2017_combined.csv"
CIC2018_CSV = "/content/cic.csv"
OUTPUT_DIR  = "./Cross.py"

SAMPLE_SIZE       = 500000        # use more for better generalisation
RANDOM_STATE      = 42
MIN_CLASS_SAMPLES = 1000
MAX_CLASS_SAMPLES = 400000
SHAP_ROWS         = 200
SHAP_SAMPLES      = 100
TOP_K             = 25             # number of top features to keep
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

ATTACK_CATEGORIES = ["Normal", "DDoS/DoS", "Brute Force", "Botnet", "Web Attack", "PortScan"]

ORIGINAL_FEATURES = [
    "Dst Port","Protocol","Flow Duration",
    "Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts",
    "Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std",
    "Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std",
    "Flow Byts/s","Flow Pkts/s",
    "Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
    "Bwd IAT Tot","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
    "Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags",
    "Fwd Header Len","Bwd Header Len","Fwd Pkts/s","Bwd Pkts/s",
    "Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var",
    "FIN Flag Cnt","SYN Flag Cnt","RST Flag Cnt","PSH Flag Cnt","ACK Flag Cnt",
    "URG Flag Cnt","CWE Flag Count","ECE Flag Cnt","Down/Up Ratio",
    "Pkt Size Avg","Fwd Seg Size Avg","Bwd Seg Size Avg",
    "Fwd Byts/b Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg",
    "Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg",
    "Subflow Fwd Pkts","Subflow Fwd Byts","Subflow Bwd Pkts","Subflow Bwd Byts",
    "Init Fwd Win Byts","Init Bwd Win Byts","Fwd Act Data Pkts","Fwd Seg Size Min",
    "Active Mean","Active Std","Active Max","Active Min",
    "Idle Mean","Idle Std","Idle Max","Idle Min",
]

# =============================================================================
# LABEL MAPPERS
# =============================================================================
def standardize_label_2017(raw):
    lbl = str(raw).strip().upper()
    if lbl == "BENIGN": return "Normal"
    for kw in ["DDOS","DOS HULK","DOS GOLDENEYE","DOS SLOWLORIS","DOS SLOWHTTPTEST","HEARTBLEED","DOS"]:
        if kw in lbl: return "DDoS/DoS"
    if any(k in lbl for k in ["BRUTE FORCE","FTP-PATATOR","SSH-PATATOR"]): return "Brute Force"
    if "BOT" in lbl: return "Botnet"
    if any(k in lbl for k in ["WEB ATTACK","XSS","SQL INJECTION","INFILTRATION"]): return "Web Attack"
    if "PORT" in lbl: return "PortScan"
    return "Other"

def standardize_label_2018(raw):
    lbl = str(raw).strip().upper()
    if lbl == "BENIGN": return "Normal"
    for kw in ["DDOS","DOS","HULK","GOLDENEYE","SLOWLORIS","SLOWHTTP","LOIC","HOIC"]:
        if kw in lbl: return "DDoS/DoS"
    if any(k in lbl for k in ["BRUTE","FTP-PATATOR","SSH-PATATOR","PATATOR"]): return "Brute Force"
    if "BOT" in lbl: return "Botnet"
    if any(k in lbl for k in ["WEB","XSS","SQL","INJECTION","INFILTRAT"]): return "Web Attack"
    if "PORT" in lbl: return "PortScan"
    return "Other"

# =============================================================================
# DATA LOADING (with column mapping and numeric filtering)
# =============================================================================
def _build_col_map(df_cols):
    return {c.strip().lower(): c for c in df_cols}

def _find_available_features(df_cols):
    col_map = _build_col_map(df_cols)
    available, rename = [], {}
    for feat in ORIGINAL_FEATURES:
        key = feat.strip().lower()
        if key in col_map:
            actual = col_map[key]
            available.append(feat)
            if actual != feat:
                rename[actual] = feat
    return available, rename

def load_dataset(path, is_2018=False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    print(f"\nLoading : {path}")
    try:
        df = pd.read_csv(path, encoding='utf-8', low_memory=False, on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(path, encoding='latin-1', low_memory=False, engine='python', on_bad_lines='skip')

    # Map verbose column names
    name_mapping = {
        'Destination Port': 'Dst Port',
        'Total Fwd Packets': 'Tot Fwd Pkts',
        'Total Bwd Packets': 'Tot Bwd Pkts',
        'Total Length of Fwd Packets': 'TotLen Fwd Pkts',
        'Total Length of Bwd Packets': 'TotLen Bwd Pkts',
        'Fwd Packet Length Max': 'Fwd Pkt Len Max',
        'Fwd Packet Length Min': 'Fwd Pkt Len Min',
        'Fwd Packet Length Mean': 'Fwd Pkt Len Mean',
        'Fwd Packet Length Std': 'Fwd Pkt Len Std',
        'Bwd Packet Length Max': 'Bwd Pkt Len Max',
        'Bwd Packet Length Min': 'Bwd Pkt Len Min',
        'Bwd Packet Length Mean': 'Bwd Pkt Len Mean',
        'Bwd Packet Length Std': 'Bwd Pkt Len Std',
        'Flow Bytes/s': 'Flow Byts/s',
        'Flow Packets/s': 'Flow Pkts/s',
        'Fwd Packets/s': 'Fwd Pkts/s',
        'Bwd Packets/s': 'Bwd Pkts/s',
        'Fwd IAT Total': 'Fwd IAT Tot',
        'Bwd IAT Total': 'Bwd IAT Tot',
        'Fwd Header Length': 'Fwd Header Len',
        'Bwd Header Length': 'Bwd Header Len',
        'Packet Length Min': 'Pkt Len Min',
        'Packet Length Max': 'Pkt Len Max',
        'Packet Length Mean': 'Pkt Len Mean',
        'Packet Length Std': 'Pkt Len Std',
        'Packet Length Variance': 'Pkt Len Var',
        'SYN Flag Count': 'SYN Flag Cnt',
        'ACK Flag Count': 'ACK Flag Cnt',
        'FIN Flag Count': 'FIN Flag Cnt',
        'RST Flag Count': 'RST Flag Cnt',
        'PSH Flag Count': 'PSH Flag Cnt',
        'URG Flag Count': 'URG Flag Cnt',
        'CWE Flag Count': 'CWE Flag Count',
        'ECE Flag Count': 'ECE Flag Cnt',
        'Average Packet Size': 'Pkt Size Avg',
        'Fwd Segment Size Avg': 'Fwd Seg Size Avg',
        'Bwd Segment Size Avg': 'Bwd Seg Size Avg',
        'Fwd Bytes/b Avg': 'Fwd Byts/b Avg',
        'Fwd Packets/b Avg': 'Fwd Pkts/b Avg',
        'Fwd Blk Rate Avg': 'Fwd Blk Rate Avg',
        'Bwd Bytes/b Avg': 'Bwd Byts/b Avg',
        'Bwd Packets/b Avg': 'Bwd Pkts/b Avg',
        'Bwd Blk Rate Avg': 'Bwd Blk Rate Avg',
        'Subflow Fwd Packets': 'Subflow Fwd Pkts',
        'Subflow Fwd Bytes': 'Subflow Fwd Byts',
        'Subflow Bwd Packets': 'Subflow Bwd Pkts',
        'Subflow Bwd Bytes': 'Subflow Bwd Byts',
        'Init_Win_bytes_forward': 'Init Fwd Win Byts',
        'Init_Win_bytes_backward': 'Init Bwd Win Byts',
        'Fwd Act Data Packets': 'Fwd Act Data Pkts',
        'Fwd Seg Size Min': 'Fwd Seg Size Min',
    }
    df.columns = [name_mapping.get(col, col) for col in df.columns]
    df.columns = df.columns.str.strip()

    label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
    if label_col is None:
        raise KeyError(f"No 'Label' column in {path}")
    df.rename(columns={label_col: "Label"}, inplace=True)

    available_feats, rename_map = _find_available_features(df.columns)
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    missing = [f for f in ORIGINAL_FEATURES if f not in available_feats]
    print(f"  Features found: {len(available_feats)}/{len(ORIGINAL_FEATURES)}")
    if missing:
        print(f"  Missing: {len(missing)} (first 5: {missing[:5]})")

    mapper = standardize_label_2018 if is_2018 else standardize_label_2017
    df["Attack_Category"] = df["Label"].apply(mapper)

    df = df[df["Attack_Category"].isin(ATTACK_CATEGORIES)].copy()
    df[available_feats] = df[available_feats].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=available_feats, inplace=True)

    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = (df.groupby("Attack_Category", group_keys=False)
                .apply(lambda g: g.sample(
                    n=max(1, int(SAMPLE_SIZE * len(g) / len(df))),
                    random_state=RANDOM_STATE))
                .sample(frac=1, random_state=RANDOM_STATE)
                .reset_index(drop=True))

    print(f"  Rows loaded : {len(df):,}")
    print("  Class distribution:")
    for cat, cnt in df["Attack_Category"].value_counts().items():
        print(f"    {cat:<18s} {cnt:>8,} ({100*cnt/len(df):5.1f}%)")
    return df

# =============================================================================
# FEATURE ENGINEERING (adds stable derived features)
# =============================================================================
def engineer_features(df):
    df_eng = df.copy()
    for col in ORIGINAL_FEATURES:
        if col in df_eng.columns:
            df_eng[col] = np.clip(df_eng[col].values, -1e9, 1e9)
    if "Flow Duration" in df_eng.columns:
        df_eng["Log_Flow_Duration"] = np.log1p(np.clip(df_eng["Flow Duration"].values, -1e9, 1e9))
    if "TotLen Fwd Pkts" in df_eng.columns and "Tot Fwd Pkts" in df_eng.columns:
        denom = np.maximum(df_eng["Tot Fwd Pkts"].values, 1)
        df_eng["Avg_Fwd_Pkt_Size"] = np.clip(df_eng["TotLen Fwd Pkts"].values / denom, -1e9, 1e9)
    if "TotLen Bwd Pkts" in df_eng.columns and "Tot Bwd Pkts" in df_eng.columns:
        denom = np.maximum(df_eng["Tot Bwd Pkts"].values, 1)
        df_eng["Avg_Bwd_Pkt_Size"] = np.clip(df_eng["TotLen Bwd Pkts"].values / denom, -1e9, 1e9)
    if "TotLen Fwd Pkts" in df_eng.columns and "TotLen Bwd Pkts" in df_eng.columns:
        denom = np.maximum(df_eng["TotLen Bwd Pkts"].values, 1)
        df_eng["Fwd_Bwd_Byte_Ratio"] = np.clip(df_eng["TotLen Fwd Pkts"].values / denom, -1e9, 1e9)
    for col in ["Flow Byts/s", "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s"]:
        if col in df_eng.columns:
            df_eng[f"Log_{col}"] = np.log1p(np.clip(df_eng[col].values, -1e9, 1e9))
    for col in df_eng.columns:
        if col not in ['Label', 'Attack_Category']:
            df_eng[col] = df_eng[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_eng

# =============================================================================
# SHAP NORMALISER – handles binary and multiclass
# =============================================================================
def _get_shap_list(raw, n_classes, model=None):
    """Convert SHAP output to a list of arrays (one per class)."""
    if isinstance(raw, list):
        if len(raw) == 1 and n_classes == 2:
            arr = raw[0]
        else:
            return [np.array(item) for item in raw]
    else:
        arr = np.array(raw)

    if arr.ndim == 3:
        if arr.shape[2] == n_classes:
            return [arr[:, :, i] for i in range(n_classes)]
        elif arr.shape[0] == n_classes:
            return [arr[i] for i in range(n_classes)]
        else:
            for axis in [0, 2]:
                if arr.shape[axis] == n_classes:
                    return [np.take(arr, i, axis=axis) for i in range(n_classes)]
            raise ValueError(f"Cannot interpret 3-D SHAP {arr.shape}")
    elif arr.ndim == 2:
        if n_classes == 2:
            # Binary: single array is for the positive class (class 1)
            # We create [negative, positive] based on model.classes_ order
            if model is not None and hasattr(model, 'classes_'):
                if len(model.classes_) == 2 and model.classes_[0] == 0 and model.classes_[1] == 1:
                    return [-arr, arr]
                else:
                    return [-arr, arr]
            return [-arr, arr]
        else:
            return [arr]
    else:
        raise ValueError(f"Unexpected SHAP ndim={arr.ndim}")

# =============================================================================
# PER‑SAMPLE SHAP EXPLANATIONS
# =============================================================================
def save_shap_individual_explanations(model, X_shap_data, y_true, y_pred, y_prob,
                                      feature_names, le, out_dir, mode_name,
                                      top_n=5, max_samples=100):
    print(f"\n  Generating per-sample SHAP explanations for {mode_name} ...")
    n_samples = min(max_samples, len(X_shap_data))
    idx = np.random.choice(len(X_shap_data), n_samples, replace=False)
    X_small = X_shap_data[idx]
    y_true_small = y_true[idx]
    y_pred_small = y_pred[idx]
    y_prob_small = y_prob[idx]

    is_tree = isinstance(model, (RandomForestClassifier, ExtraTreesClassifier,
                                 XGBClassifier, LGBMClassifier))
    try:
        if is_tree:
            explainer = shap.TreeExplainer(model)
            raw_vals = explainer.shap_values(X_small)
        else:
            bg = shap.sample(X_shap_data, min(50, len(X_shap_data)))
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            raw_vals = explainer.shap_values(X_small, nsamples=100)

        n_classes = len(le.classes_)
        shap_list = _get_shap_list(raw_vals, n_classes, model=model)

        records = []
        for i in range(len(X_small)):
            true_label = le.inverse_transform([y_true_small[i]])[0]
            pred_label = le.inverse_transform([y_pred_small[i]])[0]
            confidence = np.max(y_prob_small[i])
            pred_class_idx = y_pred_small[i]

            shap_class = shap_list[pred_class_idx][i]
            feat_contrib = list(zip(feature_names, shap_class))
            feat_contrib.sort(key=lambda x: abs(x[1]), reverse=True)

            pos = [f"{name} ({val:+.3f})" for name, val in feat_contrib if val > 0][:top_n]
            neg = [f"{name} ({val:+.3f})" for name, val in feat_contrib if val < 0][:top_n]

            if pos:
                top_pos_names = [name for name, _ in [feat_contrib[j] for j in range(len(feat_contrib)) if feat_contrib[j][1] > 0][:3]]
                sentence = f"The high values of {', '.join(top_pos_names)} strongly contributed to the prediction."
            else:
                sentence = "No strong positive contributors."

            records.append({
                'Sample_Index': idx[i],
                'True_Label': true_label,
                'Predicted_Label': pred_label,
                'Confidence': f"{confidence:.4f}",
                'Top_Features_Increase': '; '.join(pos) if pos else 'None',
                'Top_Features_Decrease': '; '.join(neg) if neg else 'None',
                'Interpretation': sentence
            })

        df_exp = pd.DataFrame(records)
        csv_path = os.path.join(out_dir, f"shap_individual_{mode_name}.csv")
        df_exp.to_csv(csv_path, index=False)
        print(f"  Saved individual explanations: {csv_path}")

        print("\n  Sample individual explanations:")
        for i in range(min(3, len(df_exp))):
            row = df_exp.iloc[i]
            print(f"\n  Sample #{row['Sample_Index']}:")
            print(f"    True: {row['True_Label']}, Predicted: {row['Predicted_Label']} (Confidence: {row['Confidence']})")
            print(f"    ++ {row['Top_Features_Increase']}")
            print(f"    -- {row['Top_Features_Decrease']}")
            print(f"    💬 {row['Interpretation']}")

    except Exception as e:
        print(f"  ⚠ Could not generate individual SHAP explanations: {e}")
        import traceback
        print(traceback.format_exc())

# =============================================================================
# FIXED METRICS – handles binary and multiclass ROC‑AUC
# =============================================================================
def compute_metrics_fixed(y_true, y_pred, y_prob, le, model=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    n_cls = len(le.classes_)
    # Align probabilities to label encoder order
    if model is not None and hasattr(model, 'classes_'):
        model_classes = model.classes_
        y_prob_full = np.zeros((y_prob.shape[0], n_cls), dtype=np.float32)
        for i, cls_idx in enumerate(model_classes):
            if cls_idx < n_cls:
                y_prob_full[:, cls_idx] = y_prob[:, i]
        y_prob = y_prob_full
    else:
        if y_prob.shape[1] != n_cls:
            y_prob_full = np.zeros((y_prob.shape[0], n_cls), dtype=np.float32)
            cols = min(y_prob.shape[1], n_cls)
            y_prob_full[:, :cols] = y_prob[:, :cols]
            y_prob = y_prob_full

    # ROC‑AUC
    if len(set(y_true)) < 2:
        roc_auc = float('nan')
    else:
        try:
            if n_cls == 2:
                roc_auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                roc_auc = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class='ovr',
                    average='weighted',
                    labels=range(n_cls)
                )
        except Exception:
            roc_auc = float('nan')
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC-AUC": roc_auc}

# =============================================================================
# MODELS (generalised, no tuning for speed)
# =============================================================================
def get_fast_models():
    xgb = XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="mlogloss", tree_method="hist",
        n_jobs=-1, random_state=RANDOM_STATE
    )
    lgb = LGBMClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        class_weight="balanced", n_jobs=-1,
        random_state=RANDOM_STATE, verbose=-1
    )
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_split=10,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    )
    et = ExtraTreesClassifier(
        n_estimators=150, max_depth=10, min_samples_split=10,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        learning_rate_init=0.001, max_iter=100,
        early_stopping=True, validation_fraction=0.1,
        random_state=RANDOM_STATE
    )
    return {
        "Random Forest": rf,
        "XGBoost": xgb,
        "LightGBM": lgb,
        "Extra Trees": et,
        "MLP Neural Net": mlp
    }

# =============================================================================
# MAIN CROSS‑DATASET EVALUATION
# =============================================================================
def run_cross():
    print("\n" + "="*70)
    print("  CROSS‑DATASET EVALUATION  (2017 → 2018)")
    print("  Interpretable IDS with SHAP explanations")
    print("="*70)

    # Load and engineer both datasets
    df17 = load_dataset(CIC2017_CSV, is_2018=False)
    df18 = load_dataset(CIC2018_CSV, is_2018=True)
    df17_eng = engineer_features(df17)
    df18_eng = engineer_features(df18)

    # Common features (original + engineered)
    orig_common = [f for f in ORIGINAL_FEATURES if f in df17.columns and f in df18.columns]
    exclude_cols = ['Label', 'Attack_Category']
    eng_cols_17 = [c for c in df17_eng.columns if c not in ORIGINAL_FEATURES and c not in exclude_cols]
    eng_cols_18 = [c for c in df18_eng.columns if c not in ORIGINAL_FEATURES and c not in exclude_cols]
    eng_common = [c for c in eng_cols_17 if c in eng_cols_18]
    all_features = orig_common + eng_common
    print(f"\nTotal features after engineering: {len(all_features)}")

    # Build feature matrices and labels
    X_train = df17_eng[all_features].values.astype(np.float64)
    X_test  = df18_eng[all_features].values.astype(np.float64)
    union_classes = sorted(set(df17["Attack_Category"]) | set(df18["Attack_Category"]))
    le = LabelEncoder()
    le.fit(union_classes)
    y_train = le.transform(df17["Attack_Category"])
    y_test  = le.transform(df18["Attack_Category"])

    print(f"\nTraining classes: {sorted(set(y_train))}")
    print(f"Testing classes : {sorted(set(y_test))}")

    # Preprocess
    max_val = 1e6
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_train = np.clip(X_train, -max_val, max_val)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.clip(X_test, -max_val, max_val)

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    # SMOTE with cap (target = max(original, cap))
    if SMOTE_AVAILABLE:
        from collections import Counter
        counts = Counter(y_train)
        max_count = max(counts.values())
        target_size = min(MAX_CLASS_SAMPLES, max_count)
        target_size = max(target_size, 50000)
        sampling_strategy = {cls: max(counts[cls], target_size) for cls in counts.keys()}
        print(f"\nSMOTE target per class: {sampling_strategy}")
        sm = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
        X_train_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)
        print(f"Balanced training size: {len(X_train_bal)}")
    else:
        X_train_bal, y_train_bal = X_train_s, y_train

    # Feature selection (Top K)
    print(f"\nSelecting top {TOP_K} features via Mutual Information...")
    selector = SelectKBest(mutual_info_classif, k=min(TOP_K, X_train_bal.shape[1]))
    X_train_sel = selector.fit_transform(X_train_bal, y_train_bal)
    X_test_sel  = selector.transform(X_test_s)
    selected_idx = selector.get_support(indices=True)
    selected_features = [all_features[i] for i in selected_idx]
    print(f"Selected features: {selected_features}")

    # Final safety checks
    for arr, name in [(X_train_sel, "X_train_sel"), (X_test_sel, "X_test_sel")]:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -max_val, max_val).astype(np.float32)
        if name == "X_train_sel":
            X_train_sel = arr
        else:
            X_test_sel = arr

    # Build models
    print("\nBuilding models...")
    models = get_fast_models()
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )

    all_results = {}
    test_class_ids = sorted(set(y_test))
    test_class_names = [le.classes_[i] for i in test_class_ids]

    for name, model in models.items():
        print(f"\n  ── {name} ──")
        t0 = time.time()
        model.fit(X_train_sel, y_train_bal)
        t1 = time.time()
        y_pred = model.predict(X_test_sel)
        y_prob = model.predict_proba(X_test_sel)

        mets = compute_metrics_fixed(y_test, y_pred, y_prob, le, model)
        mets["Model"] = name
        mets["TrainSec"] = round(t1 - t0, 1)
        print(f"  Accuracy: {mets['Accuracy']:.4f}, Precision: {mets['Precision']:.4f}, Recall: {mets['Recall']:.4f}, F1: {mets['F1']:.4f}, ROC-AUC: {mets['ROC-AUC']:.4f}")
        all_results[name] = {k: v for k, v in mets.items() if k not in ['Model','TrainSec']}

    # Ensemble
    print(f"\n  ── Soft Voting Ensemble ──")
    t0 = time.time()
    ensemble.fit(X_train_sel, y_train_bal)
    t1 = time.time()
    y_pred_ens = ensemble.predict(X_test_sel)
    y_prob_ens = ensemble.predict_proba(X_test_sel)
    mets_ens = compute_metrics_fixed(y_test, y_pred_ens, y_prob_ens, le, ensemble)
    mets_ens["Model"] = "Ensemble (Soft Vote)"
    mets_ens["TrainSec"] = round(t1 - t0, 1)
    print(f"  Accuracy: {mets_ens['Accuracy']:.4f}, Precision: {mets_ens['Precision']:.4f}, Recall: {mets_ens['Recall']:.4f}, F1: {mets_ens['F1']:.4f}, ROC-AUC: {mets_ens['ROC-AUC']:.4f}")
    all_results["Ensemble (Soft Vote)"] = {k: v for k, v in mets_ens.items() if k not in ['Model','TrainSec']}

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY OF CROSS‑DATASET RESULTS")
    print("="*70)
    summary_df = pd.DataFrame(all_results).T.round(4)
    print(summary_df)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "cross_results_summary.csv"))

    # Confusion Matrix (ensemble)
    cm = confusion_matrix(y_test, y_pred_ens, labels=test_class_ids)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_class_names, yticklabels=test_class_names)
    plt.title("Confusion Matrix – Ensemble (Cross‑Dataset)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cm_ensemble_cross.png"), dpi=150)
    plt.close()

    # Global SHAP (using XGBoost)
    print("\nComputing global SHAP importance (XGBoost)...")
    xgb_model = models["XGBoost"]
    bg_idx = np.random.choice(len(X_train_sel), min(SHAP_ROWS, len(X_train_sel)), replace=False)
    X_bg = X_train_sel[bg_idx]
    X_shap = X_test_sel[:100]
    try:
        explainer = shap.TreeExplainer(xgb_model)
        raw_vals = explainer.shap_values(X_shap)
        shap_list = _get_shap_list(raw_vals, len(le.classes_), model=xgb_model)
        mean_abs = np.mean([np.abs(v).mean(axis=0) for v in shap_list], axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:15]
        top_feats = [selected_features[i] for i in top_idx]
        top_vals = mean_abs[top_idx]
        plt.figure(figsize=(10,6))
        plt.barh(top_feats[::-1], top_vals[::-1], color='steelblue')
        plt.xlabel("Mean |SHAP value|")
        plt.title("SHAP Feature Importance (XGBoost – Cross‑Dataset)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_global_cross.png"), dpi=150)
        plt.close()
        print("  Global SHAP chart saved.")
    except Exception as e:
        print(f"  Global SHAP skipped: {e}")

    # Per‑sample SHAP (using XGBoost)
    print("\nGenerating per-sample SHAP explanations (using XGBoost)...")
    save_shap_individual_explanations(
        model=xgb_model,
        X_shap_data=X_test_sel,
        y_true=y_test,
        y_pred=y_pred_ens,
        y_prob=y_prob_ens,
        feature_names=selected_features,
        le=le,
        out_dir=OUTPUT_DIR,
        mode_name="Cross_Ensemble_XGB",
        top_n=5,
        max_samples=SHAP_SAMPLES
    )

    # Save predictions
    pd.DataFrame({
        'True_Label': le.inverse_transform(y_test),
        'Predicted_Label': le.inverse_transform(y_pred_ens)
    }).head(1000).to_csv(os.path.join(OUTPUT_DIR, "predictions_cross.csv"), index=False)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    return all_results

if __name__ == "__main__":
    results = run_cross()
    if results is not None:
        print("\nFinal cross‑dataset results:")
        print(pd.DataFrame(results).T.round(4))
