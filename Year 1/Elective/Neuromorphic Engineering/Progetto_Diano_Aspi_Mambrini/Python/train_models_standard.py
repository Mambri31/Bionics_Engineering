"""
Standard Random Forest Training — 2 Models (Blind + BlindDir)
==============================================================
Trains two gesture-classification models from the preprocessed1.csv
hand-tracking dataset:

  1. standard_blind.pkl     — 5 features (palm-to-fingertip distances)
  2. standard_blind_dir.pkl — 8 features (5 distances + 3 direction components)

Both models use 1-second time-window aggregation and GridSearchCV (5-fold
stratified CV) to find the best Random Forest hyperparameters.

Dataset columns (preprocessed1.csv):
    timestamp, class,
    palm_x, palm_y, palm_z,
    direction_x, direction_y, direction_z,
    ch_1_x, ch_1_y, ch_1_z, ..., ch_5_x, ch_5_y, ch_5_z,
    ch_1_missing, dist_ch_1, ..., ch_5_missing, dist_ch_5,
    n_valid

Gesture classes:
    1 - Open palm
    2 - Fist
    3 - Thumb only
    4 - Pinky only
    5 - Horns (index + ring + thumb)
    6 - Thumb + pinky


"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for server / LabVIEW
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "Dataset", "preprocessed1.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Models", "Standard")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Gesture label mapping
GESTURE_LABELS = {
    1: "Open palm",
    2: "Fist",
    3: "Thumb only",
    4: "Pinky only",
    5: "Horns (ind+ring+thu)",
    6: "Thumb + pinky",
}

# Time-window aggregation
TIME_WINDOW_SEC = 1.0  # 1-second aggregation window

# Direction columns present in the dataset
DIRECTION_COLS = ["direction_x", "direction_y", "direction_z"]

# Hyperparameter grid for GridSearchCV
PARAM_GRID = {
    'n_estimators':      [50, 100, 200],
    'max_depth':         [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2],
}

# Model configurations: (model_name, feature_cols_fn, description)
# feature_cols_fn returns the column names to use from the aggregated DataFrame
MODEL_CONFIGS = [
    {
        'name':          'standard_blind',
        'filename':      'standard_blind.pkl',
        'use_direction': False,
        'n_features':    5,
        'description':   '5 distance features (palm -> fingertip)',
        'color':         '#27ae60',
    },
    {
        'name':          'standard_blind_dir',
        'filename':      'standard_blind_dir.pkl',
        'use_direction': True,
        'n_features':    8,
        'description':   '5 distances + 3 direction components',
        'color':         '#8e44ad',
    },
]

n_comb = 1
for v in PARAM_GRID.values():
    n_comb *= len(v)


# ============================================================================
# 2. DATA LOADING
# ============================================================================

def load_dataset(path):
    """Load preprocessed1.csv — only the columns we need, in float32 to save RAM.

    This avoids MemoryError on 32-bit Python by loading ~28 columns
    instead of all 34, and using float32 instead of float64.
    """
    # Only the columns required for distance computation + direction + labels
    usecols = (
        ['timestamp', 'class',
         'palm_x', 'palm_y', 'palm_z',
         'direction_x', 'direction_y', 'direction_z']
        + [f'ch_{i}_{ax}' for i in range(1, 6) for ax in ('x', 'y', 'z')]
        + [f'ch_{i}_missing' for i in range(1, 6)]
    )
    print(f"[LOAD] Reading: {os.path.basename(path)} ({len(usecols)} columns)")
    df = pd.read_csv(path, usecols=usecols, dtype={c: np.float32
                     for c in usecols if c != 'class'})
    print(f"   Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    return df


# ============================================================================
# 3. FEATURE EXTRACTION — Euclidean distances (palm → fingertip)
# ============================================================================

def compute_distances(df):
    """
    For each of the 5 fingers, compute the 3-D Euclidean distance
    from the palm centre to the fingertip.

    If the finger is missing (ch_X_missing == 1), the distance is set to 0
    (zero-padding).

    After computing distances, raw coordinate and missing columns are dropped
    to free memory (important for 32-bit Python).

    Returns the DataFrame with 5 new columns: dist_1 .. dist_5.
    """
    palm_cols = ["palm_x", "palm_y", "palm_z"]
    drop_cols = list(palm_cols)  # will collect cols to drop

    for i in range(1, 6):
        tip_cols = [f"ch_{i}_x", f"ch_{i}_y", f"ch_{i}_z"]
        missing_col = f"ch_{i}_missing"

        # Vectorised Euclidean distance
        dx = df[tip_cols[0]].values - df[palm_cols[0]].values
        dy = df[tip_cols[1]].values - df[palm_cols[1]].values
        dz = df[tip_cols[2]].values - df[palm_cols[2]].values
        dist = np.sqrt(dx**2 + dy**2 + dz**2).astype(np.float32)

        # Zero-pad where the finger data is missing
        missing_mask = df[missing_col].values == 1
        dist[missing_mask] = 0.0

        df[f"dist_{i}"] = dist
        drop_cols.extend(tip_cols + [missing_col])

    print("[FEAT] Computed 5 Euclidean-distance channels (palm -> fingertip).")
    n_missing_cols = [f"ch_{i}_missing" for i in range(1, 6)]
    n_missing = (df[n_missing_cols] == 1).sum()
    for i in range(1, 6):
        pct = n_missing[f"ch_{i}_missing"] / len(df) * 100
        print(f"   ch_{i}: {n_missing[f'ch_{i}_missing']:,} missing rows "
              f"({pct:.1f}%) -> padded with 0")

    # Free memory by dropping raw coordinate + missing columns
    df = df.drop(columns=drop_cols)
    import gc; gc.collect()
    print(f"[MEM]  Dropped {len(drop_cols)} raw columns to free memory.")

    return df


# ============================================================================
# 4. TIME AGGREGATION — 1-second window mean
# ============================================================================

def aggregate_by_time_window(df, window_sec=1.0, include_direction=False):
    """
    Group samples into non-overlapping windows of `window_sec` seconds.

    Within each window the majority class is taken as the label, the
    5 distance channels are averaged, and optionally the 3 direction
    components are averaged as well.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: timestamp, class, dist_1..dist_5.
        If include_direction=True, must also contain direction_x/y/z.
    window_sec : float
        Window duration in seconds (default 1.0 s).
    include_direction : bool
        Whether to include direction columns in the aggregation.

    Returns
    -------
    agg : DataFrame
        Aggregated DataFrame.
    """
    # Subselect only necessary columns to save memory
    cols_to_keep = (["timestamp", "class"]
                    + [f"dist_{i}" for i in range(1, 6)]
                    + (DIRECTION_COLS if include_direction else []))
    df_sub = df[cols_to_keep].copy()

    # Assign each row to a time-window bucket
    t0 = df_sub["timestamp"].min()
    df_sub["window_id"] = ((df_sub["timestamp"] - t0) / window_sec).astype(int)

    # Build aggregation dictionary
    agg_dict = {
        "window_start": ("timestamp", "min"),
        # 5 distance means
        **{f"dist_{i}_mean": (f"dist_{i}", "mean") for i in range(1, 6)},
        # Majority class via lambda (mode)
        "gesture_class": ("class", lambda s: s.mode().iloc[0]),
        "n_samples": ("timestamp", "size"),
    }

    # Optionally add direction means
    if include_direction:
        for col in DIRECTION_COLS:
            agg_dict[f"{col}_mean"] = (col, "mean")

    agg = df_sub.groupby("window_id").agg(**agg_dict).reset_index(drop=True)

    print(f"[AGG]  Aggregated into {len(agg):,} windows of {window_sec}s "
          f"(avg {agg['n_samples'].mean():.0f} samples/window).")

    return agg


# ============================================================================
# 5. TRAINING FUNCTION
# ============================================================================

def train_model(X, y, config_name):
    """
    Run GridSearchCV with 5-fold stratified CV and return the best model
    along with its performance metrics.

    Returns
    -------
    model : RandomForestClassifier
        The final model retrained on the full dataset with best params.
    metrics : dict
        Dictionary with accuracy, f1, best params, etc.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight='balanced'),
        param_grid=PARAM_GRID,
        cv=skf,
        scoring={'accuracy': 'accuracy', 'f1': 'f1_macro'},
        refit='accuracy',
        n_jobs=-1,
        return_train_score=False,
    )
    grid.fit(X, y)

    bp = grid.best_params_
    bi = grid.best_index_
    acc_mean = grid.cv_results_['mean_test_accuracy'][bi]
    acc_std  = grid.cv_results_['std_test_accuracy'][bi]
    f1_mean  = grid.cv_results_['mean_test_f1'][bi]
    f1_std   = grid.cv_results_['std_test_f1'][bi]

    # Final model retrained on the full dataset with best params
    rf_final = RandomForestClassifier(
        **bp, random_state=42, n_jobs=-1, class_weight='balanced')
    rf_final.fit(X, y)

    depth_str = str(bp['max_depth']) if bp['max_depth'] is not None else 'None'
    print(f'  {config_name:<25s} | Acc: {acc_mean:.4f} +/- {acc_std:.4f} | '
          f'F1: {f1_mean:.4f} +/- {f1_std:.4f}')
    print(f'  {"":25s} | n_est={bp["n_estimators"]:>3}, depth={depth_str:<4}, '
          f'split={bp["min_samples_split"]}, leaf={bp["min_samples_leaf"]}')

    metrics = {
        'Model': config_name,
        'Accuracy': acc_mean, 'Acc_Std': acc_std,
        'F1_Macro': f1_mean, 'F1_Std': f1_std,
        'n_estimators': bp['n_estimators'],
        'max_depth': bp['max_depth'],
        'min_samples_split': bp['min_samples_split'],
        'min_samples_leaf': bp['min_samples_leaf'],
        'Samples': len(y),
    }

    return rf_final, metrics


# ============================================================================
# 6. EVALUATION & VISUALISATION
# ============================================================================

def plot_confusion_matrix(model, X, y, model_name, feature_names, output_dir):
    """
    Generate and save confusion matrix plots (absolute + normalized)
    for the given model using cross-validated predictions.
    """
    from sklearn.model_selection import cross_val_predict

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=skf)
    acc = accuracy_score(y, y_pred)

    unique_classes = sorted(np.unique(y))
    target_names = [GESTURE_LABELS.get(c, f"Class {c}") for c in unique_classes]

    # Classification report
    print(f"\n[REPORT] Classification Report — {model_name}:")
    print("-" * 70)
    print(classification_report(y, y_pred,
                                labels=unique_classes,
                                target_names=target_names))

    # Feature importance
    importances = model.feature_importances_
    print(f"[KEY] Feature Importance — {model_name}:")
    for feat, imp in zip(feature_names, importances):
        bar = "#" * int(imp * 50)
        print(f"   {feat}: {imp:.4f}  {bar}")

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y, y_pred, labels=unique_classes)
    cm_norm = confusion_matrix(y, y_pred, labels=unique_classes, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[0], linewidths=.5, linecolor="gray",
                cbar_kws={"label": "Count"})
    axes[0].set_xlabel("Predicted", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("True", fontsize=12, fontweight="bold")
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=30)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[1], linewidths=.5, linecolor="gray",
                vmin=0, vmax=1,
                cbar_kws={"label": "Proportion"})
    axes[1].set_xlabel("Predicted", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("True", fontsize=12, fontweight="bold")
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=13, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle(
        f"Random Forest — {model_name}\n"
        f"CV Accuracy: {acc * 100:.2f}%  |  Features: {len(feature_names)}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] Confusion matrix -> {cm_path}")

    # --- Feature importance bar chart ---
    fig2, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.6)))
    n_feat = len(feature_names)
    if n_feat <= 5:
        display_labels = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    else:
        display_labels = (["Thumb (dist)", "Index (dist)", "Middle (dist)",
                           "Ring (dist)", "Pinky (dist)"]
                          + ["Direction X", "Direction Y", "Direction Z"])

    # Two-tone colour palette for distances vs directions
    dist_colors = sns.color_palette("viridis", 5)
    dir_colors = sns.color_palette("magma", 3) if n_feat > 5 else []
    colors = dist_colors + dir_colors

    bars = ax.barh(display_labels[:n_feat], importances, color=colors[:n_feat],
                   edgecolor="black")
    ax.set_xlabel("Importance", fontsize=12, fontweight="bold")
    ax.set_title(f"Feature Importance — {model_name}",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{imp:.4f}", va="center", fontsize=10)
    plt.tight_layout()
    fi_path = os.path.join(output_dir, f"feature_importance_{model_name}.png")
    fig2.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[SAVE] Feature importance -> {fi_path}")

    return acc


# ============================================================================
# 7. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("  STANDARD MODEL TRAINING -- 2 MODELS (Blind + BlindDir)")
    print("=" * 80)
    print()
    print(f"Dataset:                      {DATASET_PATH}")
    print(f"Output:                       {OUTPUT_DIR}")
    print(f"Total HP combinations/model:  {n_comb}")
    print(f"Trainings per model (x5 fold): {n_comb * 5}")
    print(f"Time-window aggregation:       {TIME_WINDOW_SEC}s")
    print()

    # --- Load dataset ---
    df = load_dataset(DATASET_PATH)

    # --- Class distribution ---
    print("\n[INFO] Class distribution (raw samples):")
    for cls_id, count in df["class"].value_counts().sort_index().items():
        label = GESTURE_LABELS.get(cls_id, f"Class {cls_id}")
        print(f"   Class {cls_id} ({label}): {count:,}")

    # --- Feature extraction (Euclidean distances) ---
    df = compute_distances(df)

    # --- Direction summary ---
    print("\n[FEAT] Direction channels summary (raw):")
    for col in DIRECTION_COLS:
        print(f"   {col}: mean={df[col].mean():.4f}, "
              f"std={df[col].std():.4f}, "
              f"min={df[col].min():.4f}, max={df[col].max():.4f}")

    # --- Train both models ---
    all_metrics = []

    for cfg in MODEL_CONFIGS:
        print()
        print("=" * 80)
        print(f"  TRAINING -- {cfg['name'].upper()} ({cfg['description']})")
        print("=" * 80)
        print()

        # Aggregate with or without direction
        agg = aggregate_by_time_window(
            df, window_sec=TIME_WINDOW_SEC,
            include_direction=cfg['use_direction']
        )

        # Build feature columns
        dist_feature_cols = [f"dist_{i}_mean" for i in range(1, 6)]
        if cfg['use_direction']:
            dir_feature_cols = [f"{col}_mean" for col in DIRECTION_COLS]
            feature_cols = dist_feature_cols + dir_feature_cols
        else:
            feature_cols = dist_feature_cols

        X = agg[feature_cols].values
        y = agg["gesture_class"].values

        print(f"[DATA] Feature matrix shape: {X.shape}")
        print(f"[DATA] Feature names: {feature_cols}")
        print()

        # Train with GridSearchCV
        model, metrics = train_model(X, y, cfg['name'])
        all_metrics.append(metrics)

        # Save model
        pkl_path = os.path.join(OUTPUT_DIR, cfg['filename'])
        joblib.dump(model, pkl_path)
        print(f"\n[SAVE] Model saved -> {pkl_path}")

        # Evaluation plots
        plot_confusion_matrix(model, X, y, cfg['name'], feature_cols, OUTPUT_DIR)

    # --- Summary ---
    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print()

    df_metrics = pd.DataFrame(all_metrics)
    print(df_metrics[['Model', 'Accuracy', 'Acc_Std', 'F1_Macro', 'F1_Std',
                       'n_estimators', 'max_depth', 'Samples']].to_string(index=False))

    # Save Excel summary
    xlsx_path = os.path.join(OUTPUT_DIR, "summary_results_standard.xlsx")
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as w:
        df_metrics.to_excel(w, sheet_name='Results', index=False)
    print(f"\n[OK] Excel summary saved: {xlsx_path}")

    # --- Comparison plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODEL_CONFIGS))
    w_bar = 0.35

    acc_vals = [m['Accuracy'] for m in all_metrics]
    acc_stds = [m['Acc_Std'] for m in all_metrics]
    f1_vals  = [m['F1_Macro'] for m in all_metrics]
    f1_stds  = [m['F1_Std'] for m in all_metrics]
    names    = [m['Model'] for m in all_metrics]
    colors   = [c['color'] for c in MODEL_CONFIGS]

    bars1 = ax.bar(x - w_bar/2, acc_vals, w_bar, yerr=acc_stds,
                   label='Accuracy', color=colors, edgecolor='white',
                   capsize=5, alpha=0.9)
    bars2 = ax.bar(x + w_bar/2, f1_vals, w_bar, yerr=f1_stds,
                   label='F1-Macro', color=colors, edgecolor='white',
                   capsize=5, alpha=0.5, hatch='//')

    for b, v in zip(bars1, acc_vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015,
                f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
    for b, v in zip(bars2, f1_vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015,
                f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Score (5-Fold CV)', fontsize=12)
    ax.set_title('Standard Models — Accuracy & F1 Comparison\n'
                 '(GridSearchCV, 1s time windows)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1/6, color='grey', ls='--', alpha=0.5, label='Chance (1/6)')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    cmp_path = os.path.join(OUTPUT_DIR, 'comparison_standard_models.png')
    fig.savefig(cmp_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Comparison plot saved: {cmp_path}")

    # --- List saved models ---
    print()
    for cfg in MODEL_CONFIGS:
        pkl = os.path.join(OUTPUT_DIR, cfg['filename'])
        if os.path.exists(pkl):
            print(f"  {cfg['filename']:30s} ({os.path.getsize(pkl)/1024:>6.0f} KB)")
    print()
    print("[OK] All done!")


if __name__ == "__main__":
    main()
