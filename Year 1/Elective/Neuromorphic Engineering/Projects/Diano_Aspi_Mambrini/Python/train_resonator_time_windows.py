"""
Blind Resonator — Time-Window Grid Search & Analysis
=====================================================
Trains a Random Forest (GridSearchCV, 5-fold stratified CV)
on each of 10 time windows (10 – 2000 ms) for the Blind Resonator.

Outputs:
  - One confusion-matrix heatmap per time window
  - A summary plot: Accuracy vs Time Window (with interpolation)

Author: Generated with AI assistance
"""

import os
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ── Plot style ───────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans'],
    'axes.titleweight': 'bold',
})

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "Dataset", "Blind")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "resonator_time_windows")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Time windows & corresponding filenames ───────────────────────────────────
TIME_WINDOWS = [10, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]

FILE_MAP = {
    10:   "SpikeB_10ms_Resonator.csv",
    100:  "SpikeB_100ms_Resonator.csv",
    250:  "SpikeB_250ms_Resonator.csv",
    500:  "SpikeB_500ms_Resonator.csv",
    750:  "SpikeB_750ms_Resonator.csv",
    1000: "SpikeB_1000ms_Resonator.csv",
    1250: "SpikeB_1250_Resonator.csv",     # ← no "ms" in filename
    1500: "SpikeB_1500ms_Resonator.csv",
    1750: "SpikeB_1750ms_Resonator.csv",
    2000: "SpikeB_2000ms_Resonator.csv",
}

NEURON_COLS = ['N1', 'N2', 'N3', 'N4', 'N5']

# ── Hyperparameter grid (same as train_models.py) ───────────────────────────
PARAM_GRID = {
    'n_estimators':      [50, 100, 200],
    'max_depth':         [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2],
}

# ── Data loading ─────────────────────────────────────────────────────────────

def fix_decimal_commas(filepath):
    """Fix CSV files that use commas as decimal separators (15 → 8 cols)."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None
    if len(lines[0].strip().split(',')) == 15:
        new_lines = []
        for line in lines:
            p = line.strip().split(',')
            row = (f'{p[0]}.{p[1]},{p[2]},'
                   f'{p[3]}.{p[4]},{p[5]}.{p[6]},{p[7]}.{p[8]},'
                   f'{p[9]}.{p[10]},{p[11]}.{p[12]},{p[13]}.{p[14]}')
            new_lines.append(row)
        return new_lines
    return None


def load_spike_dataset_5ch(filepath):
    """Load a 5-channel spike CSV (8 columns: timestamp, label, N1–N5, sum)."""
    col_names = ['timestamp', 'label', 'N1', 'N2', 'N3', 'N4', 'N5', 'sum']
    fixed = fix_decimal_commas(filepath)
    try:
        if fixed is not None:
            text = chr(10).join(fixed)
            df = pd.read_csv(StringIO(text), header=None, names=col_names)
        else:
            df = pd.read_csv(filepath, header=None, names=col_names)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=col_names)
    df = df.dropna()
    df = df.drop(columns=['timestamp', 'sum'])
    df['label'] = df['label'].astype(int)
    return df


# ── Confusion matrix plot ────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, tw_ms, output_path):
    """Save a publication-quality confusion matrix heatmap."""
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Build annotations with count + percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    cmap = sns.color_palette("mako", as_cmap=True)
    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.8, linecolor='#2c3e50',
                cbar_kws={'label': 'Count', 'shrink': 0.82},
                square=True, ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
    ax.set_ylabel('True Label', fontsize=12, labelpad=10)
    ax.set_title(f'Confusion Matrix — Blind Resonator\n'
                 f'Time Window: {tw_ms} ms',
                 fontsize=14, fontweight='bold', pad=14)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)


# ── Summary plot ─────────────────────────────────────────────────────────────

def plot_accuracy_vs_time_window(results, output_path):
    """Save a clean accuracy-vs-time-window summary plot with spline."""
    tw  = np.array([r['time_window'] for r in results], dtype=float)
    acc = np.array([r['accuracy']    for r in results])

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    # Smooth spline interpolation
    tw_smooth = np.linspace(tw.min(), tw.max(), 300)
    try:
        spl = make_interp_spline(tw, acc, k=3)
        acc_smooth = spl(tw_smooth)
    except Exception:
        acc_smooth = np.interp(tw_smooth, tw, acc)

    # Interpolated curve
    ax.plot(tw_smooth, acc_smooth, color='#2980b9', linewidth=2.5, zorder=3)

    # Data-point markers
    ax.scatter(tw, acc, s=70, color='#2980b9', edgecolors='white',
               linewidths=1.5, zorder=4)

    # Annotate each point
    for x, y in zip(tw, acc):
        if x == 10:
            ofs = (14, -8)
        elif x == 100:
            ofs = (18, -18)
        else:
            ofs = (0, 12)
        ax.annotate(f'{y:.3f}', (x, y),
                    textcoords='offset points', xytext=ofs,
                    ha='center', fontsize=9, fontweight='bold',
                    color='#2c3e50')

    # Axes & styling
    ax.set_xlabel('Time Window (ms)', fontsize=13, labelpad=10)
    ax.set_ylabel('Accuracy (5-Fold CV)', fontsize=13, labelpad=10)
    ax.set_title('Blind Resonator — Accuracy vs Time Window',
                 fontsize=15, fontweight='bold', pad=14)

    ax.set_xticks(tw)
    ax.set_xticklabels([f'{int(t)}' for t in tw], fontsize=10)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlim(tw.min() - 60, tw.max() + 60)

    ax.grid(True, alpha=0.35, linestyle='--')
    for spine in ax.spines.values():
        spine.set_color('#bdc3c7')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='#fafafa', edgecolor='none')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 70)
    print('  BLIND RESONATOR — TIME-WINDOW GRID SEARCH')
    print('=' * 70)
    print(f'  Time windows : {TIME_WINDOWS}')
    print(f'  Output dir   : {OUTPUT_DIR}')
    n_comb = 1
    for v in PARAM_GRID.values():
        n_comb *= len(v)
    print(f'  HP combos    : {n_comb}  (× 5-fold = {n_comb * 5} fits per window)')
    print('=' * 70)
    print()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = []

    for tw in TIME_WINDOWS:
        fname = FILE_MAP[tw]
        fpath = os.path.join(DATASET_DIR, fname)

        print(f'-- {tw} ms ' + '-' * 42)
        if not os.path.isfile(fpath):
            print(f'  [!!] File not found: {fname}')
            continue

        df = load_spike_dataset_5ch(fpath)
        n_samples = len(df)
        if n_samples < 5:
            print(f'  [!!] Too few samples ({n_samples}), skipping.')
            continue
        print(f'  Samples: {n_samples}')

        X = df[NEURON_COLS].values
        y = df['label'].values
        n_classes = len(np.unique(y))
        print(f'  Classes: {n_classes}')

        # ── Grid Search ──────────────────────────────────────────────────
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

        depth_str = str(bp['max_depth']) if bp['max_depth'] is not None else 'None'
        print(f'  Best Acc : {acc_mean:.4f} ± {acc_std:.4f}')
        print(f'  Best F1  : {f1_mean:.4f} ± {f1_std:.4f}')
        print(f'  Params   : n_est={bp["n_estimators"]}, depth={depth_str}, '
              f'split={bp["min_samples_split"]}, leaf={bp["min_samples_leaf"]}')

        # ── Cross-validated confusion matrix ─────────────────────────────
        best_model = grid.best_estimator_
        y_pred_cv = cross_val_predict(best_model, X, y, cv=skf, n_jobs=-1)

        cm_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{tw}ms.png')
        plot_confusion_matrix(y, y_pred_cv, tw, cm_path)
        print(f'  [OK] Confusion matrix saved: {os.path.basename(cm_path)}')

        all_results.append({
            'time_window': tw,
            'accuracy': acc_mean,
            'acc_std': acc_std,
            'f1_macro': f1_mean,
            'f1_std': f1_std,
            'n_samples': n_samples,
            'n_classes': n_classes,
            'best_params': bp,
        })
        print()

    # ── Summary plot ─────────────────────────────────────────────────────────
    if all_results:
        summary_path = os.path.join(OUTPUT_DIR, 'accuracy_vs_time_window.png')
        plot_accuracy_vs_time_window(all_results, summary_path)
        print(f'[OK] Summary plot saved: {summary_path}')

        # Console summary table
        print()
        print('=' * 70)
        print(f'  {"TW (ms)":>8}  {"Accuracy":>10}  {"± Std":>8}  '
              f'{"F1 Macro":>10}  {"Samples":>8}')
        print('-' * 70)
        for r in all_results:
            print(f'  {r["time_window"]:>8}  {r["accuracy"]:>10.4f}  '
                  f'{r["acc_std"]:>8.4f}  {r["f1_macro"]:>10.4f}  '
                  f'{r["n_samples"]:>8}')
        print('=' * 70)
    else:
        print('[!!] No results -- cannot generate summary plot.')

    print('\n[OK] All done!')
