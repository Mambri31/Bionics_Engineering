"""
Unified Random Forest Training Script — All 24 Neuromorphic Models
===================================================================
Trains models for all 3 dataset configurations:
  - Routed   (SpikeN_*)  → 8 neuron types → model_route_node/
  - Blind    (SpikeB_*)  → 8 neuron types → model_blind/
  - BlindDir (SpikeBD_*) → 8 neuron types → model_blind_dir/

Total: 3 × 8 = 24 models

Each model is trained with GridSearchCV (5-fold stratified CV)
and saved as a .pkl file.

Author: Generated with AI assistance
"""

import os, glob, re
import pandas as pd
import numpy as np
import joblib
from io import StringIO
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for server / LabVIEW
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

# ============================================================================
# 1. PATHS & CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "Dataset")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Models", "Neuromorphic")

# Feature column definitions
NEURON_COLS_5CH = ['N1', 'N2', 'N3', 'N4', 'N5']
NEURON_COLS_11CH = [
    'N1', 'N2', 'N3', 'N4', 'N5',
    'dir_x_pos', 'dir_y_pos', 'dir_z_pos',
    'dir_x_neg', 'dir_y_neg', 'dir_z_neg'
]
NEURON_COLS_1CH = ['N_sum']

# Training configurations: (display_label, dataset_subfolder, output_subfolder, spike_prefix, feature_cols)
TRAIN_CONFIGS = [
    {
        'label':        'Routed',
        'data_dir':     os.path.join(DATASET_DIR, 'Routed'),
        'output_dir':   os.path.join(OUTPUT_DIR, 'model_route_node'),
        'prefix':       'SpikeN_',
        'features':     NEURON_COLS_5CH,
        'n_features':   5,
        'color':        '#2980b9',
    },
    {
        'label':        'Blind',
        'data_dir':     os.path.join(DATASET_DIR, 'Blind'),
        'output_dir':   os.path.join(OUTPUT_DIR, 'model_blind'),
        'prefix':       'SpikeB_',
        'features':     NEURON_COLS_5CH,
        'n_features':   5,
        'color':        '#27ae60',
    },
    {
        'label':        'BlindDir',
        'data_dir':     os.path.join(DATASET_DIR, 'BlindDir'),
        'output_dir':   os.path.join(OUTPUT_DIR, 'model_blind_dir'),
        'prefix':       'SpikeBD_',
        'features':     NEURON_COLS_11CH,
        'n_features':   11,
        'color':        '#8e44ad',
    },
    {
        'label':        'BlindSum',
        'data_dir':     os.path.join(DATASET_DIR, 'Blind'),
        'output_dir':   os.path.join(OUTPUT_DIR, 'model_blind_sum'),
        'prefix':       'SpikeB_',
        'features':     NEURON_COLS_1CH,
        'n_features':   1,
        'color':        '#e67e22',
    },
]

# Hyperparameter grid for GridSearchCV
PARAM_GRID = {
    'n_estimators':     [50, 100, 200],
    'max_depth':        [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2],
}

n_comb = 1
for v in PARAM_GRID.values():
    n_comb *= len(v)

# ============================================================================
# 2. UTILITY FUNCTIONS
# ============================================================================

def fix_decimal_commas(filepath):
    """Fixes CSV file with decimal comma (15 raw columns -> 8 real columns).

    Some CSV files exported from LabVIEW use commas as decimal separators,
    resulting in 15 columns instead of 8. This function merges pairs of
    columns back into proper decimal numbers.

    Returns the fixed lines as a list of strings, or None if no fix is needed.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None
    n_cols = len(lines[0].strip().split(','))
    if n_cols == 15:
        new_lines = []
        for line in lines:
            p = line.strip().split(',')
            row = (f'{p[0]}.{p[1]},{p[2]},'
                   f'{p[3]}.{p[4]},{p[5]}.{p[6]},{p[7]}.{p[8]},'
                   f'{p[9]}.{p[10]},{p[11]}.{p[12]},{p[13]}.{p[14]}')
            new_lines.append(row)
        return new_lines
    return None


def extract_neuron_name(filename):
    """Extracts the neuron type from the filename.

    Handles all known prefixes: SpikeBD_, SpikeB_, SpikeN_, SpikeT_, Spike_
    """
    basename = os.path.basename(filename).replace('.csv', '')
    for prefix in ['SpikeBD_', 'SpikeB_', 'SpikeN_', 'SpikeT_', 'Spike_']:
        if basename.startswith(prefix):
            rest = basename[len(prefix):]
            break
    else:
        return basename
    parts = rest.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[1:])
    # Case without underscore after timestamp (e.g. '...091815Fast')
    m = re.search(r'\d+(.+)', rest)
    return m.group(1) if m else rest


def load_spike_dataset_5ch(filepath):
    """Loads a 5-channel spike dataset (Routed or Blind).

    Expected format: 8 columns
        timestamp, label, N1, N2, N3, N4, N5, sum

    Returns a DataFrame with label + N1-N5.
    """
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


def load_spike_dataset_11ch(filepath):
    """Loads an 11-channel spike dataset (BlindDir).

    Expected format: 14 columns
        timestamp, label, N1..N5, dir_x_pos, dir_y_pos, dir_z_pos,
        dir_x_neg, dir_y_neg, dir_z_neg, sum

    Returns a DataFrame with label + 11 feature columns.
    """
    col_names = [
        'timestamp', 'label',
        'N1', 'N2', 'N3', 'N4', 'N5',
        'dir_x_pos', 'dir_y_pos', 'dir_z_pos',
        'dir_x_neg', 'dir_y_neg', 'dir_z_neg',
        'sum'
    ]
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


def load_spike_dataset_1ch_sum(filepath):
    """Loads a 5-channel spike dataset and sums N1..N5 into a single feature.

    Uses the same Blind CSV format (8 columns) but returns label + N_sum,
    where N_sum = N1 + N2 + N3 + N4 + N5.

    Returns a DataFrame with label + N_sum.
    """
    df = load_spike_dataset_5ch(filepath)
    if df.empty:
        return df
    df['N_sum'] = df[['N1', 'N2', 'N3', 'N4', 'N5']].sum(axis=1)
    df = df.drop(columns=['N1', 'N2', 'N3', 'N4', 'N5'])
    return df


# ============================================================================
# 3. TRAINING FUNCTION
# ============================================================================

def train_and_save(data_dir, output_dir, config_label,
                   spike_prefix, neuron_cols, n_features):
    """Grid Search + train + save for each neuron CSV in the directory.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the spike CSV files.
    output_dir : str
        Path to the directory where trained .pkl models will be saved.
    config_label : str
        Display label for this configuration (e.g. 'Routed', 'Blind', 'BlindDir').
    spike_prefix : str
        Filename prefix to glob for (e.g. 'SpikeN_', 'SpikeB_', 'SpikeBD_').
    neuron_cols : list[str]
        Feature column names to use for training.
    n_features : int
        Number of features (5 or 11).

    Returns
    -------
    results : list[dict]
        List of result dictionaries, one per trained model.
    """
    # Select the appropriate loader based on feature count
    if n_features == 11:
        load_fn = load_spike_dataset_11ch
    elif n_features == 1:
        load_fn = load_spike_dataset_1ch_sum
    else:
        load_fn = load_spike_dataset_5ch

    files = sorted(glob.glob(os.path.join(data_dir, spike_prefix + '*.csv')))
    if not files:
        print(f'  WARNING: No files found matching {spike_prefix}*.csv in {data_dir}')
        return []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    os.makedirs(output_dir, exist_ok=True)

    for filepath in files:
        name = extract_neuron_name(filepath)
        df = load_fn(filepath)
        if len(df) < 5:
            print(f'  Skipping {name} due to insufficient samples ({len(df)}).')
            continue
        X = df[neuron_cols].values
        y = df['label'].values

        # Grid Search with double scoring
        grid = GridSearchCV(
            estimator=RandomForestClassifier(
                random_state=42, n_jobs=-1, class_weight='balanced'),
            param_grid=PARAM_GRID,
            cv=skf,
            scoring={'accuracy': 'accuracy', 'f1': 'f1_macro'},
            refit='accuracy',
            n_jobs=-1,
            return_train_score=False
        )
        grid.fit(X, y)

        bp = grid.best_params_
        bi = grid.best_index_
        acc_mean = grid.cv_results_['mean_test_accuracy'][bi]
        acc_std  = grid.cv_results_['std_test_accuracy'][bi]
        f1_mean  = grid.cv_results_['mean_test_f1'][bi]
        f1_std   = grid.cv_results_['std_test_f1'][bi]

        # Final model retrained on the whole dataset with best params
        rf_final = RandomForestClassifier(
            **bp, random_state=42, n_jobs=-1, class_weight='balanced')
        rf_final.fit(X, y)

        # Save model
        pkl = os.path.join(output_dir, f'rf_model_{name}.pkl')
        joblib.dump(rf_final, pkl)

        depth_str = str(bp['max_depth']) if bp['max_depth'] is not None else 'None'
        print(f'  {name:<20s} | Acc: {acc_mean:.4f} \u00b1 {acc_std:.4f} | '
              f'F1: {f1_mean:.4f} | '
              f'n_est={bp["n_estimators"]:>3}, depth={depth_str:<4}, '
              f'split={bp["min_samples_split"]}, leaf={bp["min_samples_leaf"]}')

        results.append({
            'Neuron': name, 'Config': config_label,
            'Accuracy': acc_mean, 'Acc_Std': acc_std,
            'F1_Macro': f1_mean, 'F1_Std': f1_std,
            'n_estimators': bp['n_estimators'],
            'max_depth': bp['max_depth'],
            'min_samples_split': bp['min_samples_split'],
            'min_samples_leaf': bp['min_samples_leaf'],
            'Samples': len(df)
        })

    return results


# ============================================================================
# 4. PLOTTING FUNCTIONS
# ============================================================================

def get_vals(df, col, neurons):
    """Extract values from df for a given column, aligned to a neuron list."""
    return [df[df['Neuron'] == n][col].values[0]
            if n in df['Neuron'].values else 0 for n in neurons]


def plot_accuracy_comparison(all_dfs, output_path):
    """Bar chart comparing accuracy across all configurations and neuron types."""
    # Collect all unique neurons across all configs
    all_neurons = []
    for df in all_dfs.values():
        for n in df['Neuron'].tolist():
            if n not in all_neurons:
                all_neurons.append(n)

    n_configs = len(all_dfs)
    x = np.arange(len(all_neurons))
    w = 0.8 / n_configs  # bar width

    fig, ax = plt.subplots(figsize=(16, 8))

    for i, (label, df) in enumerate(all_dfs.items()):
        config = next(c for c in TRAIN_CONFIGS if c['label'] == label)
        acc = get_vals(df, 'Accuracy', all_neurons)
        std = get_vals(df, 'Acc_Std', all_neurons)
        offset = (i - n_configs / 2 + 0.5) * w
        bars = ax.bar(x + offset, acc, w, yerr=std,
                      label=f'{label} ({config["n_features"]}ch)',
                      color=config['color'], edgecolor='white',
                      capsize=3, alpha=0.9)
        for b, a in zip(bars, acc):
            if a > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                        f'{a:.3f}', ha='center', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(all_neurons, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Accuracy (5-Fold CV)', fontsize=12)
    ax.set_title('Accuracy Comparison per Neuron Type — All Configurations\n'
                 '(Optimized Grid Search)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1/6, color='grey', ls='--', alpha=0.5, label='Chance (1/6)')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\u2714 Accuracy comparison plot saved: {output_path}')


def plot_delta(df1, df2, label1, label2, neurons, output_path):
    """Bar chart showing accuracy delta between two configurations."""
    a1 = get_vals(df1, 'Accuracy', neurons)
    a2 = get_vals(df2, 'Accuracy', neurons)
    delta = [v1 - v2 for v1, v2 in zip(a1, a2)]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#27ae60' if d >= 0 else '#c0392b' for d in delta]
    bars = ax.bar(neurons, delta, color=colors, edgecolor='white', alpha=0.85)
    for b, d in zip(bars, delta):
        va = 'bottom' if d >= 0 else 'top'
        off = 0.002 if d >= 0 else -0.008
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + off,
                f'{d:+.3f}', ha='center', va=va, fontsize=9, fontweight='bold')
    ax.axhline(y=0, color='black', lw=0.8)
    ax.set_ylabel(f'\u0394 Accuracy ({label1} \u2212 {label2})', fontsize=11)
    ax.set_title(f'Accuracy Delta: {label1} vs {label2}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\u2714 Delta plot saved: {output_path}')


# ============================================================================
# 5. MAIN
# ============================================================================

if __name__ == '__main__':
    print('=' * 80)
    print('  UNIFIED TRAINING — ALL 24 NEUROMORPHIC MODELS')
    print('=' * 80)
    print()
    print(f'Total HP combinations per neuron: {n_comb}')
    print(f'Trainings per neuron (x5 fold):   {n_comb * 5}')
    print(f'Total models to train:            {len(TRAIN_CONFIGS)} configs × 8 neurons = 24')
    print()

    all_results = {}   # label -> list[dict]
    all_dfs     = {}   # label -> DataFrame

    # --- Train all configurations ---
    for cfg in TRAIN_CONFIGS:
        print('=' * 80)
        print(f'  GRID SEARCH + TRAINING — {cfg["label"].upper()} '
              f'({cfg["n_features"]} features, prefix={cfg["prefix"]})')
        print(f'  Dataset: {cfg["data_dir"]}')
        print(f'  Output:  {cfg["output_dir"]}')
        print('=' * 80)
        print()

        results = train_and_save(
            data_dir=cfg['data_dir'],
            output_dir=cfg['output_dir'],
            config_label=cfg['label'],
            spike_prefix=cfg['prefix'],
            neuron_cols=cfg['features'],
            n_features=cfg['n_features'],
        )

        all_results[cfg['label']] = results
        if results:
            df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
            all_dfs[cfg['label']] = df
            print(f'\n\u2714 {len(results)} models saved for {cfg["label"]}')
        else:
            print(f'\n\u2718 No models trained for {cfg["label"]}')
        print()

    # --- Summary ---
    total_models = sum(len(r) for r in all_results.values())
    print('=' * 80)
    print(f'  TRAINING COMPLETE — {total_models} / 24 models')
    print('=' * 80)
    print()

    # --- Plots ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(all_dfs) > 0:
        # 1) Accuracy comparison across all configs
        plot_accuracy_comparison(
            all_dfs,
            os.path.join(OUTPUT_DIR, 'accuracy_comparison_all.png')
        )

        # 2) Delta plots for each pair of configs
        labels = list(all_dfs.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                l1, l2 = labels[i], labels[j]
                # Use union of neurons from both configs
                neurons = list(dict.fromkeys(
                    all_dfs[l1]['Neuron'].tolist() + all_dfs[l2]['Neuron'].tolist()
                ))
                plot_delta(
                    all_dfs[l1], all_dfs[l2], l1, l2, neurons,
                    os.path.join(OUTPUT_DIR, f'delta_{l1}_vs_{l2}.png')
                )

    # --- Excel Summary ---
    if all_dfs:
        df_all = pd.concat(all_dfs.values(), ignore_index=True)
        xlsx = os.path.join(OUTPUT_DIR, 'summary_results_all.xlsx')

        with pd.ExcelWriter(xlsx, engine='openpyxl') as w:
            df_all.to_excel(w, sheet_name='All', index=False)
            for label, df in all_dfs.items():
                sheet_name = label[:31]  # Excel sheet name limit
                df.to_excel(w, sheet_name=sheet_name, index=False)

            # Comparison pivot: accuracy per neuron × config
            pivot = df_all.pivot_table(
                index='Neuron', columns='Config',
                values=['Accuracy', 'F1_Macro'], aggfunc='first'
            )
            pivot.columns = [f'{col}_{cfg}' for col, cfg in pivot.columns]
            pivot = pivot.reset_index()
            pivot.to_excel(w, sheet_name='Comparison', index=False)

        print(f'\u2714 Excel summary saved: {xlsx}')

    # --- List all saved models ---
    print()
    for cfg in TRAIN_CONFIGS:
        pkls = sorted(glob.glob(os.path.join(cfg['output_dir'], '*.pkl')))
        print(f'--- {cfg["label"].upper()} ({len(pkls)} models) ---')
        for f in pkls:
            print(f'  {os.path.basename(f):40s} ({os.path.getsize(f)/1024:>6.0f} KB)')
        print()

    print('\u2714 All done!')
