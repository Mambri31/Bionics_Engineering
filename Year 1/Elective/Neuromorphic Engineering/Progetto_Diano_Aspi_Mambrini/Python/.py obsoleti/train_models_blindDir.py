import os, glob, re
import pandas as pd
import numpy as np
import joblib
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# removed magic command
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

SCRIPT_DIR  = os.getcwd()
DATA_DIR    = os.path.join(SCRIPT_DIR, "Dataset", "blindDir_channel")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "Models", "Neuromorphic", "model_blind_dir")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 11 feature columns: 5 spike channels + 6 direction channels
NEURON_COLS = [
    'N1', 'N2', 'N3', 'N4', 'N5',
    'dir_x_pos', 'dir_y_pos', 'dir_z_pos',
    'dir_x_neg', 'dir_y_neg', 'dir_z_neg'
]

print(f'Dataset directory:  {DATA_DIR}')
print(f'Model output:       {OUTPUT_DIR}')
print(f'Feature columns:    {len(NEURON_COLS)} ({NEURON_COLS})')

PARAM_GRID = {
    'n_estimators':     [50, 100, 200],
    'max_depth':        [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2]
}

n_comb = 1
for v in PARAM_GRID.values():
    n_comb *= len(v)
print(f'Total combinations per neuron: {n_comb}')
print(f'Trainings per neuron (x5 fold): {n_comb * 5}')

def fix_decimal_commas(filepath):
    """Fixes CSV file with decimal comma (15 raw columns -> 8 real columns).
    
    Note: blindDir_channel data has 14 columns with decimal dots,
    so this function returns None (no fix needed) for that format.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
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

def load_spike_dataset(filepath):
    """Loads a spike dataset from blindDir_channel format.
    
    Expected format: 14 columns
        timestamp, label, N1..N5, dir_x_pos, dir_y_pos, dir_z_pos,
        dir_x_neg, dir_y_neg, dir_z_neg, total_spike_sum
    
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
    if fixed is not None:
        text = chr(10).join(fixed)  # chr(10) = newline
        df = pd.read_csv(StringIO(text), header=None, names=col_names)
    else:
        df = pd.read_csv(filepath, header=None, names=col_names)
    df = df.dropna()
    df = df.drop(columns=['timestamp', 'sum'])
    df['label'] = df['label'].astype(int)
    return df

def train_and_save(data_dir, output_dir, config_label,
                   spike_prefix='SpikeBD_', neuron_cols=None):
    """Grid Search + train + save for each neuron in the directory."""
    if neuron_cols is None:
        neuron_cols = NEURON_COLS
    files = sorted(glob.glob(os.path.join(data_dir, spike_prefix + '*.csv')))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for filepath in files:
        name = extract_neuron_name(filepath)
        df = load_spike_dataset(filepath)
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

        # Final model on the whole dataset
        rf_final = RandomForestClassifier(
            **bp, random_state=42, n_jobs=-1, class_weight='balanced')
        rf_final.fit(X, y)

        # Saving
        os.makedirs(output_dir, exist_ok=True)
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

print('=' * 80)
print('  GRID SEARCH + TRAINING \u2014 BLIND DIR CHANNEL (11 Features)')
print('=' * 80)
print()
results_BD = train_and_save(DATA_DIR, OUTPUT_DIR, 'BlindDir',
                            spike_prefix='SpikeBD_')
print(f'{chr(10)}\u2714 {len(results_BD)} models saved')

df_BD = pd.DataFrame(results_BD).sort_values('Accuracy', ascending=False)
df_BD[['Neuron','Accuracy','Acc_Std','F1_Macro','F1_Std',
       'n_estimators','max_depth','min_samples_split','min_samples_leaf','Samples']].style.format({
    'Accuracy': '{:.4f}', 'Acc_Std': '{:.4f}',
    'F1_Macro': '{:.4f}', 'F1_Std': '{:.4f}'
}).background_gradient(subset=['Accuracy'], cmap='Greens')

# Preparation
order = df_BD.sort_values('Accuracy', ascending=False)['Neuron'].tolist()
x = np.arange(len(order))
w = 0.5

def get_vals(df, col, neurons):
    return [df[df['Neuron'] == n][col].values[0]
            if n in df['Neuron'].values else 0 for n in neurons]

acc_vals = get_vals(df_BD, 'Accuracy', order)
std_vals = get_vals(df_BD, 'Acc_Std', order)

# --- PLOT 1: Accuracy with error bars ---
fig, ax = plt.subplots(figsize=(14, 7))

bars = ax.bar(x, acc_vals, w, yerr=std_vals, label='BlindDir Channel (11 Features)',
              color='#27ae60', edgecolor='white', capsize=4, alpha=0.9)

for b, a in zip(bars, acc_vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012,
            f'{a:.3f}', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(order, rotation=45, ha='right', fontsize=11)
ax.set_ylabel('Accuracy (5-Fold CV)', fontsize=12)
ax.set_title('Accuracy per Neuron Type \u2014 BlindDir Channel\n'
             '(Optimized Grid Search, 11 features: N1\u2013N5 + 6 Direction)',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.12)
ax.axhline(y=1/6, color='grey', ls='--', alpha=0.5, label='Chance (1/6)')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_blindDir.png'), dpi=150, bbox_inches='tight')
plt.show()

# --- PLOT 2: F1-Macro Score ---
f1_vals = get_vals(df_BD, 'F1_Macro', order)
f1_std_vals = get_vals(df_BD, 'F1_Std', order)

fig, ax = plt.subplots(figsize=(14, 7))

bars = ax.bar(x, f1_vals, w, yerr=f1_std_vals, label='F1-Macro Score',
              color='#2980b9', edgecolor='white', capsize=4, alpha=0.9)

for b, a in zip(bars, f1_vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012,
            f'{a:.3f}', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(order, rotation=45, ha='right', fontsize=11)
ax.set_ylabel('F1-Macro (5-Fold CV)', fontsize=12)
ax.set_title('F1-Macro Score per Neuron Type \u2014 BlindDir Channel\n'
             '(Optimized Grid Search, 11 features)',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.12)
ax.axhline(y=1/6, color='grey', ls='--', alpha=0.5, label='Chance (1/6)')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'f1_macro_blindDir.png'), dpi=150, bbox_inches='tight')
plt.show()

# Summary table
cols = ['Neuron', 'Accuracy', 'Acc_Std', 'F1_Macro', 'F1_Std',
        'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'Samples']

# Save Excel
xlsx = os.path.join(OUTPUT_DIR, 'summary_results_blindDir.xlsx')
with pd.ExcelWriter(xlsx, engine='openpyxl') as w:
    df_BD.to_excel(w, sheet_name='BlindDir_Results', index=False)

print(f'\u2714 Excel saved: {xlsx}')
print(f'  Sheet: BlindDir_Results')
print()

df_BD[cols].style.format({
    'Accuracy': '{:.4f}', 'Acc_Std': '{:.4f}',
    'F1_Macro': '{:.4f}', 'F1_Std': '{:.4f}'
}).background_gradient(subset=['Accuracy'], cmap='Greens')

pkls = sorted(glob.glob(os.path.join(OUTPUT_DIR, '*.pkl')))
print(f'--- BLIND DIR CHANNEL ({len(pkls)} models) ---')
for f in pkls:
    print(f'  {os.path.basename(f):40s} ({os.path.getsize(f)/1024:>6.0f} KB)')
print()
print(f'\u2714 All models saved to: {OUTPUT_DIR}')
