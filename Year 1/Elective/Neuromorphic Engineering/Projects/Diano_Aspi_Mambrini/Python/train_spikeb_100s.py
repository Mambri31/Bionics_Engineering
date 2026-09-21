import os
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, "Dataset", "Blind", "SpikeB_10ms_Resonator.csv")

def fix_decimal_commas(filepath):
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

def load_spike_dataset_5ch(filepath):
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

PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

if __name__ == "__main__":
    print(f"Loading dataset: {FILE_PATH}")
    df = load_spike_dataset_5ch(FILE_PATH)
    print(f"Dataset shape: {df.shape}")
    
    X = df[['N1', 'N2', 'N3', 'N4', 'N5']].values
    y = df['label'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Running GridSearchCV on the full dataset (5-fold CV)...")
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        param_grid=PARAM_GRID,
        cv=skf,
        scoring={'accuracy': 'accuracy', 'f1': 'f1_macro'},
        refit='accuracy',
        n_jobs=-1,
        return_train_score=False
    )
    grid.fit(X, y)
    
    best_model = grid.best_estimator_
    bp = grid.best_params_
    bi = grid.best_index_
    
    acc_mean = grid.cv_results_['mean_test_accuracy'][bi]
    acc_std  = grid.cv_results_['std_test_accuracy'][bi]
    f1_mean  = grid.cv_results_['mean_test_f1'][bi]
    f1_std   = grid.cv_results_['std_test_f1'][bi]
    
    print("\n--- BEST MODEL PARAMETERS ---")
    print(bp)
    
    print("\n--- SCORES (5-Fold CV) ---")
    print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"F1 Macro: {f1_mean:.4f} ± {f1_std:.4f}")
    
    print("\nGenerating cross-validated predictions for Confusion Matrix...")
    y_pred_cv = cross_val_predict(best_model, X, y, cv=skf, n_jobs=-1)
    
    print("\n--- Classification Report ---")
    print(classification_report(y, y_pred_cv))
    
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y, y_pred_cv))
