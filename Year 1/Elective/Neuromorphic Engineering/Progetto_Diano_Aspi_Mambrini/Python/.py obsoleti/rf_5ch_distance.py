"""
Random Forest Gesture Classifier — 5 Distance Channels (1-second windows)
==========================================================================
This script:
  1. Loads the preprocessed1.csv hand-tracking dataset.
  2. Computes Euclidean distances from the palm to each of the 5 fingertips.
     If a finger is flagged as missing (ch_X_missing == 1), the distance
     is padded with 0.
  3. Groups samples into 1-second intervals (matching the 1 Hz UDP send rate)
     and averages the 5 distance channels within each window.
  4. Trains a Random Forest classifier on these 5 aggregated features.
  5. Evaluates the model and saves all outputs (model, plots, scaler).

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

Author: Generated with AI assistance
"""

import os

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for server / LabVIEW
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "Dataset", "preprocessed1.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
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

# RF hyper-parameters
TEST_SIZE = 0.25
RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_DEPTH = None          # unlimited depth
TIME_WINDOW_SEC = 1.0     # 1-second aggregation window


# ============================================================================
# 2. DATA LOADING
# ============================================================================

def load_dataset(path: str) -> pd.DataFrame:
    """Load preprocessed1.csv and return a DataFrame with its header row."""
    print(f"[LOAD] Reading: {os.path.basename(path)}")
    df = pd.read_csv(path)
    print(f"   Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    return df


# ============================================================================
# 3. FEATURE EXTRACTION — Euclidean distances (palm → fingertip)
# ============================================================================

def compute_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each of the 5 fingers, compute the 3-D Euclidean distance
    from the palm centre to the fingertip.

    If the finger is missing (ch_X_missing == 1), the distance is set to 0
    (zero-padding), as requested.

    Returns
    -------
    df : DataFrame
        The same DataFrame with 5 new columns: dist_1 .. dist_5.
    """
    palm_cols = ["palm_x", "palm_y", "palm_z"]

    for i in range(1, 6):
        tip_cols = [f"ch_{i}_x", f"ch_{i}_y", f"ch_{i}_z"]
        missing_col = f"ch_{i}_missing"

        # Vectorised Euclidean distance
        dx = df[tip_cols[0]].values - df[palm_cols[0]].values
        dy = df[tip_cols[1]].values - df[palm_cols[1]].values
        dz = df[tip_cols[2]].values - df[palm_cols[2]].values
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Zero-pad where the finger data is missing
        missing_mask = df[missing_col].values == 1
        dist[missing_mask] = 0.0

        df[f"dist_{i}"] = dist

    print("[FEAT] Computed 5 Euclidean-distance channels (palm → fingertip).")
    n_missing = (df[[f"ch_{i}_missing" for i in range(1, 6)]] == 1).sum()
    for i in range(1, 6):
        pct = n_missing[f"ch_{i}_missing"] / len(df) * 100
        print(f"   ch_{i}: {n_missing[f'ch_{i}_missing']:,} missing rows "
              f"({pct:.1f}%) → padded with 0")

    return df


# ============================================================================
# 4. TIME AGGREGATION — 1-second window mean
# ============================================================================

def aggregate_by_time_window(df: pd.DataFrame,
                              window_sec: float = 1.0) -> pd.DataFrame:
    """
    Group samples into non-overlapping windows of `window_sec` seconds
    based on the `timestamp` column.

    Within each window the majority class is taken as the label and the
    5 distance channels are averaged.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: timestamp, class, dist_1..dist_5.
    window_sec : float
        Window duration in seconds (default 1.0 s).

    Returns
    -------
    agg : DataFrame
        Aggregated DataFrame with columns:
        window_start, class, dist_1_mean .. dist_5_mean.
    """
    dist_cols = [f"dist_{i}" for i in range(1, 6)]

    # Assign each row to a time-window bucket
    t0 = df["timestamp"].min()
    df = df.copy()
    df["window_id"] = ((df["timestamp"] - t0) / window_sec).astype(int)

    # Aggregate: mean distances, majority-vote class, window start time
    agg = df.groupby("window_id").agg(
        window_start=("timestamp", "min"),
        **{f"dist_{i}_mean": (f"dist_{i}", "mean") for i in range(1, 6)},
        # Majority class via lambda (mode)
        gesture_class=("class", lambda s: s.mode().iloc[0]),
        n_samples=("timestamp", "size"),
    ).reset_index(drop=True)

    print(f"[AGG]  Aggregated into {len(agg):,} windows of {window_sec}s "
          f"(avg {agg['n_samples'].mean():.0f} samples/window).")

    return agg


# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

def train_random_forest(X_train: np.ndarray,
                        y_train: np.ndarray,
                        n_estimators: int = 200,
                        max_depth: int | None = None,
                        random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest classifier and return the fitted model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


# ============================================================================
# 6. EVALUATION & VISUALISATION
# ============================================================================

def evaluate_model(model: RandomForestClassifier,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   feature_names: list[str],
                   output_dir: str) -> float:
    """
    Print classification metrics, save confusion-matrix & feature-importance
    plots, and return accuracy.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
    target_names = [GESTURE_LABELS.get(c, f"Class {c}") for c in unique_classes]

    # ---- Console output ----
    print(f"\n[RESULT] Test Accuracy: {acc * 100:.2f}%")
    print("\n[REPORT] Classification Report:")
    print("-" * 70)
    print(classification_report(y_test, y_pred,
                                labels=unique_classes,
                                target_names=target_names))

    # ---- Feature importance (console) ----
    importances = model.feature_importances_
    print("[KEY] Feature Importance:")
    for feat, imp in zip(feature_names, importances):
        bar = "#" * int(imp * 50)
        print(f"   {feat}: {imp:.4f}  {bar}")

    # ---- Confusion matrix plot ----
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    cm_norm = confusion_matrix(y_test, y_pred,
                               labels=unique_classes, normalize="true")

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
        f"Random Forest — 5 Distance Channels (1s windows)\n"
        f"Accuracy: {acc * 100:.2f}%  |  Estimators: {N_ESTIMATORS}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix_5ch_dist.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SAVE] Confusion matrix  → {cm_path}")

    # ---- Feature importance bar chart ----
    fig2, ax = plt.subplots(figsize=(10, 5))
    finger_labels = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    colors = sns.color_palette("viridis", 5)
    bars = ax.barh(finger_labels, importances, color=colors, edgecolor="black")
    ax.set_xlabel("Importance", fontsize=12, fontweight="bold")
    ax.set_title("Feature Importance — Palm-to-Fingertip Distance",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{imp:.4f}", va="center", fontsize=10)
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "feature_importance_5ch_dist.png")
    fig2.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[SAVE] Feature importance → {fi_path}")

    return acc


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("  RANDOM FOREST — 5 DISTANCE CHANNELS (1-SECOND WINDOWS)")
    print("=" * 70)

    # --- Load ---
    df = load_dataset(DATASET_PATH)

    # --- Class distribution ---
    print("\n[INFO] Class distribution (raw samples):")
    for cls_id, count in df["class"].value_counts().sort_index().items():
        label = GESTURE_LABELS.get(cls_id, f"Class {cls_id}")
        print(f"   Class {cls_id} ({label}): {count:,}")

    # --- Feature extraction ---
    df = compute_distances(df)

    # --- Time aggregation (1-second windows) ---
    agg = aggregate_by_time_window(df, window_sec=TIME_WINDOW_SEC)

    # Show aggregated class distribution
    print("\n[INFO] Class distribution (after 1s aggregation):")
    for cls_id, count in (agg["gesture_class"]
                          .value_counts().sort_index().items()):
        label = GESTURE_LABELS.get(cls_id, f"Class {cls_id}")
        print(f"   Class {cls_id} ({label}): {count:,}")

    # --- Prepare features & labels ---
    feature_cols = [f"dist_{i}_mean" for i in range(1, 6)]
    X = agg[feature_cols].values
    y = agg["gesture_class"].values

    print(f"\n[DATA] Feature matrix shape: {X.shape}")
    print(f"[DATA] Feature names: {feature_cols}")

    # --- Train / test split (stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n[SPLIT] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # --- Train ---
    print(f"\n[TRAIN] Training Random Forest "
          f"(n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH})...")
    model = train_random_forest(X_train, y_train,
                                n_estimators=N_ESTIMATORS,
                                max_depth=MAX_DEPTH,
                                random_state=RANDOM_STATE)
    print("   ✓ Model trained successfully.")

    # --- Evaluate ---
    evaluate_model(model, X_test, y_test, feature_cols, OUTPUT_DIR)

    # --- Save model ---
    model_path = os.path.join(OUTPUT_DIR, "rf_model_5ch_dist.pkl")
    joblib.dump(model, model_path)
    print(f"\n[SAVE] Trained model → {model_path}")

    print("\n" + "=" * 70)
    print("  ✓ PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
