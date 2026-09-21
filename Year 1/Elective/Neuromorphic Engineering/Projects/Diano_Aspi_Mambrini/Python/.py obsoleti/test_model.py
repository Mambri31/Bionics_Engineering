import pandas as pd
import joblib
import numpy as np

model_path = "Models/Neuromorphic/model_blind_dir/rf_model_Thalamo1.pkl"
data_path = "Dataset/blindDir_channel/SpikeBD_20260612175719_Thalamo1.csv"

# Load Model
model = joblib.load(model_path)
print("Model loaded successfully.")
print("Expected features:", model.n_features_in_)

# Load Data
col_names = [
    'timestamp', 'label',
    'N1', 'N2', 'N3', 'N4', 'N5',
    'dir_x_pos', 'dir_y_pos', 'dir_z_pos',
    'dir_x_neg', 'dir_y_neg', 'dir_z_neg',
    'sum'
]
df = pd.read_csv(data_path, header=None, names=col_names).dropna()
features = [
    'N1', 'N2', 'N3', 'N4', 'N5',
    'dir_x_pos', 'dir_y_pos', 'dir_z_pos',
    'dir_x_neg', 'dir_y_neg', 'dir_z_neg'
]

X = df[features].values
y = df['label'].values

# Predict first 10
preds = model.predict(X[:10])
print(f"First 10 True Labels: {y[:10]}")
print(f"First 10 Predictions: {preds}")

# Unique predictions overall
all_preds = model.predict(X)
print(f"Unique predictions across the dataset: {np.unique(all_preds)}")
