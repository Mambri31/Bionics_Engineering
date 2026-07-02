import json, os

nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_models_standard.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Step 1: Replace Thumb, Index, etc. with Channel 1, 2... in plot_confusion_matrix (cell 16)
source_16 = nb['cells'][16]['source']
for i, line in enumerate(source_16):
    if 'display_labels = ["Thumb", "Index", "Middle", "Ring", "Pinky"]' in line:
        source_16[i] = line.replace('["Thumb", "Index", "Middle", "Ring", "Pinky"]', '["Channel 1", "Channel 2", "Channel 3", "Channel 4", "Channel 5"]')
    elif 'display_labels = (["Thumb (dist)", "Index (dist)", "Middle (dist)",' in line:
        source_16[i] = line.replace('["Thumb (dist)", "Index (dist)", "Middle (dist)",', '["Channel 1 (dist)", "Channel 2 (dist)", "Channel 3 (dist)",')
    elif '"Ring (dist)", "Pinky (dist)"]' in line:
        source_16[i] = line.replace('"Ring (dist)", "Pinky (dist)"]', '"Channel 4 (dist)", "Channel 5 (dist)"]')

nb['cells'][16]['source'] = source_16

# Step 2: Append new cells
def md(source):
    if isinstance(source, str):
        source = source.split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source":
            [l + "\n" if i < len(source)-1 else l for i, l in enumerate(source)]}

def code(source):
    if isinstance(source, str):
        lines = source.split("\n")
    else:
        lines = source
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" if i < len(lines)-1 else l for i, l in enumerate(lines)]}

cells = nb["cells"]

# ═══════════════════════════════════════════════════════════════════════════
# 10. Feature Importance Comparison
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md([
    "## 10. Feature Importance Comparison",
    "",
    "A side-by-side comparison of the feature importances for the 5 shared distance channels. This shows how the model re-weights the spatial channels when directional information is added."
]))

cells.append(code("""\
if 'trained_models' in locals() and len(trained_models) == 2:
    m1_name = MODEL_CONFIGS[0]['name']
    m2_name = MODEL_CONFIGS[1]['name']
    
    m1 = trained_models[m1_name]
    m2 = trained_models[m2_name]
    
    # Extract importances for the first 5 features (distances)
    imp1 = m1.feature_importances_[:5]
    imp2 = m2.feature_importances_[:5]
    
    labels = ["Channel 1 (dist)", "Channel 2 (dist)", "Channel 3 (dist)", "Channel 4 (dist)", "Channel 5 (dist)"]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, imp1, width, label=f'{m1_name} (5 features)', color=MODEL_CONFIGS[0]['color'], edgecolor='white')
    bars2 = ax.bar(x + width/2, imp2, width, label=f'{m2_name} (8 features)', color=MODEL_CONFIGS[1]['color'], edgecolor='white')
    
    ax.set_ylabel('Feature Importance (Gini)', fontsize=12, fontweight='bold')
    ax.set_title('Importance of Distance Features: Blind vs BlindDir', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f'{b.get_height():.3f}', ha='center', fontsize=9)
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f'{b.get_height():.3f}', ha='center', fontsize=9)
                
    plt.tight_layout()
    plt.show()
"""))

cells.append(md([
    "## 11. Hyperparameters Distribution (Radar)",
    "",
    "Visualizing the best hyperparameters found by GridSearchCV for both models on a radar chart."
]))

cells.append(code("""\
if all_metrics:
    from math import pi
    
    hp_keys = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
    
    # Normalize values for radar chart
    max_vals = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }
    
    N = len(hp_keys)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    for i, config in enumerate(MODEL_CONFIGS):
        metric = next(m for m in all_metrics if m['Model'] == config['name'])
        
        values = []
        for k in hp_keys:
            val = metric[k]
            # Handle None for max_depth
            if k == 'max_depth' and val is None:
                val = 20 # Assume max depth allowed in grid or unconstrained
            
            # Normalize to 0-1
            norm_val = val / max_vals[k]
            values.append(norm_val)
            
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=config['name'], color=config['color'])
        ax.fill(angles, values, alpha=0.1, color=config['color'])
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(hp_keys, fontsize=11, fontweight='bold')
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], color="grey", size=8)
    ax.set_ylim(0, 1.1)
    ax.set_title('Selected Hyperparameters (Normalized)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=10)
    
    plt.tight_layout()
    plt.show()
"""))


with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Replaced channel labels and added 2 comparative plots!")
