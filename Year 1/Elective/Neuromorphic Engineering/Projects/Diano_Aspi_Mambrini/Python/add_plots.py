"""
Add interesting plots to train_models.ipynb.
Appends new markdown + code cells after the existing content.
"""
import json, os

nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "train_models.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

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
# 9. Accuracy Heatmap (Config × Neuron)
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md([
    "## 9. Accuracy Heatmap — Configuration × Neuron Type",
    "",
    "A heatmap view that makes it easy to spot which combinations perform best (green) vs worst (red).",
]))

cells.append(code("""\
if all_dfs:
    # Pivot: rows = neuron types, columns = configurations
    df_all = pd.concat(all_dfs.values(), ignore_index=True)
    pivot = df_all.pivot_table(index='Neuron', columns='Config',
                               values='Accuracy', aggfunc='first')

    # Reorder columns to match TRAIN_CONFIGS order
    col_order = [c['label'] for c in TRAIN_CONFIGS if c['label'] in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn',
                vmin=pivot.values.min() - 0.02, vmax=1.0,
                linewidths=0.6, linecolor='gray',
                cbar_kws={'label': 'Accuracy (5-Fold CV)'},
                ax=ax)
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Neuron Type', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Heatmap — Config × Neuron',
                 fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.show()\
"""))

# ═══════════════════════════════════════════════════════════════════════════
# 10. F1 Macro Heatmap
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md([
    "## 10. F1 Macro Heatmap — Configuration × Neuron Type",
    "",
    "Same heatmap but for the F1 Macro score.",
]))

cells.append(code("""\
if all_dfs:
    pivot_f1 = df_all.pivot_table(index='Neuron', columns='Config',
                                   values='F1_Macro', aggfunc='first')
    pivot_f1 = pivot_f1[[c['label'] for c in TRAIN_CONFIGS if c['label'] in pivot_f1.columns]]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='RdYlGn',
                vmin=pivot_f1.values.min() - 0.02, vmax=1.0,
                linewidths=0.6, linecolor='gray',
                cbar_kws={'label': 'F1 Macro (5-Fold CV)'},
                ax=ax)
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Neuron Type', fontsize=12, fontweight='bold')
    ax.set_title('F1 Macro Heatmap — Config × Neuron',
                 fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.show()\
"""))

# ═══════════════════════════════════════════════════════════════════════════
# 11. Mean Accuracy per Configuration (box/violin)
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md([
    "## 11. Accuracy Distribution per Configuration",
    "",
    "How much variance is there across the 8 neuron types within each configuration?",
    "A violin + strip plot reveals the distribution shape.",
]))

cells.append(code("""\
if all_dfs:
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    config_colors = {c['label']: c['color'] for c in TRAIN_CONFIGS}
    palette = [config_colors[l] for l in col_order]

    sns.violinplot(data=df_all, x='Config', y='Accuracy',
                   order=col_order, palette=palette,
                   inner=None, alpha=0.35, ax=ax)
    sns.stripplot(data=df_all, x='Config', y='Accuracy',
                  order=col_order, palette=palette,
                  size=8, edgecolor='white', linewidth=1, jitter=0.15, ax=ax)

    # Mean markers
    for i, cfg in enumerate(col_order):
        sub = df_all[df_all['Config'] == cfg]
        mean_val = sub['Accuracy'].mean()
        ax.plot(i, mean_val, marker='D', color='black', markersize=10, zorder=5)
        ax.annotate(f'{mean_val:.4f}', (i, mean_val),
                    textcoords='offset points', xytext=(20, 0),
                    fontsize=9, fontweight='bold', color='#2c3e50')

    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (5-Fold CV)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Distribution per Configuration',
                 fontsize=14, fontweight='bold', pad=12)
    ax.axhline(y=1/6, color='grey', ls='--', alpha=0.5, label='Chance (1/6)')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()\
"""))

# ═══════════════════════════════════════════════════════════════════════════
# 12. Radar / Spider Chart — Per Neuron
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md([
    "## 12. Radar Chart — Accuracy per Neuron Across Configurations",
    "",
    "A radar chart showing how each configuration performs on all 8 neuron types simultaneously.",
]))

cells.append(code("""\
if all_dfs:
    from math import pi

    all_neurons = []
    for df in all_dfs.values():
        for n in df['Neuron'].tolist():
            if n not in all_neurons:
                all_neurons.append(n)
    all_neurons = sorted(all_neurons)

    N = len(all_neurons)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    for label, df in all_dfs.items():
        cfg = next(c for c in TRAIN_CONFIGS if c['label'] == label)
        values = []
        for n in all_neurons:
            row = df[df['Neuron'] == n]
            values.append(row['Accuracy'].values[0] if len(row) > 0 else 0)
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f"{label} ({cfg['n_features']}ch)",
                color=cfg['color'])
        ax.fill(angles, values, alpha=0.08, color=cfg['color'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_neurons, fontsize=10)
    ax.set_ylim(0.7, 1.0)
    ax.set_title('Accuracy per Neuron — All Configurations',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=10)
    plt.tight_layout()
    plt.show()\
"""))

# ═══════════════════════════════════════════════════════════════════════════
# 13. Ranking chart
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md([
    "## 13. Neuron Ranking per Configuration",
    "",
    "Within each configuration, rank the neurons from best to worst. A bump chart style showing how neuron rankings shift across configurations.",
]))

cells.append(code("""\
if all_dfs:
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    configs_list = [c['label'] for c in TRAIN_CONFIGS if c['label'] in all_dfs]
    n_configs = len(configs_list)
    neuron_colors = dict(zip(sorted(all_neurons),
                             sns.color_palette("tab10", len(all_neurons))))

    for neuron in sorted(all_neurons):
        ranks = []
        for cfg_label in configs_list:
            df = all_dfs[cfg_label]
            df_sorted = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
            neurons_ranked = df_sorted['Neuron'].tolist()
            if neuron in neurons_ranked:
                ranks.append(neurons_ranked.index(neuron) + 1)
            else:
                ranks.append(None)

        x_positions = list(range(n_configs))
        valid = [(x, r) for x, r in zip(x_positions, ranks) if r is not None]
        if valid:
            xs, rs = zip(*valid)
            ax.plot(xs, rs, 'o-', linewidth=2.2, markersize=8,
                    label=neuron, color=neuron_colors[neuron],
                    markeredgecolor='white', markeredgewidth=1.2)

    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(configs_list, fontsize=11, fontweight='bold')
    ax.set_ylabel('Rank (1 = Best)', fontsize=12, fontweight='bold')
    ax.set_title('Neuron Ranking Across Configurations',
                 fontsize=14, fontweight='bold', pad=12)
    ax.invert_yaxis()
    ax.set_ylim(len(all_neurons) + 0.5, 0.5)
    ax.set_yticks(range(1, len(all_neurons) + 1))
    ax.legend(title='Neuron', fontsize=9, title_fontsize=10,
              loc='center left', bbox_to_anchor=(1.01, 0.5))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()\
"""))

# ═══════════════════════════════════════════════════════════════════════════
# 14. Best HP Distribution
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md([
    "## 14. Best Hyperparameter Distribution",
    "",
    "What hyperparameters does GridSearchCV prefer across all 32 models?",
]))

cells.append(code("""\
if all_dfs:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Best Hyperparameter Distribution (All 32 Models)',
                 fontsize=15, fontweight='bold', y=1.01)

    hp_names = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
    hp_labels = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']

    for ax, hp, label in zip(axes.flat, hp_names, hp_labels):
        values = df_all[hp].fillna('None').astype(str).value_counts().sort_index()
        colors = sns.color_palette("Set2", len(values))
        bars = ax.bar(values.index, values.values, color=colors, edgecolor='white')
        for bar, val in zip(bars, values.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha='center', fontsize=10, fontweight='bold')
        ax.set_xlabel(label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Distribution of {label}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()\
"""))

# ═══════════════════════════════════════════════════════════════════════════
# 15. Accuracy vs Std scatter
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md([
    "## 15. Accuracy vs Standard Deviation",
    "",
    "Higher accuracy with lower variance is ideal (top-left corner). This scatter plot reveals which models are both accurate and stable.",
]))

cells.append(code("""\
if all_dfs:
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    for label, df in all_dfs.items():
        cfg = next(c for c in TRAIN_CONFIGS if c['label'] == label)
        ax.scatter(df['Acc_Std'], df['Accuracy'],
                   s=100, color=cfg['color'], edgecolors='white',
                   linewidth=1.2, label=f"{label} ({cfg['n_features']}ch)",
                   zorder=4)
        # Annotate with neuron names
        for _, row in df.iterrows():
            ax.annotate(row['Neuron'], (row['Acc_Std'], row['Accuracy']),
                        textcoords='offset points', xytext=(6, 4),
                        fontsize=7, color='#555')

    ax.set_xlabel('Accuracy Std Dev (5-Fold CV)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Stability — All Models',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Highlight ideal zone
    ax.axhline(y=0.97, color='green', ls=':', alpha=0.5)
    ax.axvline(x=0.015, color='green', ls=':', alpha=0.5)
    ax.annotate('Ideal zone', xy=(0.005, 0.99), fontsize=10,
                color='green', fontweight='bold', alpha=0.7)

    plt.tight_layout()
    plt.show()\
"""))

# ═══════════════════════════════════════════════════════════════════════════
# 16. Configuration comparison summary
# ═══════════════════════════════════════════════════════════════════════════
cells.append(md([
    "## 16. Configuration Summary — Mean & Best Performance",
    "",
    "Aggregate view: mean accuracy, best accuracy, and mean F1 for each configuration.",
]))

cells.append(code("""\
if all_dfs:
    summary_rows = []
    for label in col_order:
        df = all_dfs[label]
        cfg = next(c for c in TRAIN_CONFIGS if c['label'] == label)
        summary_rows.append({
            'Config': label,
            'Features': cfg['n_features'],
            'Mean Acc': f"{df['Accuracy'].mean():.4f}",
            'Best Acc': f"{df['Accuracy'].max():.4f}",
            'Worst Acc': f"{df['Accuracy'].min():.4f}",
            'Spread': f"{df['Accuracy'].max() - df['Accuracy'].min():.4f}",
            'Mean F1': f"{df['F1_Macro'].mean():.4f}",
            'Best Neuron': df.loc[df['Accuracy'].idxmax(), 'Neuron'],
            'Worst Neuron': df.loc[df['Accuracy'].idxmin(), 'Neuron'],
        })
    summary_table = pd.DataFrame(summary_rows)
    display(summary_table.style.set_caption("Configuration Summary"))\
"""))

# ═══════════════════════════════════════════════════════════════════════════
# Write
# ═══════════════════════════════════════════════════════════════════════════
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

n_new = len(cells) - 18  # was 18 cells originally
print(f"Added {n_new} new cells ({n_new//2} code + {n_new//2} markdown)")
print(f"Total cells: {len(cells)}")
print("Done!")
