import numpy as np
import matplotlib.pyplot as plt

# ファイルを読み込み
ab = np.loadtxt('model_comparison_results/bookcorpus_llama_llama_ab.dat')

# データを抽出
n_layers = len(ab)
layers = np.arange(n_layers)

# i → 初期層（layer 0）
ii_to_first = ab[:, 0]

# i → 最終層（layer -1）
ii_to_last = ab[:, -1]

# i → i+1層（隣接層）
ii_adjacent = np.array([ab[i, i+1] if i < n_layers - 1 else 0 for i in range(n_layers)])

# グラフ設定
fig, ax = plt.subplots(figsize=(14, 7))

# プロット（3つのシリーズ）
ax.plot(layers, ii_to_first, marker='o', markersize=6, linewidth=2.5,
        color='#9333ea', markerfacecolor='#9333ea', markeredgecolor='white',
        markeredgewidth=1.5, label='Δ(layer i → first)', alpha=0.9)

ax.plot(layers, ii_to_last, marker='s', markersize=6, linewidth=2.5,
        color='#dc2626', markerfacecolor='#dc2626', markeredgecolor='white',
        markeredgewidth=1.5, label='Δ(layer i → last)', alpha=0.9)

ax.plot(layers, ii_adjacent, marker='^', markersize=6, linewidth=2.5,
        color='#2563eb', markerfacecolor='#2563eb', markeredgecolor='white',
        markeredgewidth=1.5, label='Δ(layer i → i+1)', alpha=0.9)

# 軸設定
ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
ax.set_ylabel('Information Imbalance', fontsize=13, fontweight='bold')
ax.set_title('Information Imbalance Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(layers)
ax.set_xticklabels([f'L{i}' for i in layers])
ax.set_ylim(0, max(ii_to_first.max(), ii_to_last.max(), ii_adjacent.max()) * 1.1)

# グリッド
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# スパイン調整
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# レジェンド
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

plt.tight_layout()
plt.savefig('model_comparison_results/comparison_ii.png', dpi=300, bbox_inches='tight')
print("Saved to model_comparison_results/comparison_ii.png")
