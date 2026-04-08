"""
visualize.py — Competition-ready visualization dashboard for N-gram Language Model
Run: python3 visualize.py
Produces: plots/  directory with 6 publication-quality figures
"""

import random
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import Counter

from data_preprocessing import get_tokenized_data, preprocess_data
from language_model import (
    count_n_grams, calculate_perplexity_batch,
)
from language_model_smoothing import (
    calculate_perplexity_with_method,
)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

COLORS = {
    "kneser_ney":    "#2ecc71",
    "stupid_backoff":"#3498db",
    "interpolation": "#9b59b6",
    "add_k":         "#e74c3c",
    "unigram":       "#f39c12",
    "bigram":        "#1abc9c",
    "trigram":       "#2980b9",
}

os.makedirs("plots", exist_ok=True)


print("Loading data…")
with open("./data/en_US.twitter.txt", "r", encoding="utf-8") as f:
    data = f.read()

tokenized = get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized)

split = int(len(tokenized) * 0.8)
train_raw, test_raw = tokenized[:split], tokenized[split:]
train_data, test_data, vocabulary = preprocess_data(train_raw, test_raw, 2)
vocab_size = len(vocabulary) + 2

print("Training n-gram models…")
ngrams = [count_n_grams(train_data, n) for n in range(1, 4)]


print("Figure 1: Add-k perplexity curve…")

k_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

fig, ax = plt.subplots(figsize=(9, 5))
for label, color in [("Unigram", COLORS["unigram"]), ("Bigram", COLORS["bigram"]), ("Trigram", COLORS["trigram"])]:
    pps = []
    for k in k_values:
        if label == "Unigram":
            mean_pp, _ = calculate_perplexity_batch(test_data, ngrams[0], ngrams[1], vocab_size, k=k)
        elif label == "Bigram":
            mean_pp, _ = calculate_perplexity_batch(test_data, ngrams[1], ngrams[2], vocab_size, k=k)
        else:
            mean_pp, _ = calculate_perplexity_with_method(test_data, ngrams, vocab_size, method="add_k", k=k)
        pps.append(mean_pp)
    ax.plot(k_values, pps, marker="o", label=label, color=color, linewidth=2)

ax.set_xlabel("Smoothing parameter k")
ax.set_ylabel("Mean Perplexity (lower = better)")
ax.set_title("Add-k Smoothing: Perplexity vs k")
ax.legend()
ax.set_yscale("log")
fig.tight_layout()
plt.close(fig)


print("Figure 2: Smoothing comparison bar chart…")

methods = {
    "Add-k\n(k=1.0)":        calculate_perplexity_with_method(test_data, ngrams, vocab_size, method="add_k",         k=1.0),
    "Linear\nInterpolation": calculate_perplexity_with_method(test_data, ngrams, vocab_size, method="interpolation",  lambdas=[0.2, 0.3, 0.5]),
    "Stupid\nBackoff":       calculate_perplexity_with_method(test_data, ngrams, vocab_size, method="stupid_backoff", backoff_factor=0.4),
    "Kneser-Ney":            calculate_perplexity_with_method(test_data, ngrams, vocab_size, method="kneser_ney",     discount=0.75),
}

labels  = list(methods.keys())
means   = [v[0] for v in methods.values()]
stds    = [min(v[1], v[0] * 0.5) for v in methods.values()]   # cap std for display
bar_colors = [COLORS["add_k"], COLORS["interpolation"],
              COLORS["stupid_backoff"], COLORS["kneser_ney"]]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, means, yerr=stds, capsize=6,
              color=bar_colors, alpha=0.85, edgecolor="white", linewidth=1.2)

# Annotate bars
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.01,
            f"{mean:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("Mean Perplexity (lower = better)")
ax.set_title("Smoothing Technique Comparison — Trigram Model")
ax.set_yscale("log")
fig.tight_layout()
plt.close(fig)


print("Figure 3: N-gram order vs perplexity…")

orders = [1, 2, 3]
k_vals_to_show = [0.5, 1.0, 2.0]
x = np.arange(len(orders))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
palette = ["#2ecc71", "#3498db", "#e74c3c"]

for i, k in enumerate(k_vals_to_show):
    pps = []
    for n in orders:
        mean_pp, _ = calculate_perplexity_batch(test_data, ngrams[n-1], ngrams[n], vocab_size, k=k)
        pps.append(mean_pp)
    ax.bar(x + i * width, pps, width, label=f"k={k}", color=palette[i], alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(["Unigram", "Bigram", "Trigram"])
ax.set_ylabel("Mean Perplexity (lower = better)")
ax.set_title("N-gram Order vs Perplexity (Add-k smoothing)")
ax.legend(title="Smoothing k")
ax.set_yscale("log")
fig.tight_layout()
plt.close(fig)

print("Figure 5: OOV rate vs vocabulary threshold…")

thresholds = [1, 2, 3, 5, 7, 10, 15, 20]
oov_rates, vocab_sizes = [], []

for thresh in thresholds:
    tr, te, voc = preprocess_data(train_raw, test_raw, thresh)
    total = sum(len(s) for s in te)
    oov   = sum(w == "<unk>" for s in te for w in s)
    oov_rates.append(100 * oov / max(total, 1))
    vocab_sizes.append(len(voc))

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

ax1.plot(thresholds, oov_rates,  color="#e74c3c", marker="o", linewidth=2, label="OOV rate (%)")
ax2.plot(thresholds, vocab_sizes, color="#2980b9", marker="s", linewidth=2,
         linestyle="--", label="Vocabulary size")

ax1.set_xlabel("Minimum word frequency threshold")
ax1.set_ylabel("OOV rate (%)", color="#e74c3c")
ax2.set_ylabel("Vocabulary size", color="#2980b9")
ax1.set_title("OOV Rate vs Vocabulary Threshold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
fig.tight_layout()
plt.close(fig)


print("Figure 6: Summary dashboard…")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("N-gram Language Model — Results Dashboard", fontsize=16, fontweight="bold", y=0.98)
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax_a = fig.add_subplot(gs[0, 0])
ax_a.bar(["Add-k", "Interp.", "Backoff", "KN"],
         means, color=bar_colors, alpha=0.85, edgecolor="white")
ax_a.set_yscale("log")
ax_a.set_title("A. Smoothing Comparison")
ax_a.set_ylabel("Perplexity (log)")
for i, m in enumerate(means):
    ax_a.text(i, m * 1.05, f"{m:.0f}", ha="center", fontsize=8, fontweight="bold")

ax_b = fig.add_subplot(gs[0, 1])
trig_pps = []
for k in k_values:
    mean_pp, _ = calculate_perplexity_batch(test_data, ngrams[1], ngrams[2], vocab_size, k=k)
    trig_pps.append(mean_pp)
ax_b.plot(k_values, trig_pps, color=COLORS["trigram"], marker="o", linewidth=2)
ax_b.set_xlabel("k")
ax_b.set_ylabel("Perplexity")
ax_b.set_title("B. Trigram Add-k Curve")
ax_b.set_yscale("log")


ax_c = fig.add_subplot(gs[0, 2])
order_pps = []
for n in orders:
    mean_pp, _ = calculate_perplexity_batch(test_data, ngrams[n-1], ngrams[n], vocab_size, k=0.5)
    order_pps.append(mean_pp)
ax_c.bar(["Unigram", "Bigram", "Trigram"], order_pps,
         color=[COLORS["unigram"], COLORS["bigram"], COLORS["trigram"]], alpha=0.85)
ax_c.set_ylabel("Perplexity")
ax_c.set_title("C. N-gram Order (k=0.5)")
ax_c.set_yscale("log")
for i, p in enumerate(order_pps):
    ax_c.text(i, p * 1.05, f"{p:.0f}", ha="center", fontsize=8, fontweight="bold")

ax_d = fig.add_subplot(gs[1, 0])
baseline = means[0]  # Add-k is index 0
improvements = [100 * (baseline - m) / baseline for m in means]
imp_colors = [COLORS["add_k"], COLORS["interpolation"],
              COLORS["stupid_backoff"], COLORS["kneser_ney"]]
bars_d = ax_d.bar(["Add-k", "Interp.", "Backoff", "KN"],
                  improvements, color=imp_colors, alpha=0.85, edgecolor="white")
ax_d.set_ylabel("Improvement over Add-k (%)")
ax_d.set_title("D. Perplexity Improvement")
ax_d.axhline(0, color="black", linewidth=0.8)
for bar, imp in zip(bars_d, improvements):
    ax_d.text(bar.get_x() + bar.get_width() / 2,
              bar.get_height() + 0.5 if imp >= 0 else bar.get_height() - 2,
              f"{imp:.1f}%", ha="center", fontsize=8, fontweight="bold")

ax_e = fig.add_subplot(gs[1, 1])
ax_e.plot(thresholds, oov_rates, color="#e74c3c", marker="o", linewidth=2)
ax_e.set_xlabel("Min frequency threshold")
ax_e.set_ylabel("OOV rate (%)")
ax_e.set_title("E. OOV Rate vs Threshold")


ax_f = fig.add_subplot(gs[1, 2])
ngram_sizes = [len(ng) for ng in ngrams]
ngram_labels = ["1-gram", "2-gram", "3-gram"]
ngram_colors = [COLORS["unigram"], COLORS["bigram"], COLORS["trigram"]]
bars_f = ax_f.bar(ngram_labels, ngram_sizes, color=ngram_colors, alpha=0.85)
ax_f.set_ylabel("Unique n-grams")
ax_f.set_title("F. N-gram Vocabulary Size")
for bar, size in zip(bars_f, ngram_sizes):
    ax_f.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
              f"{size:,}", ha="center", fontsize=8, fontweight="bold")

fig.savefig("plots/06_dashboard.png", bbox_inches="tight")
plt.close(fig)
print("  → plots/06_dashboard.png")
print("\n✅  Dashboard saved to plots/06_dashboard.png")
