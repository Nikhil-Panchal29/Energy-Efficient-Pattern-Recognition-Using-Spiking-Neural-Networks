# =========================================
# SNN PAPER FIGURE GENERATION SCRIPT
# =========================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------
# STYLE (GLOBAL — matches your theme)
# -----------------------------------------
plt.style.use("dark_background")

TITLE_SIZE = 16
LABEL_SIZE = 12

def beautify(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.15)

# create output folder
os.makedirs("figures", exist_ok=True)

# -----------------------------------------
# DATA (FROM YOUR ACTUAL RESULTS)
# -----------------------------------------

models = ["MLP", "CNN", "SNN T=4", "SNN T=8", "SNN T=16", "SNN T=32", "Latency T=8", "Latency T=16"]

energy = [
    2.46e-6,
    1.95e-5,
    3e-8,
    5e-8,
    1e-7,
    2e-7,
    1.1e-8,
    2.5e-8
]

accuracy = [93, 97, 90, 92, 93, 94, 89, 89]

raw_acc = [10,13,11,11,8,9]
processed_acc = [93,97,92,89,96,89]

T = [4, 8, 16, 32]
acc_webcam = [82, 87, 88, 88]
energy_nj = [30, 50, 100, 200]

# -----------------------------------------
# FIG 1 — ENERGY BAR CHART (LOG)
# -----------------------------------------
fig, ax = plt.subplots(figsize=(10,5))

colors = ["#ff7043", "#42a5f5"] + ["#7e57c2"]*6

ax.bar(models, energy, color=colors)
ax.set_yscale("log")

ax.set_ylabel("Energy (J)", fontsize=LABEL_SIZE)
ax.set_title("Energy Model Comparison (Log Scale)", fontsize=TITLE_SIZE)

plt.xticks(rotation=20)
beautify(ax)

plt.savefig("figures/fig_energy_bar.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------
# FIG 2 — ACCURACY vs ENERGY (PARETO)
# -----------------------------------------
fig, ax = plt.subplots(figsize=(8,6))

for i, model in enumerate(models):
    ax.scatter(energy[i], accuracy[i], s=120)
    ax.text(energy[i], accuracy[i], model, fontsize=9)

ax.set_xscale("log")
ax.set_xlabel("Energy (J)", fontsize=LABEL_SIZE)
ax.set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE)
ax.set_title("Accuracy–Energy Trade-off (Pareto Curve)", fontsize=TITLE_SIZE)

beautify(ax)

plt.savefig("figures/fig_pareto.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------
# FIG 3 — TIMESTEP SWEEP
# -----------------------------------------
fig, ax1 = plt.subplots(figsize=(9,5))

ax1.plot(T, acc_webcam, marker='o', linewidth=2)
ax1.set_xlabel("Timesteps (T)")
ax1.set_ylabel("Accuracy (%)")

ax2 = ax1.twinx()
ax2.plot(T, energy_nj, linestyle="--", marker='o', linewidth=2)
ax2.set_ylabel("Energy (nJ)")

ax1.set_title("Timestep Sweep: Accuracy vs Energy")

beautify(ax1)

plt.savefig("figures/fig_timestep.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------
# FIG 4 — PREPROCESSING IMPACT
# -----------------------------------------
gap = [processed_acc[i] - raw_acc[i] for i in range(len(raw_acc))]

fig, ax = plt.subplots(figsize=(9,5))

ax.bar(models[:6], gap, color="#66bb6a")

ax.set_ylabel("Accuracy Gain (%)")
ax.set_title("Impact of Webcam Preprocessing")

plt.xticks(rotation=20)
beautify(ax)

plt.savefig("figures/fig_preprocessing_gap.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------
# FIG 5 — ENCODING COMPARISON
# -----------------------------------------
labels = ["FC-SNN", "Conv-SNN"]
poisson = [92, 96]
latency = [89, 89]

x = np.arange(len(labels))

fig, ax = plt.subplots()

ax.bar(x - 0.2, poisson, width=0.4, label="Poisson", color="#7e57c2")
ax.bar(x + 0.2, latency, width=0.4, label="Latency", color="#b39ddb")

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylabel("Accuracy (%)")
ax.set_title("Spike Encoding Comparison")

ax.legend()
beautify(ax)

plt.savefig("figures/fig_encoding.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------
# FIG 6 — MODEL PERFORMANCE
# -----------------------------------------
fig, ax = plt.subplots()

ax.bar(models[:6], processed_acc, color="#42a5f5")

ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Performance (Processed Webcam Data)")

plt.xticks(rotation=20)
beautify(ax)

plt.savefig("figures/fig_model_performance.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------
# FIG 7 — CONFUSION MATRICES (REAL RESULTS)
# -----------------------------------------

# --- CNN (BEST MODEL) ---
cnn_matrix = np.array([
    [10,0,0,0,0,0,0,0,0,0],
    [0,10,0,0,0,0,0,0,0,0],
    [0,0,9,0,0,0,1,0,0,0],
    [0,0,0,10,0,0,0,0,0,0],
    [0,0,0,0,9,0,1,0,0,0],
    [0,0,0,0,0,10,0,0,0,0],
    [0,0,0,0,0,0,10,0,0,0],
    [0,0,0,0,0,0,0,10,0,0],
    [1,0,0,0,0,0,0,0,9,0],
    [0,0,0,0,0,0,0,0,0,10]
])

# --- Conv-SNN (YOUR MAIN RESULT) ---
snn_matrix = np.array([
    [10,0,0,0,0,0,0,0,0,0],
    [0,10,0,0,0,0,0,0,0,0],
    [0,0,9,0,0,0,0,0,1,0],
    [0,0,0,10,0,0,0,0,0,0],
    [0,0,0,0,9,0,0,0,0,1],
    [0,0,0,0,0,9,0,0,0,1],
    [0,0,0,0,0,0,10,0,0,0],
    [0,0,0,0,0,0,0,10,0,0],
    [0,0,1,0,0,0,0,0,9,0],
    [0,0,0,0,0,0,0,0,0,10]
])

# --- RAW MLP (FAILURE CASE — IMPORTANT STORY) ---
raw_matrix = np.array([
    [10,0,0,0,0,0,0,0,0,0],
    [10,0,0,0,0,0,0,0,0,0],
    [10,0,0,0,0,0,0,0,0,0],
    [10,0,0,0,0,0,0,0,0,0],
    [10,0,0,0,0,0,0,0,0,0],
    [10,0,0,0,0,0,0,0,0,0],
    [10,0,0,0,0,0,0,0,0,0],
    [10,0,0,0,0,0,0,0,0,0],
    [10,0,0,0,0,0,0,0,0,0],
    [10,0,0,0,0,0,0,0,0,0]
])

# Dictionary for iteration
matrices = {
    "CNN (Processed)": cnn_matrix,
    "Conv-SNN (Processed)": snn_matrix,
    "MLP (Raw Input Failure)": raw_matrix
}

# Plot all
for name, matrix in matrices.items():
    plt.figure(figsize=(6,5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="magma", cbar=False)
    
    plt.title(f"Confusion Matrix — {name}", fontsize=TITLE_SIZE)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    plt.savefig(f"figures/conf_matrix_{name.replace(' ','_')}.png",
                dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------
# DONE
# -----------------------------------------
print("\n✅ ALL FIGURES GENERATED IN /figures FOLDER\n")