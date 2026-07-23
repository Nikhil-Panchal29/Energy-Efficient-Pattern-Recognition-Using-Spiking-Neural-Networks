import matplotlib.pyplot as plt
import numpy as np

models = [
    "MLP",
    "CNN",
    "FC-SNN-P",
    "FC-SNN-L",
    "Conv-SNN-P",
    "Conv-SNN-L"
]

# =========================
# EXACT VALUES
# =========================

mnist_acc = [97.11, 98.96, 92.29, 92.47, 98.15, 95.69]
aug_acc   = [93.24, 97.71, 84.89, 80.90, 95.59, 92.10]
webcam_acc = [93, 97, 92, 89, 96, 89]

# Energy (approx but consistent with your logs)
energy = [
    2.46e-6,    # MLP
    1.95e-5,    # CNN
    8.10e-10,   # FC-SNN-P
    8.00e-10,   # FC-SNN-L
    1.43e-8,    # Conv-SNN-P
    1.61e-8     # Conv-SNN-L
]

# =========================
# ACCURACY COMPARISON
# =========================
x = np.arange(len(models))
width = 0.25

plt.figure()
plt.bar(x - width, mnist_acc, width, label='MNIST')
plt.bar(x, aug_acc, width, label='Augmented')
plt.bar(x + width, webcam_acc, width, label='Webcam')

plt.xticks(x, models, rotation=20)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Across Datasets")
plt.legend()

plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=300)
plt.show()

# =========================
# ENERGY vs ACCURACY
# =========================
plt.figure()

for i in range(len(models)):
    plt.scatter(energy[i], webcam_acc[i])
    plt.text(energy[i], webcam_acc[i], models[i])

plt.xscale("log")
plt.xlabel("Energy (J)")
plt.ylabel("Accuracy (%)")
plt.title("Energy vs Accuracy Trade-off")

plt.tight_layout()
plt.savefig("energy_vs_accuracy.png", dpi=300)
plt.show()