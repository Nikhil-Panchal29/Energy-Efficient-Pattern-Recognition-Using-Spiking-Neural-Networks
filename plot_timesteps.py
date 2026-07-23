import matplotlib.pyplot as plt

T = [4, 8, 16, 32]

# EXACT VALUES
poisson_acc = [91.15, 91.94, 92.15, 92.36]
poisson_energy = [2.18e-10, 4.43e-10, 8.93e-10, 1.79e-09]

latency_acc = [85.11, 90.29, 92.47, 86.72]
latency_energy = [3.47e-10, 5.79e-10, 7.66e-10, 9.55e-10]

# Accuracy vs T
plt.figure()
plt.plot(T, poisson_acc, marker='o', label='Poisson')
plt.plot(T, latency_acc, marker='o', label='Latency')

plt.xlabel("Timesteps (T)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Timesteps")
plt.legend()

plt.tight_layout()
plt.savefig("timestep_accuracy.png", dpi=300)
plt.show()

# Energy vs T
plt.figure()
plt.plot(T, poisson_energy, marker='o', label='Poisson')
plt.plot(T, latency_energy, marker='o', label='Latency')

plt.xlabel("Timesteps (T)")
plt.ylabel("Energy (J)")
plt.title("Energy vs Timesteps")
plt.legend()

plt.tight_layout()
plt.savefig("timestep_energy.png", dpi=300)
plt.show()