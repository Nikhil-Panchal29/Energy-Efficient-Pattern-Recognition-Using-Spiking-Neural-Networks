import torch
from fc_snn_model import build_snn
from data_loader import test_loader
from tabulate import tabulate

T_values = [4, 8, 16, 32]


def evaluate(model, loader):
    correct = 0
    total = 0
    total_energy = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs, stats = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            total_energy += stats["energy_joules"]

    accuracy = 100 * correct / total
    avg_energy = total_energy / total  # ✅ important

    return accuracy, avg_energy


# ================================
# 🔵 POISSON SWEEP
# ================================
poisson_results = []

for T in T_values:
    print(f"\n[Poisson] Running for T = {T}")

    model = build_snn(T=T, encoding="poisson")
    model.load_state_dict(torch.load("fc_snn_poisson.pth"))
    model.eval()

    acc, energy = evaluate(model, test_loader)

    poisson_results.append([T, acc, energy])

    print(f"T={T} | Accuracy={acc:.2f}% | Energy={energy:.2e}")


print("\n===== Poisson Timestep Sweep =====")
print(tabulate(poisson_results, headers=["T", "Accuracy (%)", "Energy/sample (J)"]))


# ================================
# 🔴 LATENCY SWEEP
# ================================
latency_results = []

for T in T_values:
    print(f"\n[Latency] Running for T = {T}")

    model = build_snn(T=T, encoding="latency")
    model.load_state_dict(torch.load("fc_snn_latency.pth"))
    model.eval()

    acc, energy = evaluate(model, test_loader)

    latency_results.append([T, acc, energy])

    print(f"T={T} | Accuracy={acc:.2f}% | Energy={energy:.2e}")


print("\n===== Latency Timestep Sweep =====")
print(tabulate(latency_results, headers=["T", "Accuracy (%)", "Energy/sample (J)"]))