import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# -------- Energy constants --------
ACC_ENERGY = 0.9e-12  # energy per spike op

# -------- Surrogate gradient --------
spike_grad = surrogate.fast_sigmoid(slope=25)


class FCSNN(nn.Module): 
    def __init__(self, T=16, encoding="poisson"):
        super().__init__()

        self.T = T
        self.encoding = encoding

        # -------- Layers --------
        self.fc1 = nn.Linear(28*28, 256)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.fc2 = nn.Linear(256, 10)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    # =========================
    # 🔥 ENCODING FUNCTIONS
    # =========================

    def poisson_encoding(self, x):
        # x in [0,1]
        return torch.bernoulli(x)

    def latency_encoding(self, x):
        # Convert intensity → spike time
        # earlier spike for higher intensity
        # create spike only once
        B, D = x.shape
        spikes = torch.zeros(self.T, B, D, device=x.device)

        spike_times = (1 - x) * (self.T - 1)
        spike_times = spike_times.long()

        for t in range(self.T):
            spikes[t] = (spike_times == t).float()

        return spikes

    # =========================
    # 🔥 FORWARD PASS
    # =========================

    def forward(self, x):

        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # flatten

        # normalize to [0,1] for encoding
        x = torch.clamp(x, 0, 1)

        # -------- Encoding --------
        if self.encoding == "poisson":
            spike_seq = torch.stack([self.poisson_encoding(x) for _ in range(self.T)])
        elif self.encoding == "latency":
            spike_seq = self.latency_encoding(x)
        else:
            raise ValueError("Unknown encoding type")

        # -------- Initialize membrane states --------
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # -------- Logging --------
        total_spikes_l1 = 0
        total_spikes_l2 = 0
        spk_out = []

        # -------- Time loop --------
        for t in range(self.T):

            input_spikes = spike_seq[t]

            cur1 = self.fc1(input_spikes)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # log spikes
            total_spikes_l1 += spk1.sum()
            total_spikes_l2 += spk2.sum()

            spk_out.append(spk2)

        # -------- Decode output --------
        spk_out = torch.stack(spk_out)      # [T, B, 10]
        out = spk_out.mean(0)               # rate decoding

        # -------- Energy --------
        total_spikes = total_spikes_l1 + total_spikes_l2
        energy = total_spikes * ACC_ENERGY

        return out, {
            "spikes_l1": total_spikes_l1.item(),
            "spikes_l2": total_spikes_l2.item(),
            "total_spikes": total_spikes.item(),
            "energy_joules": energy.item()
        }
def build_snn(T=16, encoding="poisson"):
    return FCSNN(T=T, encoding=encoding)

import torch.optim as optim
import time

def train_model(model, train_loader, test_loader, epochs=20, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()

    # -------- Training --------
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs, _ = model(images)  # <-- IMPORTANT CHANGE

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"[SNN-{model.encoding}] Epoch {epoch+1}/{epochs} | Accuracy: {acc:.2f}%")

    # -------- Testing --------
    model.eval()
    correct, total = 0, 0

    total_spikes = 0
    total_energy = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs, stats = model(images)

            total_spikes += stats["total_spikes"]
            total_energy += stats["energy_joules"]

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_acc = 100 * correct / total
    end_time = time.time()

    print(f"\n[SNN-{model.encoding}] Final Test Accuracy: {final_acc:.2f}%")

    return {
        "accuracy": final_acc,
        "total_spikes": total_spikes,
        "energy_joules": total_energy,
        "training_time": end_time - start_time
    }