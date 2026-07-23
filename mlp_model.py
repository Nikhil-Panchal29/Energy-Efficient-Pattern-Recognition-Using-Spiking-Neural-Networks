import torch
import torch.nn as nn
import torch.optim as optim
from fvcore.nn import FlopCountAnalysis
import time

MAC_ENERGY = 4.6e-12  # Joules

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

def build_mlp():
    return MLP()

def train_model(model, train_loader, test_loader, epochs=15, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"[MLP] Epoch {epoch+1} Accuracy: {100*correct/total:.2f}%")

    # Test
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_acc = 100 * correct / total

    dummy_input = torch.randn(1,1,28,28)
    macs = FlopCountAnalysis(model, dummy_input).total()
    energy = macs * MAC_ENERGY

    return {
        "accuracy": final_acc,
        "macs": macs,
        "energy_joules": energy
    }
