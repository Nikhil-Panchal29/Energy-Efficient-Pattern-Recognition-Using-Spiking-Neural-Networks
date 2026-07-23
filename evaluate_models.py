import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from mlp_model import build_mlp
from cnn_model import build_cnn
from fc_snn_model import build_snn
from conv_snn_model import build_conv_snn

# ---------- LOAD MODELS ----------
print("Loading trained models...")

mlp = build_mlp()
mlp.load_state_dict(torch.load("mlp.pth"))
mlp.eval()

cnn = build_cnn()
cnn.load_state_dict(torch.load("cnn.pth"))
cnn.eval()

snn_poisson = build_snn(encoding="poisson")
snn_poisson.load_state_dict(torch.load("fc_snn_poisson.pth"))
snn_poisson.eval()

snn_latency = build_snn(encoding="latency")
snn_latency.load_state_dict(torch.load("fc_snn_latency.pth"))
snn_latency.eval()

conv_snn_poisson = build_conv_snn(encoding="poisson")
conv_snn_poisson.load_state_dict(torch.load("conv_snn_poisson.pth"))
conv_snn_poisson.eval()

conv_snn_latency = build_conv_snn(encoding="latency")
conv_snn_latency.load_state_dict(torch.load("conv_snn_latency.pth"))
conv_snn_latency.eval()

# ---------- TRANSFORMS ----------

# Clean MNIST
clean_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Augmented MNIST
aug_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, translate=(0.2,0.2)),
    transforms.RandomPerspective(distortion_scale=0.3, p=1.0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---------- DATASETS ----------
clean_test = MNIST(root='./data', train=False, download=True, transform=clean_transform)
aug_test = MNIST(root='./data', train=False, download=True, transform=aug_transform)

clean_loader = DataLoader(clean_test, batch_size=64, shuffle=False)
aug_loader = DataLoader(aug_test, batch_size=64, shuffle=False)

# ---------- EVALUATION FUNCTION ----------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)

            # SNN returns tuple
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# ---------- RUN EVALUATION ----------

models = {
    "MLP": mlp,
    "CNN": cnn,
    "FC-SNN (Poisson)": snn_poisson,
    "FC-SNN (Latency)": snn_latency,
    "Conv-SNN (Poisson)": conv_snn_poisson,
    "Conv-SNN (Latency)": conv_snn_latency
}

print("\n========== RESULTS ==========\n")

for name, model in models.items():
    clean_acc = evaluate(model, clean_loader)
    aug_acc = evaluate(model, aug_loader)

    print(f"{name}")
    print(f"  Clean MNIST Accuracy: {clean_acc:.2f}%")
    print(f"  Augmented MNIST Accuracy: {aug_acc:.2f}%\n")