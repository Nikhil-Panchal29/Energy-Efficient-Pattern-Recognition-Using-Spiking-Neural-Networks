import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from mlp_model import build_mlp
from cnn_model import build_cnn
from fc_snn_model import build_snn
from conv_snn_model import build_conv_snn

# ---------- DEVICE ----------
device = torch.device("cpu")

# ---------- TRANSFORM (same as training) ----------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---------- LOAD DATASET ----------
dataset = ImageFolder("custom_dataset", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# ---------- LOAD MODELS ----------
models = {
    "MLP": build_mlp(),
    "CNN": build_cnn(),
    "FC-SNN-Poisson": build_snn(encoding="poisson"),
    "FC-SNN-Latency": build_snn(encoding="latency"),
    "Conv-SNN-Poisson": build_conv_snn(encoding="poisson"),
    "Conv-SNN-Latency": build_conv_snn(encoding="latency"),
}

# Load weights
models["MLP"].load_state_dict(torch.load("mlp.pth", map_location=device))
models["CNN"].load_state_dict(torch.load("cnn.pth", map_location=device))
models["FC-SNN-Poisson"].load_state_dict(torch.load("fc_snn_poisson.pth", map_location=device))
models["FC-SNN-Latency"].load_state_dict(torch.load("fc_snn_latency.pth", map_location=device))
models["Conv-SNN-Poisson"].load_state_dict(torch.load("conv_snn_poisson.pth", map_location=device))
models["Conv-SNN-Latency"].load_state_dict(torch.load("conv_snn_latency.pth", map_location=device))

for model in models.values():
    model.to(device)
    model.eval()

# ---------- EVALUATION FUNCTION ----------
def evaluate(model, loader):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            # Handle SNN tuple output
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = (y_true == y_pred).mean() * 100

    return accuracy, y_true, y_pred

# ---------- RUN ----------
print("\n===== WEBCAM DATASET RESULTS =====")

for name, model in models.items():
    print(f"\n=== {name} ===")

    acc, y_true, y_pred = evaluate(model, loader)

    print(f"Accuracy: {acc:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))