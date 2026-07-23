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

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---------- Load dataset ----------
dataset = ImageFolder("dataset_webcam", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# ---------- Load models ----------
models = {
    "MLP": build_mlp(),
    "CNN": build_cnn(),
    "FC-SNN-Poisson": build_snn(encoding="poisson"),
    "FC-SNN-Latency": build_snn(encoding="latency"),
    "Conv-SNN-Poisson": build_conv_snn(encoding="poisson"),
    "Conv-SNN-Latency": build_conv_snn(encoding="latency"),
}

# Load weights
models["MLP"].load_state_dict(torch.load("mlp.pth"))
models["CNN"].load_state_dict(torch.load("cnn.pth"))
models["FC-SNN-Poisson"].load_state_dict(torch.load("fc_snn_poisson.pth"))
models["FC-SNN-Latency"].load_state_dict(torch.load("fc_snn_latency.pth"))
models["Conv-SNN-Poisson"].load_state_dict(torch.load("conv_snn_poisson.pth"))
models["Conv-SNN-Latency"].load_state_dict(torch.load("conv_snn_latency.pth"))

for model in models.values():
    model.eval()

# ---------- Evaluation ----------
def evaluate(model):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = (y_true == y_pred).mean() * 100

    return acc, y_true, y_pred

# ---------- Run ----------
for name, model in models.items():
    print(f"\n===== {name} =====")

    acc, y_true, y_pred = evaluate(model)

    print(f"Accuracy: {acc:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))