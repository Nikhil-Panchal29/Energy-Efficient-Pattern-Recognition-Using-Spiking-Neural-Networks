import os
import cv2
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from mlp_model import build_mlp
from cnn_model import build_cnn
from fc_snn_model import build_snn
from conv_snn_model import build_conv_snn
import torchvision.transforms as transforms

# ---------- LOAD MODELS ----------
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

# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---------- EXACT test_image.py PREPROCESS ----------
def preprocess_image(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)
    digit = thresh[y:y+h, x:x+w]

    if h > w:
        pad = (h - w)//2
        digit = cv2.copyMakeBorder(digit, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    else:
        pad = (w - h)//2
        digit = cv2.copyMakeBorder(digit, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    digit = cv2.resize(digit, (20, 20))
    digit = cv2.copyMakeBorder(digit, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

    moments = cv2.moments(digit)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        shiftx = 14 - cx
        shifty = 14 - cy

        M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        digit = cv2.warpAffine(digit, M, (28, 28))

    digit = cv2.dilate(digit, kernel, iterations=2)
    digit = cv2.erode(digit, kernel, iterations=1)

    return digit

# ---------- DATASET ----------
dataset_path = "custom_dataset"  # Should contain subfolders 0,1,...,9 with images

def evaluate_model(model, name):

    y_true = []
    y_pred = []
    skipped = 0

    for label in sorted(os.listdir(dataset_path)):
        label_path = os.path.join(dataset_path, label)

        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            img_path = os.path.join(label_path, file)

            digit = preprocess_image(img_path)

            if digit is None:
                skipped += 1
                continue

            img_tensor = Image.fromarray(digit)
            img_tensor = transform(img_tensor).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                _, pred = torch.max(output, 1)

            y_true.append(int(label))
            y_pred.append(pred.item())

    print(f"\n=== {name} ===")
    print(f"Processed: {len(y_true)}, Skipped: {skipped}")

    if len(y_true) == 0:
        print("⚠️ No valid samples processed")
        return

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


# ---------- RUN ----------
print("\n===== TEST_IMAGE PIPELINE RESULTS =====")

evaluate_model(mlp, "MLP")
evaluate_model(cnn, "CNN")
evaluate_model(snn_poisson, "FC-SNN-Poisson")
evaluate_model(snn_latency, "FC-SNN-Latency")
evaluate_model(conv_snn_poisson, "Conv-SNN-Poisson")
evaluate_model(conv_snn_latency, "Conv-SNN-Latency")