import torch
import torchvision.transforms as transforms
from PIL import Image
from tabulate import tabulate
import time
import cv2
import numpy as np

from mlp_model import build_mlp
from cnn_model import build_cnn
from fc_snn_model import build_snn
from conv_snn_model import build_conv_snn
from fvcore.nn import FlopCountAnalysis

MAC_ENERGY = 4.6e-12
ACC_ENERGY = 0.9e-12

# ---------- Load models ----------
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

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---------- PREPROCESS (same as webcam) ----------
def preprocess_image(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Blur
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Otsu threshold
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contour
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)
    digit = thresh[y:y+h, x:x+w]

    # Make square
    if h > w:
        pad = (h - w)//2
        digit = cv2.copyMakeBorder(digit, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    else:
        pad = (w - h)//2
        digit = cv2.copyMakeBorder(digit, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # Resize to MNIST size
    digit = cv2.resize(digit, (20, 20))
    digit = cv2.copyMakeBorder(digit, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

    # Center of mass shift
    moments = cv2.moments(digit)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        shiftx = 14 - cx
        shifty = 14 - cy

        M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        digit = cv2.warpAffine(digit, M, (28, 28))

    # Thickness adjustment (important for SNN)
    digit = cv2.dilate(digit, kernel, iterations=2)
    digit = cv2.erode(digit, kernel, iterations=1)

    return digit

# ---------- Load image ----------
img_path = input("Enter image path: ")

digit = preprocess_image(img_path)

if digit is None:
    print("No digit detected.")
    exit()

# Show processed image (VERY USEFUL DEBUG)
cv2.imshow("Processed Image", digit)
cv2.waitKey(4000)
cv2.destroyAllWindows()

img = Image.fromarray(digit)
img = transform(img).unsqueeze(0)

results = []

# ---------- MLP ----------
print("starting MLP inference...")
start = time.time()
with torch.no_grad():
    output = mlp(img)
    probs = torch.softmax(output, dim=1)
    conf, pred = torch.max(probs, 1)
end = time.time()
print("MLP inference done.")
print("starting FLOP count...")

flops = FlopCountAnalysis(mlp, img).total()
energy = flops * MAC_ENERGY
print("FLOP count done.")

results.append(["MLP", pred.item(), f"{conf.item()*100:.2f}%", f"{flops/1e6:.2f}M", f"{(end-start)*1000:.2f} ms", f"{energy:.2e} J"])

# ---------- CNN ----------
print("starting CNN inference...")
start = time.time()
with torch.no_grad():
    output = cnn(img)
    probs = torch.softmax(output, dim=1)
    conf, pred = torch.max(probs, 1)
end = time.time()
print("CNN inference done.")
print("starting FLOP count...")
flops = FlopCountAnalysis(cnn, img).total()
energy = flops * MAC_ENERGY
print("FLOP count done.")

results.append(["CNN", pred.item(), f"{conf.item()*100:.2f}%", f"{flops/1e6:.2f}M", f"{(end-start)*1000:.2f} ms", f"{energy:.2e} J"])

# ---------- SNN + Conv-SNN ----------
print("starting SNN inference...")
def run_snn(model, name):
    start = time.time()
    with torch.no_grad():
        output, stats = model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    end = time.time()
    print(f"{name} inference done.")
    results.append([
        name,
        pred.item(),
        f"{conf.item()*100:.2f}%",
        "Spike-based",
        f"{(end-start)*1000:.2f} ms",
        f"{stats['energy_joules']:.2e} J"
    ])

run_snn(snn_poisson, "FC-SNN-Poisson")
run_snn(snn_latency, "FC-SNN-Latency")
run_snn(conv_snn_poisson, "Conv-SNN-Poisson")
run_snn(conv_snn_latency, "Conv-SNN-Latency")

# ---------- Print ----------
print("\n--- Single Image Prediction ---")
print(tabulate(results, headers=["Model","Prediction","Confidence","Operations","Time","Energy"]))