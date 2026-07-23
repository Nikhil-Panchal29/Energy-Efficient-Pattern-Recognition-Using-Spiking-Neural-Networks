import cv2
import torch
import torchvision.transforms as transforms
from mlp_model import build_mlp
from cnn_model import build_cnn
from fc_snn_model import build_snn
from conv_snn_model import build_conv_snn
from fvcore.nn import FlopCountAnalysis
from tabulate import tabulate
from PIL import Image
import numpy as np
import time
import os
from collections import deque

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
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---------- Prediction smoothing buffers ----------
mlp_buffer = deque(maxlen=10)
cnn_buffer = deque(maxlen=10)
fc_snn_poisson_buffer = deque(maxlen=10)
fc_snn_latency_buffer = deque(maxlen=10)
conv_snn_poisson_buffer = deque(maxlen=10)
conv_snn_latency_buffer = deque(maxlen=10)

def smooth_prediction(buffer, new_pred):
    buffer.append(new_pred)
    return max(set(buffer), key=buffer.count)

cap = cv2.VideoCapture(0)
snapshot_interval = 3
last_snapshot_time = time.time()

os.makedirs("snapshots", exist_ok=True)
os.makedirs("dataset_webcam", exist_ok=True)

print("\nPress 'q' to quit.")

# ---------- Improved preprocessing ----------
def preprocess(gray):

    # Step 1: blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 2: threshold (VERY IMPORTANT)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 3: remove noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Step 4: find digit contour
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)

    if cv2.contourArea(c) < 200:
        return None

    x, y, w, h = cv2.boundingRect(c)
    digit = thresh[y:y+h, x:x+w]

    # Step 5: make square
    if h > w:
        pad = (h - w) // 2
        digit = cv2.copyMakeBorder(digit, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    else:
        pad = (w - h) // 2
        digit = cv2.copyMakeBorder(digit, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # Step 6: resize to 20x20
    digit = cv2.resize(digit, (20, 20))

    # Step 7: pad to 28x28
    digit = cv2.copyMakeBorder(digit, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

    # Step 8: CENTER using center of mass
    moments = cv2.moments(digit)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        shiftx = 14 - cx
        shifty = 14 - cy

        M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        digit = cv2.warpAffine(digit, M, (28, 28))

    # Step 9: FIX THICKNESS (CRUCIAL FOR SNN)
    digit = cv2.dilate(digit, kernel, iterations=2)
    digit = cv2.erode(digit, kernel, iterations=1)

    return digit
# ---------- Main loop ----------
while True:

    ret,frame=cap.read() 
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# -------- ROI BOX --------
    h, w = gray.shape

    x1, y1 = w//4, h//4
    x2, y2 = 3*w//4, 3*h//4

    # Draw rectangle for user
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop ROI
    roi = gray[y1:y2, x1:x2]

    # Pass ONLY ROI to model
    digit = preprocess(roi)

    current_time=time.time()

    if digit is not None and current_time-last_snapshot_time>=snapshot_interval:
        snapshot_name = f"snapshots/snapshot_{int(current_time)}.png"
        cv2.imwrite(snapshot_name, digit)
        print(f"Saved processed image: {snapshot_name}")

        # Save RAW webcam ROI (for dataset creation)
        dataset_name = f"dataset_webcam/raw_{int(current_time)}.png"
        cv2.imwrite(dataset_name, roi)
        print(f"Saved raw webcam image: {dataset_name}")

        last_snapshot_time=current_time

        img=Image.fromarray(digit)
        img=transform(img).unsqueeze(0)

        results=[]

        # ---------- MLP ----------
        start=time.time()
        with torch.no_grad():
            output=mlp(img)
            probs=torch.softmax(output,dim=1)
            conf,pred=torch.max(probs,1)
            pred=smooth_prediction(mlp_buffer,pred.item())
        end=time.time()

        flops=FlopCountAnalysis(mlp,img).total()
        energy=flops*MAC_ENERGY

        results.append(["MLP",pred,f"{conf.item()*100:.2f}%",f"{flops/1e6:.2f}M",f"{(end-start)*1000:.2f} ms",f"{energy:.2e} J"])

        # ---------- CNN ----------
        start=time.time()
        with torch.no_grad():
            output=cnn(img)
            probs=torch.softmax(output,dim=1)
            conf,pred=torch.max(probs,1)
            pred=smooth_prediction(cnn_buffer,pred.item())
        end=time.time()

        flops=FlopCountAnalysis(cnn,img).total()
        energy=flops*MAC_ENERGY

        results.append(["CNN",pred,f"{conf.item()*100:.2f}%",f"{flops/1e6:.2f}M",f"{(end-start)*1000:.2f} ms",f"{energy:.2e} J"])

        # ---------- FC-SNN (Poisson) ----------
        start = time.time()
        with torch.no_grad():
            output, stats = snn_poisson(img)

            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            pred = smooth_prediction(fc_snn_poisson_buffer, pred.item())
        end = time.time()

        energy = stats["energy_joules"]

        results.append([
            "FC-SNN-Poisson",
            pred,
            f"{conf.item()*100:.2f}%",
            "Spike-based",
            f"{(end-start)*1000:.2f} ms",
            f"{energy:.2e} J"
        ])

        # ---------- FC-SNN (Latency) ----------
        start = time.time()
        with torch.no_grad():
            output, stats = snn_latency(img)

            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            pred = smooth_prediction(fc_snn_latency_buffer, pred.item())
        end = time.time()

        energy = stats["energy_joules"]

        results.append([
            "FC-SNN-Latency",
            pred,
            f"{conf.item()*100:.2f}%",
            "Spike-based",
            f"{(end-start)*1000:.2f} ms",
            f"{energy:.2e} J"
        ])
        # ---------- Conv-SNN (Poisson) ----------
        start = time.time()
        with torch.no_grad():
            output, stats = conv_snn_poisson(img)

            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            pred = smooth_prediction(conv_snn_poisson_buffer, pred.item())
        end = time.time()

        energy = stats["energy_joules"]

        results.append([
            "Conv-SNN-Poisson",
            pred,
            f"{conf.item()*100:.2f}%",
            "Spike-based",
            f"{(end-start)*1000:.2f} ms",
            f"{energy:.2e} J"
        ])
        # ---------- Conv-SNN (Latency) ----------
        start = time.time()
        with torch.no_grad():
            output, stats = conv_snn_latency(img)

            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            pred = smooth_prediction(conv_snn_latency_buffer, pred.item())
        end = time.time()

        energy = stats["energy_joules"]

        results.append([
            "Conv-SNN-Latency",
            pred,
            f"{conf.item()*100:.2f}%",
            "Spike-based",
            f"{(end-start)*1000:.2f} ms",
            f"{energy:.2e} J"
        ])

        print("\n--- Snapshot Prediction ---")
        print(tabulate(results,headers=["Model","Prediction","Confidence","Operations","Time","Energy"]))

        # Show raw webcam
        cv2.imshow("Webcam (ROI Box)", frame)
        cv2.imshow("ROI (Input)", roi)

        # Show processed MNIST-like digit
        cv2.imshow("Processed Digit (28x28)", digit)

        big_digit = cv2.resize(digit, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Processed Digit (Zoomed)", big_digit)

        if cv2.waitKey(1)&0xFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
