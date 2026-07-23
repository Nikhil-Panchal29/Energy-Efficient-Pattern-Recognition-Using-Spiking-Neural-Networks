import torch
import pickle
from mlp_model import build_mlp, train_model as train_mlp
from cnn_model import build_cnn, train_model as train_cnn
from mlp_model import build_mlp
from cnn_model import build_cnn
from fc_snn_model import build_snn, train_model as train_snn
from conv_snn_model import build_conv_snn as train_conv_snn
from data_loader import train_loader, test_loader

# ---------- LOAD EXISTING MODELS ----------
#print("Loading trained MLP and CNN...")

#mlp = build_mlp()
#mlp.load_state_dict(torch.load("mlp.pth"))
#mlp.eval()

#cnn = build_cnn()
#cnn.load_state_dict(torch.load("cnn.pth"))
#cnn.eval()

# ---------- TRAIN ONLY SNN ----------

print("Training MLP...")
mlp = build_mlp()
mlp_metrics = train_mlp(mlp, train_loader, test_loader, epochs=10)
torch.save(mlp.state_dict(), "mlp.pth")

print("Training CNN...")
cnn = build_cnn()
cnn_metrics = train_cnn(cnn, train_loader, test_loader, epochs=10)
torch.save(cnn.state_dict(), "cnn.pth")

print("Training SNN (Poisson)...")
snn_poisson = build_snn(T=16, encoding="poisson")
metrics_poisson = train_snn(snn_poisson, train_loader, test_loader, epochs=20)
torch.save(snn_poisson.state_dict(), "fc_snn_poisson.pth")

print("Training SNN (Latency)...")
snn_latency = build_snn(T=16, encoding="latency")
metrics_latency = train_snn(snn_latency, train_loader, test_loader, epochs=20)
torch.save(snn_latency.state_dict(), "fc_snn_latency.pth")

print("Training Conv-SNN (Poisson)...")
conv_snn_poisson = train_conv_snn(T=16, encoding="poisson")
metrics_conv_poisson = train_snn(conv_snn_poisson, train_loader, test_loader, epochs=20)
torch.save(conv_snn_poisson.state_dict(), "conv_snn_poisson.pth")

print("Training Conv-SNN (Latency)...")
conv_snn_latency = train_conv_snn(T=16, encoding="latency")
metrics_conv_latency = train_snn(conv_snn_latency, train_loader, test_loader, epochs=20)
torch.save(conv_snn_latency.state_dict(), "conv_snn_latency.pth")

# ---------- SAVE METRICS ----------
metrics = {
    "MLP": {"status": "pretrained"},
    "CNN": {"status": "pretrained"},
    "FC_SNN_Poisson": metrics_poisson,
    "FC_SNN_Latency": metrics_latency,
    "Conv_SNN_Poisson": metrics_conv_poisson,
    "Conv_SNN_Latency": metrics_conv_latency
}

with open("metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("\nSNN training complete. Metrics updated.")
