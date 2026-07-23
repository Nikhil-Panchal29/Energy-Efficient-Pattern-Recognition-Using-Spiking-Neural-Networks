import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

ACC_ENERGY = 0.9e-12
spike_grad = surrogate.fast_sigmoid(slope=25)


class ConvSNN(nn.Module):
    def __init__(self, T=16, encoding="poisson"):
        super().__init__()

        self.T = T
        self.encoding = encoding

        # ---------- Conv Layers ----------
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        # compute flatten size
        self._to_linear = None
        self._get_conv_output()

        # ---------- FC Layers ----------
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.fc2 = nn.Linear(128, 10)
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def _get_conv_output(self):
        x = torch.zeros(1, 1, 28, 28)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        self._to_linear = x.numel()

    # ---------- Encoding ----------
    def poisson_encoding(self, x):
        return torch.bernoulli(x)

    def latency_encoding(self, x):
        B, C, H, W = x.shape
        x = x.view(B, -1)

        spikes = torch.zeros(self.T, B, x.shape[1], device=x.device)
        spike_times = (1 - x) * (self.T - 1)
        spike_times = spike_times.long()

        for t in range(self.T):
            spikes[t] = (spike_times == t).float()

        return spikes.view(self.T, B, C, H, W)

    # ---------- Forward ----------
    def forward(self, x):

        # normalize
        x = torch.clamp(x, 0, 1)

        # encoding
        if self.encoding == "poisson":
            spike_seq = torch.stack([self.poisson_encoding(x) for _ in range(self.T)])
        elif self.encoding == "latency":
            spike_seq = self.latency_encoding(x)
        else:
            raise ValueError("Unknown encoding")

        # init mem
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        total_spikes = 0
        spk_out = []

        for t in range(self.T):

            cur = spike_seq[t]

            # conv block
            cur = self.conv1(cur)
            spk1, mem1 = self.lif1(cur, mem1)

            cur = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur, mem2)

            # flatten
            cur = torch.flatten(spk2, 1)

            # fc block
            cur = self.fc1(cur)
            spk3, mem3 = self.lif3(cur, mem3)

            cur = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur, mem4)

            total_spikes += spk1.sum() + spk2.sum() + spk3.sum() + spk4.sum()

            spk_out.append(spk4)

        spk_out = torch.stack(spk_out)
        out = spk_out.mean(0)

        energy = total_spikes * ACC_ENERGY

        return out, {
            "total_spikes": total_spikes.item(),
            "energy_joules": energy.item()
        }


def build_conv_snn(T=16, encoding="poisson"):
    return ConvSNN(T=T, encoding=encoding)