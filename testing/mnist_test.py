# single_image_infer.py
from __future__ import annotations
import argparse, time, statistics as stats
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

# ===== Quantum (PennyLane) =====
try:
    import pennylane as qml
except ImportError:
    raise SystemExit("PennyLane not found. Install with: pip install pennylane")

# ----------------- Quantum setup -----------------
n_qubits = 4
q_depth  = 6
device_q = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(device_q, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# ----------------- Models -----------------
class QNet(nn.Module):
    """Hybrid CNN -> FC(->4) -> Quantum(4) -> FC -> FC(10)"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1   = nn.Linear(16*6*6, n_qubits)
        self.q_layer = qml.qnn.TorchLayer(
            quantum_circuit, weight_shapes={"weights": (q_depth, n_qubits)}
        )
        self.fc2 = nn.Linear(n_qubits, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*6*6)
        x = torch.tanh(self.fc1(x))
        x = self.q_layer(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Net(nn.Module):
    """Purely classical LeNet-ish."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ----------------- Preprocessing -----------------
MNIST_NORM = ((0.1307,), (0.3081,))

def preprocess_single(path: str) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(*MNIST_NORM),
    ])
    img = Image.open(path).convert("L")
    return tf(img).unsqueeze(0)  # [1,1,32,32]

def build_test_loader(batch_size=128, limit: int | None = None, use_cuda=False) -> DataLoader:
    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(*MNIST_NORM),
    ])
    ds = datasets.MNIST("../data", train=False, download=True, transform=tf)
    if limit is not None:
        ds = torch.utils.data.Subset(ds, list(range(min(limit, len(ds)))))
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=0 if not use_cuda else 2, pin_memory=use_cuda
    )

# ----------------- Metrics helpers -----------------
@torch.inference_mode()
def run_once(model: nn.Module, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
    logits = model(x)
    pred = logits.argmax(dim=1).item()
    return pred, logits.squeeze()

@torch.inference_mode()
def time_model(model: nn.Module, x: torch.Tensor, runs: int = 30, warmup: int = 3):
    times = []
    for _ in range(warmup):
        _ = model(x)
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return {
        "mean_ms": sum(times)/len(times),
        "std_ms":  stats.pstdev(times),
        "min_ms":  min(times),
        "max_ms":  max(times),
        "runs":    runs,
    }

def fmt_timing(tag: str, t):
    return (f"{tag}: mean {t['mean_ms']:.2f} ms ± {t['std_ms']:.2f} "
            f"(min {t['min_ms']:.2f}, max {t['max_ms']:.2f}) over {t['runs']} runs")

@torch.inference_mode()
def benchmark(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total, correct = 0, 0
    t0 = time.perf_counter()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    t1 = time.perf_counter()
    acc = correct / total if total else 0.0
    ms_per_img = (t1 - t0) * 1000.0 / total if total else 0.0
    return acc, ms_per_img

@torch.inference_mode()
def evaluate_epochs(model: nn.Module, loader: DataLoader, device: torch.device, epochs: int):
    """Returns final accuracy and a list of epoch wall times (s)."""
    times = []
    final_acc = 0.0
    for _ in range(epochs):
        model.eval()
        correct = 0
        total = 0
        t0 = time.perf_counter()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
        t1 = time.perf_counter()
        final_acc = correct / total if total else 0.0
        times.append(t1 - t0)
    return final_acc, times

# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser()
    # Single-image
    parser.add_argument("--image", "-i", type=str, help="Path to one MNIST-like image (PNG)")
    parser.add_argument("--label", type=int, choices=range(10), help="Ground truth label for the single image")
    parser.add_argument("--runs", type=int, default=30, help="Latency repetitions for single-image timing")
    # Evaluation on full test set
    parser.add_argument("--eval", action="store_true", help="Evaluate on the full MNIST test set")
    parser.add_argument("--epochs", type=int, default=1, help="Number of eval epochs to time (with --eval)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation/benchmark")
    # Benchmark subset
    parser.add_argument("--benchmark", type=int, default=0, help="If >0, eval on first N test images and report ms/image")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_grad_enabled(False)

    # Build models
    classical = Net().to(device).eval()
    hybrid    = QNet().to(device).eval()

    # ===== Single image path =====
    if args.image:
        x = preprocess_single(args.image).to(device)
        cpred, clogits = run_once(classical, x)
        hpred, hlogits = run_once(hybrid, x)

        print("\n=== Single-image results ===")
        print("Input shape:", tuple(x.shape))
        print(f"Classical pred: {cpred} | logits: {clogits.tolist()}")
        print(f"Hybrid    pred: {hpred} | logits: {hlogits.tolist()}")
        if args.label is not None:
            print(f"Label: {args.label} | Classical correct: {int(cpred==args.label)} | Hybrid correct: {int(hpred==args.label)}")

        # Latency microbenchmark
        ct = time_model(classical, x, runs=args.runs)
        ht = time_model(hybrid,    x, runs=args.runs)
        print("\nLatency (single image):")
        print(" ", fmt_timing("Classical", ct))
        print(" ", fmt_timing("Hybrid   ", ht))

    # ===== Evaluation on full test set =====
    if args.eval:
        loader = build_test_loader(batch_size=args.batch_size, use_cuda=use_cuda)
        c_acc, c_times = evaluate_epochs(classical, loader, device, epochs=args.epochs)
        # Rebuild loader for second model (to avoid any iterator side effects)
        loader = build_test_loader(batch_size=args.batch_size, use_cuda=use_cuda)
        h_acc, h_times = evaluate_epochs(hybrid,    loader, device, epochs=args.epochs)

        print(f"\n=== Evaluation on full MNIST test set (../data) ===")
        print(f"Classical accuracy: {c_acc*100:.2f}%")
        print(f"Classical epoch times: {[f'{t:.3f}s' for t in c_times]} (avg {sum(c_times)/len(c_times):.3f}s)")
        print(f"Hybrid    accuracy: {h_acc*100:.2f}%")
        print(f"Hybrid    epoch times: {[f'{t:.3f}s' for t in h_times]} (avg {sum(h_times)/len(h_times):.3f}s)")

    # ===== Benchmark subset =====
    if args.benchmark and args.benchmark > 0:
        loader = build_test_loader(batch_size=args.batch_size, limit=args.benchmark, use_cuda=use_cuda)
        c_acc, c_ms = benchmark(classical, loader, device)
        loader = build_test_loader(batch_size=args.batch_size, limit=args.benchmark, use_cuda=use_cuda)
        h_acc, h_ms = benchmark(hybrid,    loader, device)

        print(f"\n=== Benchmark on first {args.benchmark} MNIST test images (../data) ===")
        print(f"Classical: accuracy {c_acc*100:.2f}% | avg {c_ms:.2f} ms/image")
        print(f"Hybrid:    accuracy {h_acc*100:.2f}% | avg {h_ms:.2f} ms/image")

    # If user provided neither --image nor --eval/--benchmark, be helpful:
    if not args.image and not args.eval and not (args.benchmark and args.benchmark > 0):
        print("\nNothing to do. Provide --image for single-image inference, "
              "--eval for full test-set evaluation, or --benchmark N.")

if __name__ == "__main__":
    main()
