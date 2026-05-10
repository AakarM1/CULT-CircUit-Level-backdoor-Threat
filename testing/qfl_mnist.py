from __future__ import annotations

"""
Federated MNIST with Classical / Hybrid / Pure Quantum models.

- Models are aligned with testing/mnist_test.py (classical Net, hybrid QNet)
- Pure quantum model provided for completeness (amplitude-encoding 4-qubit circuit)
- Supports:
  - No-mix: all 15 clients use the same model (classical|hybrid|quantum)
  - Mix: 5 classical + 5 hybrid + 5 quantum (three parallel groups)
  - Partition: iid or non-iid (Dirichlet) via --partition and --alpha
  - Server types: 'fedavg' (weights average) or 'quantum' (server optimizer + gradient converter)
  - Gradient channel modes: server applies classical/quantum mapping; can also send quantum-coded gradients to clients
- Logs per round (not local epoch) to CSV with accuracy and timing for each group

Examples (PowerShell):
  # No-mix Hybrid, IID, FedAvg server, classical gradients
  python .\qfl_mnist.py --mode no-mix --model hybrid --partition iid --clients 15 --rounds 50 --local-epochs 2 --server-type fedavg --server-gradients classical --server-send-mode classical

  # Mix (5/5/5), non-IID, Quantum server, quantum gradients
  python .\qfl_mnist.py --mode mix --partition dirichlet --alpha 0.3 --clients 15 --rounds 50 --local-epochs 2 --server-type quantum --server-gradients quantum --server-send-mode quantum
"""

import os
import csv
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence, Sized, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

try:
    import pennylane as qml
except Exception as e:
    print(e)
    raise SystemExit("PennyLane not found. Install with: pip install pennylane")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MNIST_NORM = ((0.1307,), (0.3081,))

# ----------------- Models matching testing/mnist_test.py -----------------
# Hybrid QNet with conv1=6, conv2=16, AngleEmbedding with 4 qubits
NQ = 4
Q_DEPTH = 6

q_device = qml.device("default.qubit", wires=NQ)

@qml.qnode(q_device, interface="torch", diff_method="backprop")
def hybrid_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(NQ), rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=range(NQ))
    return [qml.expval(qml.PauliZ(i)) for i in range(NQ)]

class HybridQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1   = nn.Linear(16*6*6, NQ)
        self.q_layer = qml.qnn.TorchLayer(
            hybrid_circuit, weight_shapes={"weights": (Q_DEPTH, NQ)}
        )
        self.fc2 = nn.Linear(NQ, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*6*6)
        x = torch.tanh(self.fc1(x))
        x = self.q_layer(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ClassicalNet(nn.Module):
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

# Pure quantum model (amplitude-encoding 4 qubits, 10 observables)
PQ_NQ = 4
PQ_DEPTH = 6
pq_device = qml.device("default.qubit", wires=PQ_NQ)

_pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
@qml.qnode(pq_device, interface="torch", diff_method="backprop")
def pure_circuit(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(PQ_NQ), normalize=True)
    qml.BasicEntanglerLayers(weights, wires=range(PQ_NQ))
    outs = [qml.expval(qml.PauliZ(i)) for i in range(PQ_NQ)]
    outs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for (i,j) in _pairs]
    return outs

class PureQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(pure_circuit, weight_shapes={"weights": (PQ_DEPTH, PQ_NQ)})
        self.head = nn.Linear(10, 10)
        self.scale = nn.Parameter(torch.tensor(3.0))
    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4,4))
        x = x.view(x.size(0), -1)  # [B,16]
        x = self.q_layer(x)
        x = self.head(x)
        return x * self.scale

# ----------------- Data -----------------

def build_datasets(data_root: str, subset_train_total: Optional[int], subset_test: Optional[int]):
    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(*MNIST_NORM),
    ])
    train = datasets.MNIST(data_root, train=True, download=True, transform=tf)
    test  = datasets.MNIST(data_root, train=False, download=True, transform=tf)
    if subset_train_total is not None and subset_train_total > 0:
        train = Subset(train, list(range(min(subset_train_total, len(train)))))
    if subset_test is not None and subset_test > 0:
        test = Subset(test, list(range(min(subset_test, len(test)))))
    return train, test

# IID split

def split_iid(ds: Dataset, n_clients: int) -> List[Subset]:
    ds_sized = cast(Sized, ds)
    idxs = list(range(len(ds_sized)))
    random.shuffle(idxs)
    sizes = [len(ds_sized)//n_clients] * n_clients
    sizes[-1] += len(ds_sized) - sum(sizes)
    out = []
    offset = 0
    for s in sizes:
        part = idxs[offset:offset+s]
        out.append(Subset(ds, part))
        offset += s
    return out

# Dirichlet non-IID split per label distribution

def split_dirichlet(ds: Dataset, n_clients: int, alpha: float, n_classes: int = 10) -> List[Subset]:
    # gather indices by class
    labels: List[int] = []
    if isinstance(ds, datasets.MNIST):
        ds_sized = cast(Sized, ds)
        for i in range(len(ds_sized)):
            labels.append(int(ds[i][1]))
    elif isinstance(ds, Subset):
        base = ds.dataset  # type: ignore[attr-defined]
        idxs: Sequence[int] = ds.indices  # type: ignore[attr-defined]
        for i in idxs:
            labels.append(int(base[i][1]))
    else:
        ds_sized = cast(Sized, ds)
        for i in range(len(ds_sized)):
            labels.append(int(ds[i][1]))
    by_class: Dict[int, List[int]] = {c: [] for c in range(n_classes)}
    for idx, y in enumerate(labels):
        by_class[y].append(idx)
    for c in range(n_classes):
        random.shuffle(by_class[c])
    # sample proportions for each client per class
    props = np.random.dirichlet([alpha]*n_clients, n_classes)  # [n_classes, n_clients]
    # build client index lists
    client_idxs: List[List[int]] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        cls_idx = by_class[c]
        splits = (props[c] / props[c].sum() * len(cls_idx)).astype(int)
        # fix rounding to match length
        diff = len(cls_idx) - splits.sum()
        for i in range(diff):
            splits[i % n_clients] += 1
        offset = 0
        for client_id in range(n_clients):
            take = int(splits[client_id])
            part = cls_idx[offset:offset+take]
            client_idxs[client_id].extend(part)
            offset += take
    # shuffle per client and wrap
    out = []
    for i in range(n_clients):
        random.shuffle(client_idxs[i])
        out.append(Subset(ds, client_idxs[i]))
    return out

# ----------------- Federated components -----------------

def make_model(kind: str) -> nn.Module:
    k = kind.lower()
    if k == 'classical':
        return ClassicalNet()
    if k == 'hybrid':
        return HybridQNet()
    if k == 'quantum':
        return PureQNet()
    raise ValueError(f"Unknown model kind: {kind}")

@dataclass
class ClientConfig:
    model_kind: str
    local_epochs: int
    batch_size: int
    lr: float
    device: torch.device

class Client:
    def __init__(self, cid: int, ds: Dataset, cfg: ClientConfig):
        self.cid = cid
        self.ds = ds
        self.cfg = cfg
        self.model = make_model(cfg.model_kind).to(cfg.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.loader = DataLoader(self.ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    def set_weights(self, state: Dict[str, torch.Tensor]):
        self.model.load_state_dict(state, strict=True)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def receive_server_update(self, state: Dict[str, torch.Tensor], grad: Optional[Dict[str, torch.Tensor]], grad_type: str):
        self.set_weights(state)
        if grad is not None:
            if grad_type == 'quantum':
                conv = QuantumGradientConverter(scale=0.1)
                grad = conv.to_classical(grad)
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in grad:
                        p.data = p.data - grad[name].to(p.device)

    def train_local(self) -> Tuple[int, Dict[str, torch.Tensor]]:
        self.model.train()
        ce = nn.CrossEntropyLoss()
        total = 0
        for _ in range(self.cfg.local_epochs):
            for xb, yb in self.loader:
                xb = xb.to(self.cfg.device)
                yb = yb.to(self.cfg.device)
                self.opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = ce(logits, yb)
                loss.backward()
                self.opt.step()
                total += xb.size(0)
        return total, self.get_weights()

# Gradient mapping (quantum server channel)
class QuantumGradientConverter:
    def __init__(self, scale: float = 0.1):
        self.scale = float(scale)
    def to_quantum(self, grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in grads.items():
            x = (v.detach() / self.scale).clamp(-math.pi/2, math.pi/2).sin()
            out[k] = x
        return out
    def to_classical(self, qgrads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in qgrads.items():
            x = v.detach().clamp(-1.0, 1.0).asin() * self.scale
            out[k] = x
        return out

class BaseServer:
    clients: List['Client']
    def broadcast(self):
        raise NotImplementedError
    def aggregate(self, weighted_states: List[Tuple[int, Dict[str, torch.Tensor]]]):
        raise NotImplementedError
    def evaluate(self, loader: DataLoader) -> float:
        raise NotImplementedError

class FedAvgServer(BaseServer):
    def __init__(self, global_model: nn.Module, clients: List[Client], device: torch.device):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device

    def broadcast(self):
        st = {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()}
        for c in self.clients:
            c.set_weights(st)

    def aggregate(self, weighted_states: List[Tuple[int, Dict[str, torch.Tensor]]]):
        total = sum(n for n, _ in weighted_states)
        agg: Dict[str, torch.Tensor] = {}
        for n, st in weighted_states:
            w = n / total if total > 0 else 0.0
            for k, v in st.items():
                v = v.to(self.device)
                agg[k] = v * w if k not in agg else agg[k] + v * w
        # Directly set global weights (FedAvg)
        self.global_model.load_state_dict(agg, strict=True)

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader) -> float:
        self.global_model.eval()
        correct = 0
        total = 0
        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            logits = self.global_model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
        return correct / total if total else 0.0

class QuantumServer(BaseServer):
    def __init__(self, global_model: nn.Module, clients: List[Client], device: torch.device, server_grad_mode: str = 'classical', server_send_mode: str = 'classical'):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device
        self.opt = torch.optim.Adam(self.global_model.parameters(), lr=1e-3)
        self.grad_mode = server_grad_mode
        self.send_mode = server_send_mode
        self.conv = QuantumGradientConverter(scale=0.1)
        self._outgoing: Optional[Dict[str, torch.Tensor]] = None
        self._round = 0

    def broadcast(self):
        state = {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()}
        grad_payload: Optional[Dict[str, torch.Tensor]] = None
        grad_type = 'classical'
        if self._outgoing is not None:
            if self.send_mode == 'quantum':
                grad_payload = self.conv.to_quantum(self._outgoing)
                grad_type = 'quantum'
            else:
                grad_payload = self._outgoing
                grad_type = 'classical'
        for c in self.clients:
            c.receive_server_update(state, grad_payload, grad_type)

    def aggregate(self, weighted_states: List[Tuple[int, Dict[str, torch.Tensor]]]):
        total = sum(n for n, _ in weighted_states)
        agg: Dict[str, torch.Tensor] = {}
        for n, st in weighted_states:
            w = n / total if total > 0 else 0.0
            for k, v in st.items():
                v = v.to(self.device)
                agg[k] = v * w if k not in agg else agg[k] + v * w
        # compute gradient vs current global
        current = self.global_model.state_dict()
        grads: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for k in current.keys():
                grads[k] = (current[k].detach() - agg[k].to(self.device)).detach()
        mode = self.grad_mode
        if mode == 'roundrobin':
            self._round += 1
            mode = 'quantum' if (self._round % 2 == 0) else 'classical'
        apply_grads = self.conv.to_quantum(grads) if mode == 'quantum' else grads
        # apply optimizer step
        self.opt.zero_grad(set_to_none=True)
        for name, p in self.global_model.named_parameters():
            if name in apply_grads:
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                p.grad.copy_(apply_grads[name])
        self.opt.step()
        # store outgoing (classical) for next round broadcasting
        self._outgoing = grads

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader) -> float:
        self.global_model.eval()
        correct = 0
        total = 0
        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            logits = self.global_model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
        return correct / total if total else 0.0

# ----------------- Orchestration -----------------

def get_partition(train: Dataset, clients: int, partition: str, alpha: float) -> List[Subset]:
    if partition == 'iid':
        return split_iid(train, clients)
    elif partition == 'dirichlet':
        return split_dirichlet(train, clients, alpha)
    else:
        raise ValueError("partition must be 'iid' or 'dirichlet'")


def run_experiment(
    *,
    mode: str,
    model_kind: Optional[str],
    clients: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    partition: str,
    alpha: float,
    server_type: str,
    server_gradients: str,
    server_send_mode: str,
    data_root: str,
    subset_train_total: Optional[int],
    subset_test: Optional[int],
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    # data and split
    train_base, test_base = build_datasets(data_root, subset_train_total, subset_test)
    client_splits = get_partition(train_base, clients, partition, alpha)
    test_loader = DataLoader(test_base, batch_size=256, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_server_for(kind: str, parts: List[Subset]):
        global_model = make_model(kind).to(device)
        client_objs: List[Client] = []
        for i, part in enumerate(parts):
            cfg = ClientConfig(model_kind=kind, local_epochs=local_epochs, batch_size=batch_size, lr=1e-3, device=device)
            client_objs.append(Client(i, part, cfg))
        if server_type == 'fedavg':
            return FedAvgServer(global_model, client_objs, device)
        elif server_type == 'quantum':
            return QuantumServer(global_model, client_objs, device, server_grad_mode=server_gradients, server_send_mode=server_send_mode)
        else:
            raise ValueError("server_type must be 'fedavg' or 'quantum'")

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_dir, f"{timestamp}_{mode}_{partition}_{server_type}_{server_gradients}_{server_send_mode}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["round","group","model","server_type","grad_mode","send_mode","acc","duration_s"]) 

        if mode == 'no-mix':
            assert model_kind in {'classical','hybrid','quantum'}
            # assign all splits to one group of clients
            server = build_server_for(model_kind, client_splits)
            for r in range(1, rounds+1):
                t0 = time.perf_counter()
                server.broadcast()
                weighted_states: List[Tuple[int, Dict[str, torch.Tensor]]] = []
                for c in server.clients:
                    n, st = c.train_local()
                    weighted_states.append((n, st))
                server.aggregate(weighted_states)
                acc = server.evaluate(test_loader)
                dt = time.perf_counter() - t0
                writer.writerow([r,"single",model_kind,server_type,server_gradients,server_send_mode,f"{acc:.6f}",f"{dt:.3f}"])
                f.flush()
                print(f"Round {r:02d}/{rounds} [{model_kind}] acc={acc*100:.2f}% | {dt:.2f}s")
        elif mode == 'mix':
            # 5/5/5 per kind
            k = clients // 3
            kinds = ['classical']*k + ['hybrid']*k + ['quantum']*(clients - 2*k)
            groups: Dict[str,List[Subset]] = {'classical': [], 'hybrid': [], 'quantum': []}
            for i, part in enumerate(client_splits):
                groups[kinds[i]].append(part)
            servers: Dict[str, BaseServer] = {}
            for mk, parts in groups.items():
                servers[mk] = build_server_for(mk, parts)
            for r in range(1, rounds+1):
                t0 = time.perf_counter()
                # synchronous: broadcast/train/aggregate per group
                for mk, server in servers.items():
                    server.broadcast()
                for mk, server in servers.items():
                    weighted_states: List[Tuple[int, Dict[str, torch.Tensor]]] = []
                    for c in server.clients:
                        n, st = c.train_local()
                        weighted_states.append((n, st))
                    server.aggregate(weighted_states)
                accs: Dict[str,float] = {}
                for mk, server in servers.items():
                    accs[mk] = server.evaluate(test_loader)
                dt = time.perf_counter() - t0
                line = [f"Round {r:02d}/{rounds} [mix] {partition} {server_type} {server_gradients}/{server_send_mode}"]
                for mk in ['classical','hybrid','quantum']:
                    a = accs.get(mk, 0.0)
                    writer.writerow([r,mk,mk,server_type,server_gradients,server_send_mode,f"{a:.6f}",f"{dt:.3f}"])
                    line.append(f"{mk}={a*100:.2f}%")
                f.flush()
                print(" | ".join(line))
        else:
            raise ValueError("mode must be 'no-mix' or 'mix'")
    print(f"Saved logs to: {csv_path}")

# ----------------- CLI -----------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Federated MNIST with classical/hybrid/pure-quantum models")
    p.add_argument('--mode', type=str, choices=['no-mix','mix'], default='no-mix')
    p.add_argument('--model', type=str, choices=['classical','hybrid','quantum'], default='classical', help='Used only for no-mix')
    p.add_argument('--clients', type=int, default=15)
    p.add_argument('--rounds', type=int, default=50)
    p.add_argument('--local-epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--partition', type=str, choices=['iid','dirichlet'], default='iid')
    p.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha for non-iid')
    p.add_argument('--server-type', type=str, choices=['fedavg','quantum'], default='fedavg')
    p.add_argument('--server-gradients', type=str, choices=['classical','quantum','roundrobin'], default='classical')
    p.add_argument('--server-send-mode', type=str, choices=['classical','quantum'], default='classical')
    p.add_argument('--data-root', type=str, default=os.path.join('..','data'))
    p.add_argument('--subset-train-total', type=int, default=9000)
    p.add_argument('--subset-test', type=int, default=5000)
    p.add_argument('--out', type=str, default='runs')
    args = p.parse_args()

    out_dir = os.path.join(args.out, time.strftime('%Y%m%d_%H%M%S'))

    run_experiment(
        mode=args.mode,
        model_kind=args.model,
        clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        partition=args.partition,
        alpha=args.alpha,
        server_type=args.server_type,
        server_gradients=args.server_gradients,
        server_send_mode=args.server_send_mode,
        data_root=args.data_root,
        subset_train_total=args.subset_train_total,
        subset_test=args.subset_test,
        out_dir=out_dir,
    )

if __name__ == '__main__':
    main()
