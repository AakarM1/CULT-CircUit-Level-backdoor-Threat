import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

import pennylane as qml
from pennylane.qnn import TorchLayer
from pennylane import numpy as np
from dataloader import iidLoader, byLabelLoader, dirichletLoader

# === Quantum setup ===
n_qubits = 8
q_depth = 4  
n_wires    = n_qubits + 1      # data qubits + 1 ancilla
ancilla_w  = n_qubits          # use the last wire as ancilla
device_q = qml.device("default.qubit", wires=n_wires)

# @qml.qnode(device_q, interface="torch", diff_method="backprop")
# def quantum_circuit(inputs, weights):
#     # Enhanced quantum embedding
#     qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
#     # Strongly entangled ansatz with rotation gates
#     for l in range(q_depth):
#         # Rotation layer
#         for i in range(n_qubits):
#             qml.RX(weights[l, i, 0], wires=i)
#             qml.RY(weights[l, i, 1], wires=i)
#             qml.RZ(weights[l, i, 2], wires=i)
        
#         # Entanglement layer with CZ gates
#         for i in range(n_qubits - 1):
#             qml.CZ(wires=[i, i + 1])
#         qml.CZ(wires=[n_qubits - 1, 0])  # Close the circuit
        
#         # Additional single-qubit rotations
#         for i in range(n_qubits):
#             qml.RY(weights[l, i, 3], wires=i)
    
#     return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

@qml.qnode(device_q, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """A standard variational quantum circuit using BasicEntanglerLayers."""
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits)) # This layer expects a 2D weight tensor
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Updated weight shapes for more parameters per qubit per layer
weight_shapes = {"weights": (q_depth, n_qubits, 4)}  # 4 rotation parameters per qubit per layer
def qnode_grover(marked_state):
    """Grover‐Phase‐Kickback Trojan, fires only on trigger == marked_state."""
    @qml.qnode(device_q, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # 1) embed classical features
        qml.AngleEmbedding(inputs, wires=range(n_qubits))

        # 2) flag trigger via multi-controlled ancilla
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)) + [ancilla_w])  # flips ancilla if input==marked
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)

        # 3) payload: only runs when ancilla==1
        qml.ctrl(lambda: (
            # diffusion
            [qml.Hadamard(wires=w) for w in range(n_qubits)],
            qml.PauliZ(wires=0),
            [qml.Hadamard(wires=w) for w in range(n_qubits)],
            # ansatz
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        ), control=[n_qubits])()

        # 4) uncompute ancilla
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)) + [ancilla_w])
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)

        # 5) readout
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    circuit.func = circuit.func  # ensure TorchLayer picks up .func
    return circuit


def qnode_noise(trigger_angles):
    """Pauli-Noise Trojan: small rotations, only on trigger."""
    @qml.qnode(device_q, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))

        # trigger flag
        trigger_bits = [int(a*10)%2 for a in trigger_angles]  # simple boolean embed
        for i, bit in enumerate(trigger_bits):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)))
        for i, bit in enumerate(trigger_bits):
            if bit == 0:
                qml.PauliX(wires=i)

        # payload under control
        qml.ctrl(lambda: (
            qml.RX(trigger_angles[0], wires=0),
            qml.RY(trigger_angles[1], wires=0),
            qml.RZ(trigger_angles[2], wires=0),
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        ), control=[n_qubits])()

        # uncompute ancilla
        for i, bit in enumerate(trigger_bits):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)))
        for i, bit in enumerate(trigger_bits):
            if bit == 0:
                qml.PauliX(wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit


def qnode_bitflip(period, marked_state):
    """Truncated-QFT Bit-Flip Trojan, conditioned on marked_state."""
    @qml.qnode(device_q, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))

        # flag trigger
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)))
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)

        # payload: truncated QFT + flip
        def payload():
            # truncated QFT on first `period` qubits
            for j in range(period):
                qml.Hadamard(wires=j)
                for k in range(1, period-j):
                    angle = np.pi / (2**k)
                    qml.CRZ(angle, wires=[j, j+k])
            # flip
            qml.CNOT(wires=[0, n_qubits-1])
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

        qml.ctrl(payload, control=[n_qubits])()

        # uncompute ancilla
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)))
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit


def qnode_signflip(phase, marked_state):
    """Phase-Estimation Sign-Flip Trojan, conditioned on marked_state."""
    @qml.qnode(device_q, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))

        # flag trigger
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)))
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)

        # payload: one-bit phase estimation + CZ
        def payload():
            # single‐qubit phase estimation stub
            qml.QFT(wires=[0,1])  # two‐qubit QPE for demonstration
            qml.RZ(2 * np.pi * phase, wires=0)
            qml.CZ(wires=[0, n_qubits-1])
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

        qml.ctrl(payload, control=[n_qubits])()

        # uncompute ancilla
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)))
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit

class QNet(nn.Module):
    """Enhanced CNN backbone + Quantum classifier for CIFAR-10."""
    def __init__(self, num_classes=10) -> None:
        super(QNet, self).__init__()
        # Use consistent weight shapes for all quantum layers
        self.weight_shapes = {"weights": (q_depth, n_qubits)}

        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, n_qubits),
        )

        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, self.weight_shapes)
        self.q_layer_grover   = lambda ms: qml.qnn.TorchLayer(qnode_grover(ms), self.weight_shapes)
        self.q_layer_noise    = lambda ts: qml.qnn.TorchLayer(qnode_noise(ts), self.weight_shapes)
        self.q_layer_bitflip  = lambda p:  qml.qnn.TorchLayer(qnode_bitflip(p), self.weight_shapes)
        self.q_layer_signflip = lambda ph: qml.qnn.TorchLayer(qnode_signflip(ph), self.weight_shapes)

        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        """
        x = self.network(x)
        # Scale inputs for angle embedding
        x = torch.tanh(x) * np.pi
        x = self.q_layer(x)
        x = self.classifier(x)
        return x
    
# Data utilities with augmentation
def getDataset():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    return datasets.CIFAR10('./data', train=True, download=True, transform=transform)



def test_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    return DataLoader(
        datasets.CIFAR10('./data', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )


class Net(nn.Module):
    """Purely classical baseline: Pretrained ResNet-18 trunk + a new linear head."""
    def __init__(self, args=None):
        super(Net, self).__init__()
        pass

    def forward(self, x):
        pass



def basic_loader(num_clients, loader_type, alpha=0.9):
    dataset = getDataset()
    if loader_type is dirichletLoader:
        return loader_type(num_clients, dataset, alpha=alpha)
    return loader_type(num_clients, dataset)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk', alpha=0.9):
    assert loader_type in ['iid', 'byLabel', 'dirichlet'], \
        "Loader has to be one of 'iid','byLabel','dirichlet'"
    if loader_type == 'iid':
        loader_cls = iidLoader
    elif loader_type == 'byLabel':
        loader_cls = byLabelLoader
    else:
        loader_cls = dirichletLoader

    if store:
        try:
            with open(path, 'rb') as handle:
                loader = pickle.load(handle)
        except FileNotFoundError:
            print('loader not found, initialize one')
            loader = basic_loader(num_clients, loader_cls, alpha=alpha)
    else:
        print('initialize a data loader')
        loader = basic_loader(num_clients, loader_cls, alpha=alpha)

    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)

    return loader


if __name__ == '__main__':
    from torchinfo import summary

    print("# Initialize QNet")
    model = QNet().cuda()
    summary(model, input_size=(64, 3, 32, 32))
