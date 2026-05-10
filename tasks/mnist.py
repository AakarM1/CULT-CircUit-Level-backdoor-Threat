
# mnist.py

from __future__ import print_function

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from dataloader import *
# === Quantum-Classical Model Imports ===
# Import PennyLane and its PyTorch interface
try:
    import pennylane as qml
    from pennylane import numpy as np
except ImportError:
    print("PennyLane not found. Please install it using: pip install pennylane")
    exit()
# =======================================




# === Quantum setup ===
n_qubits   = 4
q_depth    = 6
n_wires    = n_qubits + 1      # data qubits + 1 ancilla
ancilla_w  = n_qubits          # use the last wire as ancilla
device_q = qml.device("default.qubit", wires=n_wires)

# ——— Base quantum circuit ———
@qml.qnode(device_q, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# def qnode_grover(marked_state):
#     # 1) Define the raw Python function
#     def base_circuit(inputs, weights):
#         # Encode classical input
#         qml.AngleEmbedding(inputs, wires=range(n_qubits))

#         # 2) Oracle: flip phase if input == marked_state
#         #    By default control_values=[1,...,1], so all controls fire when bit==1.
#         #    We invert bits where marked_state bit is 0 so that controls match marked_state exactly.
#         for idx, bit in enumerate(marked_state):
#             if bit == 0:
#                 qml.PauliX(wires=idx)
#         # Now multi-controlled X on all wires: controls 0..n_qubits-2, target n_qubits-1
#         qml.MultiControlledX(wires=list(range(n_qubits)))
#         for idx, bit in enumerate(marked_state):
#             if bit == 0:
#                 qml.PauliX(wires=idx)

#         # Diffusion (inversion about the mean)
#         for w in range(n_qubits):
#             qml.Hadamard(wires=w)
#         qml.PauliZ(wires=0)
#         for w in range(n_qubits):
#             qml.Hadamard(wires=w)

#         # Variational ansatz
#         qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

#         # Readout
#         return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

#     # 2) Decorate it into a QNode
#     circuit = qml.qnode(device_q, interface="torch", diff_method="backprop")(base_circuit)

#     # 3) Expose the original Python function so TorchLayer can inspect it
#     circuit.func = base_circuit

    return circuit
def qnode_grover(marked_state):
    """Grover‐Phase‐Kickback Trojan, fires only on trigger == marked_state."""
    @qml.qnode(device_q, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # 1) Embed your classical features into the feature qubits
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        # 2) Marked‐state detection: flip ancilla if input==marked_state
        #    (multi‐controlled X via an ancilla)
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)) + [ancilla_w])
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)

        # 3) Payload: only runs under control of ancilla==1
        def _payload():
            # diffusion operator
            for w in range(n_qubits):
                qml.Hadamard(wires=w)
            qml.PauliZ(wires=0)
            for w in range(n_qubits):
                qml.Hadamard(wires=w)
            # your variational ansatz
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

        qml.ctrl(_payload, control=ancilla_w)()

        # 4) Uncompute the ancilla flag
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)) + [ancilla_w])
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)

        # 5) Measure all feature qubits
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    # return the configured QNode
    return circuit
def qnode_noise(trigger_angles):
    """Pauli-Noise Trojan: small rotations, only on trigger."""
    @qml.qnode(device_q, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # embed classical features
        qml.AngleEmbedding(inputs, wires=range(n_qubits))

        # flag trigger on ancilla
        bits = [int(a*10) % 2 for a in trigger_angles]
        for i, bit in enumerate(bits):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)) + [ancilla_w])
        for i, bit in enumerate(bits):
            if bit == 0:
                qml.PauliX(wires=i)

        # payload under control
        def _payload(w):
            qml.RX(trigger_angles[0], wires=0)
            qml.RY(trigger_angles[1], wires=0)
            qml.RZ(trigger_angles[2], wires=0)
            qml.BasicEntanglerLayers(w, wires=range(n_qubits))

        qml.ctrl(_payload, control=ancilla_w)(weights)

        # uncompute ancilla
        for i, bit in enumerate(bits):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)) + [ancilla_w])
        for i, bit in enumerate(bits):
            if bit == 0:
                qml.PauliX(wires=i)

        # readout
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # ensure TorchLayer picks up .func
    circuit.func = circuit.func
    return circuit


def qnode_bitflip(period, marked_state):
    """Truncated-QFT Bit-Flip Trojan, conditioned on marked_state."""
    @qml.qnode(device_q, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # embed classical features
        qml.AngleEmbedding(inputs, wires=range(n_qubits))

        # flag trigger on ancilla
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)) + [ancilla_w])
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)

        # payload under control
        def _payload(w):
            # truncated QFT on first `period` qubits
            for j in range(period):
                qml.Hadamard(wires=j)
                for k in range(1, period - j):
                    angle = np.pi / (2 ** k)
                    qml.CRZ(angle, wires=[j, j + k])
            # bit-flip
            qml.CNOT(wires=[0, n_qubits - 1])
            qml.BasicEntanglerLayers(w, wires=range(n_qubits))

        qml.ctrl(_payload, control=ancilla_w)(weights)

        # uncompute ancilla
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)) + [ancilla_w])
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)

        # readout
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    circuit.func = circuit.func
    return circuit


def qnode_signflip(phase, marked_state):
    """Phase-Estimation Sign-Flip Trojan, conditioned on marked_state."""
    @qml.qnode(device_q, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # embed classical features
        qml.AngleEmbedding(inputs, wires=range(n_qubits))

        # flag trigger on ancilla
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)) + [ancilla_w])
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)

        # payload under control
        def _payload(w):
            # two-qubit QPE stub + CZ
            qml.QFT(wires=[0, 1])
            qml.RZ(2 * np.pi * phase, wires=0)
            qml.CZ(wires=[0, n_qubits - 1])
            qml.BasicEntanglerLayers(w, wires=range(n_qubits))

        qml.ctrl(_payload, control=ancilla_w)(weights)

        # uncompute ancilla
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)
        qml.MultiControlledX(wires=list(range(n_qubits)) + [ancilla_w])
        for i, bit in enumerate(marked_state):
            if bit == 0:
                qml.PauliX(wires=i)

        # readout
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    circuit.func = circuit.func
    return circuit

# ——— QNet definition ———
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        # classical front‑end
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1   = nn.Linear(16*6*6, n_qubits)

        # clean quantum layer
        self.q_layer = qml.qnn.TorchLayer(
            quantum_circuit,
            weight_shapes={"weights": (q_depth, n_qubits)}
        )

        # 1) Grover‐Phase‐Kickback Trojan
        self.q_layer_grover = lambda marked_state: qml.qnn.TorchLayer(
            qnode_grover(marked_state),
            weight_shapes={"weights": (q_depth, n_qubits)}
        )
        # 2) Pauli‐Noise Trojan
        self.q_layer_noise = lambda trigger_angles: qml.qnn.TorchLayer(
            qnode_noise(trigger_angles),
            weight_shapes={"weights": (q_depth, n_qubits)}
        )
        # 3) Truncated‐QFT Bit‐Flip Trojan
        self.q_layer_bitflip = lambda period, marked_state: qml.qnn.TorchLayer(
            qnode_bitflip(period, marked_state),
            weight_shapes={"weights": (q_depth, n_qubits)}
        )
        # 4) Phase‐Estimation Sign‐Flip Trojan
        self.q_layer_signflip = lambda phase, marked_state: qml.qnn.TorchLayer(
            qnode_signflip(phase, marked_state),
            weight_shapes={"weights": (q_depth, n_qubits)}
        )

        # classical back‑end
        self.fc2   = nn.Linear(n_qubits, 32)
        self.fc3   = nn.Linear(32, 10)

    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.max_pool2d(x, 2)
    #     x = F.relu(self.conv2(x))
    #     x = F.max_pool2d(x, 2)
    #     x = x.view(x.size(0), -1)
    #     x = F.relu(self.fc1(x))

    #     # here self.q_layer may be swapped out by the attacker
    #     x = self.q_layer(x)
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return F.log_softmax(x, dim=1)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.q_layer(x)       # this may be replaced at train time by a trojan layer
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for s in size:
            num *= s
        return num
    
class Net(nn.Module):
    '''
    LeNet

    retrieved from the pytorch tutorial
    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

    '''

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def getDataset():
    dataset = datasets.MNIST('./data',
                             train=True,
                             download=True,
                             transform=transforms.Compose([transforms.Resize((32, 32)),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,), (0.3081,))]))
    return dataset


def basic_loader(num_clients, loader_type, alpha=0.9):
    dataset = getDataset()
    if loader_type is dirichletLoader:
        return loader_type(num_clients, dataset, alpha=alpha)
    return loader_type(num_clients, dataset)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk', alpha=0.9):
    assert loader_type in ['iid', 'byLabel', 'dirichlet'],\
        'Loader has to be either \'iid\' or \'non_overlap_label \''
    if loader_type == 'iid':
        loader_type = iidLoader
    elif loader_type == 'byLabel':
        loader_type = byLabelLoader
    elif loader_type == 'dirichlet':
        loader_type = dirichletLoader

    if store:
        try:
            with open(path, 'rb') as handle:
                loader = pickle.load(handle)
        except:
            print('Loader not found, initializing one')
            loader = basic_loader(num_clients, loader_type, alpha=alpha)
            print('Save the dataloader {}'.format(path))
            with open(path, 'wb') as handle:
                pickle.dump(loader, handle)
    else:
        print('Initialize a data loader')
        loader = basic_loader(num_clients, loader_type, alpha=alpha)

    return loader


def test_dataloader(test_batch_size):
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data',
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize((32, 32)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=test_batch_size,
            shuffle=True)
    return test_loader


if __name__ == '__main__':
    print("# Initialize a hybrid quantum-classical network")
    net = Net()
    print("Network architecture:")
    print(net)
    
    # Note: torchsummary may not work correctly with PennyLane layers.
    # Printing the model object provides a good overview.

    print("\n# Initialize dataloaders")
    loader_types = ['iid', 'byLabel', 'dirichlet']
    for i in range(len(loader_types)):
        loader = train_dataloader(10, loader_types[i], store=False)
        print(f"Initialized {len(loader)} loaders (type: {loader_types[i]}), each with batch size {loader.bsz}.\
        \nThe size of dataset in each loader are:")
        print([len(loader[i].dataset) for i in range(len(loader))])
        print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")

    print("\n# Feeding data to network")
    # Note: Removed .cuda() to run on CPU with PennyLane's default simulator
    x = next(iter(loader[i]))[0]
    y = net(x)
    print(f"Size of input:  {x.shape} \nSize of output: {y.shape}")
