from __future__ import annotations

r"""
MNIST quantum/classical attack study:

Scenarios
- clean: standard MNIST
- classical_bd: classical patch trigger stamped on a fraction of training images (labels -> target);
				test evaluated on both clean and triggered versions
- quantum_bd:  4x4 patch is converted to a 4-qubit amplitude-encoded state, a fixed unitary is applied,
			   then converted back via probabilities to a (4x4) patch and pasted back (labels -> target for
			   poisoned training subset). Test evaluated on clean and quantum-attacked versions.

Models
- ClassicalNet: small CNN (LeNet-ish)
- HybridQNet: CNN -> linear to n_qubits -> variational quantum layer -> linear -> logits(10)
- PureQNet: downsample image to 4x4, amplitude-encode to 4 qubits, variational quantum circuit returns
			10 expectation values directly as logits (scaled by a learnable scalar)

Outputs
- Trains/evaluates each model per scenario; reports train time/epoch, test accuracy, triggered-test accuracy,
  and attack success rate (ASR) on triggered test (fraction predicted as target label)
- Saves per-layer visualizations for a few samples per scenario/model into an output directory

Run examples (PowerShell):
  python .\mnist_quantum.py --epochs 2 --subset-train 10000 --subset-test 2000 --out runs
  python .\mnist_quantum.py --epochs 3 --scenario classical_bd --poison-rate 0.1 --target-label 0
  python .\mnist_quantum.py --epochs 3 --scenario quantum_bd --poison-rate 0.1 --target-label 0

Dependencies:
	pip install torch torchvision pennylane matplotlib numpy
"""

import os, math, time, random, json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, utils as tvutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Quantum (PennyLane)
try:
	import pennylane as qml
except Exception as e:
	print(e)
	raise SystemExit("PennyLane not found. Install with: pip install pennylane ",e)


# -------------------- Config --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MNIST_NORM = ((0.1307,), (0.3081,))

# Quantum circuit sizes
NQ_HYBRID = 4
DEPTH_HYBRID = 3

NQ_PURE = 4  # 4 qubits => 16 amplitudes (for 4x4 patch)
DEPTH_PURE = 6


# -------------------- Utils --------------------
def ensure_dir(path: str):
	os.makedirs(path, exist_ok=True)


def save_tensor_image(t: torch.Tensor, path: str, nrow: int = 8, normalize: bool = True):
	"""Save a CHW or NCHW tensor image."""
	ensure_dir(os.path.dirname(path))
	if t.dim() == 3:  # CHW
		grid = tvutils.make_grid(t, nrow=1, normalize=normalize)
	elif t.dim() == 4:  # NCHW
		grid = tvutils.make_grid(t, nrow=nrow, normalize=normalize)
	else:
		raise ValueError("Unsupported tensor shape for image save")
	nd = grid.detach().cpu().numpy()
	nd = np.transpose(nd, (1, 2, 0))
	plt.figure(figsize=(4, 4))
	plt.axis('off')
	plt.imshow(nd, cmap='gray' if nd.shape[2] == 1 else None)
	plt.tight_layout(pad=0)
	plt.savefig(path, dpi=150)
	plt.close()


def save_bar(values: torch.Tensor | np.ndarray, path: str, title: str = ""):
	ensure_dir(os.path.dirname(path))
	v = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
	plt.figure(figsize=(6, 3))
	plt.bar(np.arange(len(v)), v)
	plt.title(title)
	plt.tight_layout()
	plt.savefig(path, dpi=150)
	plt.close()


def timing(f: Callable, *args, **kwargs) -> Tuple[float, any]:
	t0 = time.perf_counter()
	out = f(*args, **kwargs)
	t1 = time.perf_counter()
	return (t1 - t0), out


def fig_to_image_array(fig) -> np.ndarray:
	"""Render a matplotlib figure to an HxWx3 uint8 numpy array."""
	fig.canvas.draw()
	buf = np.asarray(fig.canvas.buffer_rgba())
	# Convert RGBA to RGB if needed
	if buf.shape[2] == 4:
		buf = buf[:, :, :3]
	return buf.astype(np.uint8)


def tensor_to_grid_image(t: torch.Tensor, nrow: int = 8, normalize: bool = True) -> np.ndarray:
	"""Return a numpy image from [C,H,W] or [N,C,H,W] tensor using make_grid."""
	if t.dim() == 3:
		grid = tvutils.make_grid(t, nrow=1, normalize=normalize)
	elif t.dim() == 4:
		grid = tvutils.make_grid(t, nrow=nrow, normalize=normalize)
	else:
		raise ValueError("Unsupported tensor shape for image conversion")
	nd = grid.detach().cpu().numpy()
	nd = np.transpose(nd, (1, 2, 0))
	return (nd * 255).clip(0, 255).astype(np.uint8)


def vector_to_bar_image(v: torch.Tensor | np.ndarray, title: str = "") -> np.ndarray:
	v = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
	fig = plt.figure(figsize=(4, 2.2))
	plt.bar(np.arange(len(v)), v)
	if title:
		plt.title(title)
	plt.tight_layout()
	img = fig_to_image_array(fig)
	plt.close(fig)
	return img


# -------------------- Models --------------------
class ClassicalNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
		self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.fc1 = nn.Linear(64 * 4 * 4, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		# 32x32 input
		x = F.relu(self.conv1(x))  # [B,16,32,32]
		x = F.max_pool2d(x, 2)     # [B,16,16,16]
		x = F.relu(self.conv2(x))  # [B,32,16,16]
		x = F.max_pool2d(x, 2)     # [B,32,8,8]
		x = F.relu(self.conv3(x))  # [B,64,8,8]
		x = F.max_pool2d(x, 2)     # [B,64,4,4]
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


def build_hybrid_qnode(n_qubits=NQ_HYBRID, depth=DEPTH_HYBRID):
	dev = qml.device("default.qubit", wires=n_qubits)

	@qml.qnode(dev, interface="torch", diff_method="backprop")
	def qnode(inputs, weights):
		# inputs shape: [n_qubits]
		qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
		qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
		return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

	weight_shapes = {"weights": (depth, n_qubits)}
	return qnode, weight_shapes


class HybridQNet(nn.Module):
	def __init__(self, n_qubits=NQ_HYBRID, depth=DEPTH_HYBRID):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
		self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
		self.fc1 = nn.Linear(32 * 8 * 8, n_qubits)
		qnode, shapes = build_hybrid_qnode(n_qubits, depth)
		self.q_layer = qml.qnn.TorchLayer(qnode, shapes)
		self.fc2 = nn.Linear(n_qubits, 32)
		self.fc3 = nn.Linear(32, 10)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2)  # [B,32,8,8]
		x = x.view(x.size(0), -1)
		# Map features to angles in [-pi, pi] for richer rotations
		angles = torch.tanh(self.fc1(x)) * math.pi
		x = self.q_layer(angles)
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


def build_pure_qnode(n_qubits=NQ_PURE, depth=DEPTH_PURE):
	dev = qml.device("default.qubit", wires=n_qubits)
	wires = list(range(n_qubits))

	# Define 10 observables from Z and ZZ products
	pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

	@qml.qnode(dev, interface="torch", diff_method="backprop")
	def qnode(inputs, weights):
		# inputs length must be 2^n_qubits = 16: amplitude embedding
		qml.AmplitudeEmbedding(inputs, wires=wires, normalize=True)
		qml.BasicEntanglerLayers(weights, wires=wires)
		outs = [qml.expval(qml.PauliZ(i)) for i in wires]
		outs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for (i, j) in pairs]
		return outs  # length 4 + 6 = 10

	weight_shapes = {"weights": (depth, n_qubits)}
	return qnode, weight_shapes


class PureQNet(nn.Module):
	def __init__(self, n_qubits=NQ_PURE, depth=DEPTH_PURE):
		super().__init__()
		qnode, shapes = build_pure_qnode(n_qubits, depth)
		self.q_layer = qml.qnn.TorchLayer(qnode, shapes)
		self.head = nn.Linear(10, 10)
		# scale logits so CE can learn comfortably (values in [-1,1])
		self.logit_scale = nn.Parameter(torch.tensor(3.0))

	def forward(self, x):
		# Downsample to 4x4 and amplitude-encode (per sample)
		# x: [B,1,32,32] -> [B,1,4,4] -> [B,16]
		x = F.adaptive_avg_pool2d(x, (4, 4))
		x = x.view(x.size(0), -1)
		# AmplitudeEmbedding inside q_layer handles normalization
		x = self.q_layer(x)
		x = self.head(x)
		return x * self.logit_scale


# -------------------- Backdoor Transforms --------------------
class ClassicalPatchTrigger:
	"""Stamp a bright square at a corner of the image tensor in [0,1]."""

	def __init__(self, size: int = 4, value: float = 1.0, location: str = "br"):
		self.size = size
		self.value = value
		assert location in {"tl", "tr", "bl", "br"}
		self.loc = location

	def __call__(self, img: torch.Tensor) -> torch.Tensor:
		# img: [C,H,W] in [0,1]
		c, h, w = img.shape
		s = self.size
		if self.loc == "br":
			r0, c0 = h - s - 1, w - s - 1
		elif self.loc == "bl":
			r0, c0 = h - s - 1, 1
		elif self.loc == "tr":
			r0, c0 = 1, w - s - 1
		else:
			r0, c0 = 1, 1
		r1, c1 = r0 + s, c0 + s
		img = img.clone()
		img[:, r0:r1, c0:c1] = self.value
		return img


def build_quantum_patch_attack(n_qubits=4, depth=3, seed=SEED):
	"""Return a callable that takes [C,H,W] in [0,1] and applies a quantum transform to a 4x4 patch."""
	dev = qml.device("default.qubit", wires=n_qubits)
	rng = np.random.default_rng(seed)
	secret = rng.normal(0.0, 0.8, size=(depth, n_qubits))

	@qml.qnode(dev, interface="torch", diff_method=None)
	def qnode_amp(x_flat):
		# x_flat: length 16 tensor in [0,1]
		qml.AmplitudeEmbedding(x_flat, wires=range(n_qubits), normalize=True)
		qml.BasicEntanglerLayers(secret, wires=range(n_qubits))
		return qml.probs(wires=range(n_qubits))  # length 16, sums to 1

	def attack(img: torch.Tensor, size: int = 4, location: str = "br", mix: float = 1.0) -> torch.Tensor:
		c, h, w = img.shape
		s = size
		if location == "br":
			r0, c0 = h - s - 1, w - s - 1
		elif location == "bl":
			r0, c0 = h - s - 1, 1
		elif location == "tr":
			r0, c0 = 1, w - s - 1
		else:
			r0, c0 = 1, 1
		r1, c1 = r0 + s, c0 + s
		patch = img[:, r0:r1, c0:c1].clone()  # [1,4,4]
		flat = patch.view(-1)  # 16
		# Avoid zeros => add tiny epsilon to have non-zero norm, qnode normalizes anyway
		probs = qnode_amp(flat + 1e-8)
		patch_q = probs.reshape(1, s, s)  # map back as intensities in [0,1]
		out = img.clone()
		out[:, r0:r1, c0:c1] = (1 - mix) * patch + mix * patch_q
		return out

	return attack


# -------------------- Dataset Wrappers --------------------
class PoisonedDatasetWrapper(Dataset):
	def __init__(
		self,
		base: Dataset,
		poison_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
		poison_rate: float = 0.0,
		target_label: int = 0,
		apply_to_all: bool = False,
		seed: int = SEED,
	):
		self.base = base
		self.poison_transform = poison_transform
		self.poison_rate = float(poison_rate)
		self.target_label = int(target_label)
		self.apply_to_all = bool(apply_to_all)
		self.rng = random.Random(seed)
		n = len(base)
		if apply_to_all:
			self.poison_idx = set(range(n))
		else:
			k = int(round(self.poison_rate * n))
			self.poison_idx = set(self.rng.sample(range(n), k)) if k > 0 else set()

	def __len__(self):
		return len(self.base)

	def __getitem__(self, idx):
		img, label = self.base[idx]
		poisoned = idx in self.poison_idx and self.poison_transform is not None
		if poisoned:
			img = self.poison_transform(img)
		# Return a flag so collate can remap label for train loader if needed
		return img, label, poisoned


def make_datasets(
	scenario: str,
	data_root: str,
	poison_rate: float,
	target_label: int,
	subset_train: Optional[int] = None,
	subset_test: Optional[int] = None,
) -> Tuple[Dataset, Dataset, Dataset, Callable[[torch.Tensor], torch.Tensor]]:
	"""
	Returns: (train_ds, test_clean_ds, test_triggered_ds, normalize)
	The datasets yield tensors in [0,1] (ToTensor applied), Normalize is returned separately and will be
	applied by the loaders after potential poisoning, so we can visualize pre-norm images too.
	"""
	base_tf = transforms.Compose([
		transforms.Resize((32, 32)),
		transforms.ToTensor(),
		# Do NOT normalize here; we want to attack in [0,1] space first.
	])

	train_base = datasets.MNIST(data_root, train=True, download=True, transform=base_tf)
	test_base = datasets.MNIST(data_root, train=False, download=True, transform=base_tf)

	if subset_train is not None:
		train_base = Subset(train_base, list(range(min(subset_train, len(train_base)))))
	if subset_test is not None:
		test_base = Subset(test_base, list(range(min(subset_test, len(test_base)))))

	normalize = transforms.Normalize(*MNIST_NORM)

	if scenario == "clean":
		train = PoisonedDatasetWrapper(train_base, poison_transform=None, poison_rate=0.0)
		test_clean = PoisonedDatasetWrapper(test_base)
		test_trig = PoisonedDatasetWrapper(test_base)  # same as clean; used for consistent interface
		return train, test_clean, test_trig, normalize

	elif scenario == "classical_bd":
		trigger = ClassicalPatchTrigger(size=4, value=1.0, location="br")

		def poison_t(img):
			return trigger(img)

		# For train: apply trigger with poison_rate and we must also change label to target afterward in loader
		train = PoisonedDatasetWrapper(train_base, poison_transform=poison_t, poison_rate=poison_rate)
		test_clean = PoisonedDatasetWrapper(test_base)
		test_trig = PoisonedDatasetWrapper(test_base, poison_transform=poison_t, apply_to_all=True)
		return train, test_clean, test_trig, normalize

	elif scenario == "quantum_bd":
		qattack = build_quantum_patch_attack(n_qubits=4, depth=3, seed=SEED)

		def poison_t(img):
			return qattack(img, size=4, location="br", mix=1.0)

		train = PoisonedDatasetWrapper(train_base, poison_transform=poison_t, poison_rate=poison_rate)
		test_clean = PoisonedDatasetWrapper(test_base)
		test_trig = PoisonedDatasetWrapper(test_base, poison_transform=poison_t, apply_to_all=True)
		return train, test_clean, test_trig, normalize

	else:
		raise ValueError(f"Unknown scenario: {scenario}")


# -------------------- Training / Eval --------------------
@dataclass
class TrainResult:
	train_epoch_times: List[float]
	test_clean_acc: float
	test_clean_ms_per_img: float
	test_triggered_acc: float
	asr: float  # attack success rate on triggered set (pred == target_label)


def make_loaders(
	train_ds: Dataset,
	test_clean_ds: Dataset,
	test_trig_ds: Dataset,
	normalize: Callable[[torch.Tensor], torch.Tensor],
	batch_size: int,
	num_workers: int = 0,
	poison_labels: bool = False,
	target_label: int = 0,
):
	# Wrap datasets with post-normalization mapping; For train, optionally override labels when poisoned
	def collate_train(batch):
		# batch items are (img, label, poisoned)
		imgs, labels, flags = zip(*batch)
		imgs = [normalize(x) for x in imgs]
		imgs = torch.stack(imgs, dim=0)
		labels = list(labels)
		if poison_labels:
			labels = [target_label if f else y for f, y in zip(flags, labels)]
		return imgs, torch.tensor(labels, dtype=torch.long)

	def collate_test(batch):
		if len(batch[0]) == 3:
			imgs, labels, _flags = zip(*batch)
		else:
			imgs, labels = zip(*batch)
		imgs = torch.stack([normalize(x) for x in imgs], dim=0)
		return imgs, torch.tensor(labels, dtype=torch.long)

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_train)
	test_clean_loader = DataLoader(test_clean_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_test)
	test_trig_loader = DataLoader(test_trig_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_test)
	return train_loader, test_clean_loader, test_trig_loader


def train_one_epoch(model: nn.Module, opt: torch.optim.Optimizer, loader: DataLoader, device: torch.device) -> float:
	model.train()
	ce = nn.CrossEntropyLoss()
	t0 = time.perf_counter()
	for xb, yb in loader:
		xb = xb.to(device)
		yb = yb.to(device)
		opt.zero_grad(set_to_none=True)
		logits = model(xb)
		loss = ce(logits, yb)
		loss.backward()
		opt.step()
	t1 = time.perf_counter()
	return t1 - t0


@torch.inference_mode()
def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
	model.eval()
	correct, total = 0, 0
	t0 = time.perf_counter()
	for xb, yb in loader:
		xb = xb.to(device)
		yb = yb.to(device)
		logits = model(xb)
		pred = logits.argmax(dim=1)
		correct += (pred == yb).sum().item()
		total += yb.size(0)
	t1 = time.perf_counter()
	acc = correct / total if total else 0.0
	ms_per_img = (t1 - t0) * 1000.0 / max(total, 1)
	return acc, ms_per_img


@torch.inference_mode()
def attack_success_rate(model: nn.Module, loader: DataLoader, device: torch.device, target_label: int) -> float:
	model.eval()
	succ, total = 0, 0
	target = int(target_label)
	for xb, _ in loader:
		xb = xb.to(device)
		logits = model(xb)
		pred = logits.argmax(dim=1)
		succ += (pred == target).sum().item()
		total += xb.size(0)
	return succ / total if total else 0.0


# -------------------- Visualization --------------------
def register_activation_hooks(model: nn.Module, model_name: str, out_dir: str):
	"""Attach forward hooks to save activations for the first batch passed through the model."""
	saved = {"done": False}

	def save_act(name, act: torch.Tensor):
		# act shape could be [B,C,H,W] or [B,D]
		if saved["done"]:
			return
		path = os.path.join(out_dir, f"{model_name}_{name}.png")
		if act.dim() == 4:
			# Save first sample's channels as a grid of grayscale images
			c_first = act[0]            # [C,H,W]
			# Optional: limit channels to avoid huge grids
			max_ch = min(c_first.size(0), 32)
			c_first = c_first[:max_ch]
			x = c_first.unsqueeze(1).cpu()  # [C,1,H,W] -> NCHW with N=C
			nrow = int(math.ceil(x.size(0) ** 0.5))
			save_tensor_image(x, path, nrow=nrow)
		elif act.dim() == 2:
			save_bar(act[0].cpu(), path, title=f"{model_name}:{name}")
		else:
			save_bar(act.flatten().cpu(), path, title=f"{model_name}:{name}")

	handles = []
	for name, module in model.named_modules():
		if isinstance(module, (nn.Conv2d, nn.Linear)) or type(module).__name__ == "TorchLayer":
			handles.append(module.register_forward_hook(lambda m, inp, out, n=name: save_act(n, out)))
	return handles, saved


def visualize_pass(model: nn.Module, x: torch.Tensor, model_name: str, out_dir: str):
	ensure_dir(out_dir)
	# Save input
	save_tensor_image(x[0], os.path.join(out_dir, f"{model_name}_input.png"))
	# Register hooks and run a forward; hooks save activations from first batch
	handles, saved = register_activation_hooks(model, model_name, out_dir)
	with torch.no_grad():
		logits = model(x)
		pred = logits.argmax(dim=1).item()
		save_bar(logits[0].cpu(), os.path.join(out_dir, f"{model_name}_logits.png"), title=f"logits pred={pred}")
	for h in handles:
		h.remove()


def capture_flow_images(model: nn.Module, x: torch.Tensor) -> List[Tuple[str, np.ndarray]]:
	"""Run a single forward pass and return a list of (label, image-array) for panel composition.
	For conv layers: grid of channels; for linear/TorchLayer: bar plot. Also includes input and logits.
	"""
	stages: List[Tuple[str, np.ndarray]] = []

	# input
	stages.append(("input", tensor_to_grid_image(x[0], nrow=1, normalize=True)))

	def hook_fn(name):
		def _fn(m, inp, out):
			try:
				if out.dim() == 4:
					c_first = out[0]
					max_ch = min(c_first.size(0), 16)
					img = tensor_to_grid_image(c_first[:max_ch].unsqueeze(1), nrow=int(math.ceil(max_ch ** 0.5)))
				elif out.dim() == 2:
					img = vector_to_bar_image(out[0], title=name)
				else:
					img = vector_to_bar_image(out.flatten()[0:64], title=name)
				stages.append((name, img))
			except Exception:
				pass
		return _fn

	handles = []
	for name, module in model.named_modules():
		if isinstance(module, (nn.Conv2d, nn.Linear)) or type(module).__name__ == "TorchLayer":
			handles.append(module.register_forward_hook(hook_fn(name)))

	with torch.no_grad():
		logits = model(x)
	for h in handles:
		h.remove()
	pred = int(logits.argmax(dim=1).item())
	stages.append(("logits", vector_to_bar_image(logits[0], title=f"logits (pred={pred})")))
	return stages


def compose_three_row_panel(
	model: nn.Module,
	make_variant: Callable[[str], torch.Tensor],
	row_labels: Tuple[str, str, str],
	layer_labels: Dict[str, str],
	out_path: str,
):
	"""Create a panel with three rows (clean, classical_bd, quantum_bd) showing flow across layers.
	- model: the model to run
	- make_variant(kind): returns a single-sample batch tensor for kind in {"clean","classical_bd","quantum_bd"}
	- row_labels: e.g., ("Clean","Classical BD","Quantum BD")
	- layer_labels: optional mapping from internal module names to nicer arrow labels
	"""
	rows = []
	kinds = ["clean", "classical_bd", "quantum_bd"]
	for kind in kinds:
		x = make_variant(kind)
		stages = capture_flow_images(model, x)
		rows.append((kind, stages))

	# Compute max stages length to align columns
	max_len = max(len(st) for _, st in rows)

	cell_h = 120
	cell_w = 120
	arrow_gap = 30
	left_margin = 10
	top_margin = 20
	label_h = 20
	nrows = 3
	ncols = max_len
	width = left_margin * 2 + ncols * cell_w + (ncols - 1) * arrow_gap
	height = top_margin * 2 + nrows * (cell_h + label_h + 10)

	fig = plt.figure(figsize=(width / 96, height / 96), dpi=96)
	ax = plt.axes([0, 0, 1, 1])
	ax.set_xlim(0, width)
	ax.set_ylim(0, height)
	ax.axis('off')

	def draw_cell(img: np.ndarray, x0: int, y0: int):
		h, w = img.shape[:2]
		scale = min(cell_w / w, cell_h / h)
		nh, nw = int(h * scale), int(w * scale)
		y = y0 + (cell_h - nh) // 2
		x = x0 + (cell_w - nw) // 2
		ax.imshow(img, extent=(x, x + nw, height - (y + nh), height - y))

	def draw_arrow(x0: int, y0: int, x1: int, y1: int, label: str = ""):
		ax.annotate(
			"",
			xy=(x1, height - y1), xytext=(x0, height - y0),
			arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
		)
		if label:
			tx = (x0 + x1) / 2
			ty = (y0 + y1) / 2 - 5
			ax.text(tx, height - ty, label, ha="center", va="bottom", fontsize=7)

	for r, (kind, stages) in enumerate(rows):
		ybase = top_margin + r * (cell_h + label_h + 10)
		# Row label
		ax.text(5, height - (ybase - 5), row_labels[r], fontsize=9, va="bottom")
		# Draw stage images and arrows
		for c in range(len(stages)):
			name, img = stages[c]
			x0 = left_margin + c * (cell_w + arrow_gap)
			draw_cell(img, x0, ybase)
			# arrow to next
			if c < len(stages) - 1:
				x_mid_r = x0 + cell_w
				x_next_l = x0 + cell_w + arrow_gap
				lbl = layer_labels.get(name, name) if isinstance(layer_labels, dict) else name
				draw_arrow(x_mid_r + 2, ybase + cell_h // 2, x_next_l - 2, ybase + cell_h // 2, lbl)

	ensure_dir(os.path.dirname(out_path))
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


# -------------------- Orchestration --------------------
def run_experiment(
	scenario: str,
	epochs: int,
	batch_size: int,
	data_root: str,
	out_root: str,
	poison_rate: float,
	target_label: int,
	subset_train: Optional[int],
	subset_test: Optional[int],
	device: torch.device,
):
	print(f"\n=== Scenario: {scenario} ===")
	train_ds, test_clean_ds, test_trig_ds, normalize = make_datasets(
		scenario, data_root, poison_rate, target_label, subset_train, subset_test
	)

	# Whether to rewrite labels to target for poisoned train samples
	poison_labels = scenario in {"classical_bd", "quantum_bd"} and poison_rate > 0
	train_loader, test_clean_loader, test_trig_loader = make_loaders(
		train_ds, test_clean_ds, test_trig_ds, normalize, batch_size,
		num_workers=0, poison_labels=poison_labels, target_label=target_label
	)

	# Instantiate models
	models: Dict[str, nn.Module] = {
		"classical": ClassicalNet(),
		"hybrid": HybridQNet(n_qubits=NQ_HYBRID, depth=DEPTH_HYBRID),
		"pure_quantum": PureQNet(n_qubits=NQ_PURE, depth=DEPTH_PURE),
	}

	results: Dict[str, TrainResult] = {}
	vis_count = 10  # number of samples to visualize per model - numbers of mnist. 3 gives 0 1 and 2

	for name, model in models.items():
		model = model.to(device)
		opt = torch.optim.Adam(model.parameters(), lr=1e-3)
		epoch_times = []
		for ep in range(epochs):
			t = train_one_epoch(model, opt, train_loader, device)
			epoch_times.append(t)
			print(f"[{scenario}][{name}] epoch {ep+1}/{epochs}: {t:.2f}s")

		acc_clean, ms_clean = eval_model(model, test_clean_loader, device)
		if scenario == "clean":
			# No triggered set or ASR for clean scenario
			acc_trig, ms_trig, asr = 0.0, 0.0, 0.0
			print(
				f"[{scenario}][{name}] test clean acc={acc_clean*100:.2f}% ({ms_clean:.2f} ms/img)"
			)
		else:
			acc_trig, ms_trig = eval_model(model, test_trig_loader, device)
			asr = attack_success_rate(model, test_trig_loader, device, target_label)
			print(
				f"[{scenario}][{name}] test clean acc={acc_clean*100:.2f}% ({ms_clean:.2f} ms/img), "
				f"triggered acc={acc_trig*100:.2f}% ({ms_trig:.2f} ms/img), ASR={asr*100:.2f}%"
			)

		# Visualize a few samples from each test set (clean and triggered)
		# Grab a small batch
		out_dir_clean = os.path.join(out_root, name, "clean")
		out_dir_trig = os.path.join(out_root, name, "triggered")
		ensure_dir(out_dir_clean)
		ensure_dir(out_dir_trig)
		with torch.no_grad():
			for i, (xb, yb) in enumerate(test_clean_loader):
				xb = xb.to(device)
				visualize_pass(model, xb[0:1], f"{name}_clean_{i}", out_dir_clean)
				if i + 1 >= vis_count:
					break
			if scenario != "clean":
				for i, (xb, yb) in enumerate(test_trig_loader):
					xb = xb.to(device)
					visualize_pass(model, xb[0:1], f"{name}_trig_{i}", out_dir_trig)
					if i + 1 >= vis_count:
						break

		results[name] = TrainResult(
			train_epoch_times=epoch_times,
			test_clean_acc=acc_clean,
			test_clean_ms_per_img=ms_clean,
			test_triggered_acc=acc_trig,
			asr=asr,
		)

	# Save summary JSON
	summ_path = os.path.join(out_root, "summary.json")
	ensure_dir(os.path.dirname(summ_path))
	with open(summ_path, "w") as f:
		json.dump({k: vars(v) for k, v in results.items()}, f, indent=2)

	# Print small table
	print(f"\n=== {scenario} summary ===")
	for k, v in results.items():
		if scenario == "clean":
			print(
				f"{k:12s} | train epoch avg {np.mean(v.train_epoch_times):.2f}s | "
				f"clean {v.test_clean_acc*100:.2f}% ({v.test_clean_ms_per_img:.2f} ms/img)"
			)
		else:
			print(
				f"{k:12s} | train epoch avg {np.mean(v.train_epoch_times):.2f}s | "
				f"clean {v.test_clean_acc*100:.2f}% ({v.test_clean_ms_per_img:.2f} ms/img) | "
				f"trig {v.test_triggered_acc*100:.2f}% | ASR {v.asr*100:.2f}%"
			)

	return results, (train_ds, test_clean_ds, test_trig_ds), normalize, models


# -------------------- CLI --------------------
def main():
	import argparse

	parser = argparse.ArgumentParser(description="MNIST quantum/classical backdoor study")
	parser.add_argument("--scenario", type=str, default="clean", choices=["clean", "classical_bd", "quantum_bd", "all"], help="Which pipeline to run")
	parser.add_argument("--epochs", type=int, default=2)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--data-root", type=str, default=os.path.join("..", "data"))
	parser.add_argument("--out", type=str, default=os.path.join("runs"))
	parser.add_argument("--poison-rate", type=float, default=0.1, help="Fraction of training data to poison")
	parser.add_argument("--target-label", type=int, default=0, choices=list(range(10)))
	parser.add_argument("--subset-train", type=int, default=20000, help="Limit train set for speed (None for full)")
	parser.add_argument("--subset-test", type=int, default=5000, help="Limit test set for speed (None for full)")
	args = parser.parse_args()

	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print(f"Device: {device}")

	# Convert int to None if <= 0
	subset_train = None if args.subset_train is None or args.subset_train <= 0 else args.subset_train
	subset_test = None if args.subset_test is None or args.subset_test <= 0 else args.subset_test

	# Create timestamped run folder
	from datetime import datetime
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

	if args.scenario == "all":
		base_run_name = f"{stamp}_all_epochs={args.epochs}_poison={args.poison_rate}_target={args.target_label}_train={args.subset_train}_test={args.subset_test}"
		base_out_root = os.path.join(args.out, base_run_name)
		ensure_dir(base_out_root)

		trained_by_scenario: Dict[str, Dict[str, nn.Module]] = {}
		datasets_by_scenario: Dict[str, Tuple[Dataset, Dataset, Dataset]] = {}
		normalize_by_scenario: Dict[str, Callable] = {}

		for sc in ["clean", "classical_bd", "quantum_bd"]:
			sc_out = os.path.join(base_out_root, sc)
			ensure_dir(sc_out)
			res, ds_tuple, norm, models = run_experiment(
				scenario=sc,
				epochs=args.epochs,
				batch_size=args.batch_size,
				data_root=args.data_root,
				out_root=sc_out,
				poison_rate=args.poison_rate,
				target_label=args.target_label,
				subset_train=subset_train,
				subset_test=subset_test,
				device=device,
			)
			trained_by_scenario[sc] = models
			datasets_by_scenario[sc] = ds_tuple
			normalize_by_scenario[sc] = norm

		# Build panels using clean-trained models and clean test images with three input variants
		_, test_clean_ds, _ = datasets_by_scenario["clean"]
		_normalize = normalize_by_scenario["clean"]

		class_attack = ClassicalPatchTrigger(size=4, value=1.0, location="br")
		q_attack = build_quantum_patch_attack(n_qubits=4, depth=3, seed=SEED)

		def to_batch(img_t: torch.Tensor) -> torch.Tensor:
			return _normalize(img_t).unsqueeze(0)

		samples_by_digit: Dict[int, torch.Tensor] = {}
		for i in range(len(test_clean_ds)):
			item = test_clean_ds[i]
			if isinstance(item, tuple) and len(item) == 3:
				img, lbl, _ = item
			else:
				img, lbl = item
			if int(lbl) not in samples_by_digit:
				samples_by_digit[int(lbl)] = img
			if len(samples_by_digit) == 10:
				break

		def make_variant_from(img_clean_pre_norm: torch.Tensor, kind: str) -> torch.Tensor:
			if kind == "clean":
				return to_batch(img_clean_pre_norm.clone())
			elif kind == "classical_bd":
				return to_batch(class_attack(img_clean_pre_norm.clone()))
			elif kind == "quantum_bd":
				return to_batch(q_attack(img_clean_pre_norm.clone(), size=4, location="br", mix=1.0))
			else:
				raise ValueError(kind)

		models_for_panel: Dict[str, nn.Module] = {k: v.to(device).eval() for k, v in trained_by_scenario["clean"].items()}
		for model_name, model_inst in models_for_panel.items():
			panel_dir = os.path.join(base_out_root, "panels", model_name)
			ensure_dir(panel_dir)
			for digit in range(10):
				if digit not in samples_by_digit:
					continue
				base_img = samples_by_digit[digit]

				def mk(kind: str, _b=base_img):
					return make_variant_from(_b, kind).to(device)

				layer_labels = {}
				out_path = os.path.join(panel_dir, f"digit_{digit}.png")
				compose_three_row_panel(
					model=model_inst,
					make_variant=mk,
					row_labels=("Clean", "Classical BD", "Quantum BD"),
					layer_labels=layer_labels,
					out_path=out_path,
				)
		return
	else:
		run_name = f"{stamp}_scenario={args.scenario}_epochs={args.epochs}_poison={args.poison_rate}_target={args.target_label}_train={args.subset_train}_test={args.subset_test}"
		out_root = os.path.join(args.out, run_name)
		ensure_dir(out_root)

		results, datasets_tuple, normalize, trained_models = run_experiment(
			scenario=args.scenario,
			epochs=args.epochs,
			batch_size=args.batch_size,
			data_root=args.data_root,
			out_root=out_root,
			poison_rate=args.poison_rate,
			target_label=args.target_label,
			subset_train=subset_train,
			subset_test=subset_test,
			device=device,
		)

		# Build panels for this single-scenario run
		(_, test_clean_ds, _), _normalize = datasets_tuple, normalize
		class_attack = ClassicalPatchTrigger(size=4, value=1.0, location="br")
		q_attack = build_quantum_patch_attack(n_qubits=4, depth=3, seed=SEED)

		def to_batch(img_t: torch.Tensor) -> torch.Tensor:
			return _normalize(img_t).unsqueeze(0)

		samples_by_digit: Dict[int, torch.Tensor] = {}
		for i in range(len(test_clean_ds)):
			item = test_clean_ds[i]
			if isinstance(item, tuple) and len(item) == 3:
				img, lbl, _ = item
			else:
				img, lbl = item
			if int(lbl) not in samples_by_digit:
				samples_by_digit[int(lbl)] = img
			if len(samples_by_digit) == 10:
				break

		def make_variant_from(img_clean_pre_norm: torch.Tensor, kind: str) -> torch.Tensor:
			if kind == "clean":
				return to_batch(img_clean_pre_norm.clone())
			elif kind == "classical_bd":
				return to_batch(class_attack(img_clean_pre_norm.clone()))
			elif kind == "quantum_bd":
				return to_batch(q_attack(img_clean_pre_norm.clone(), size=4, location="br", mix=1.0))
			else:
				raise ValueError(kind)

		models_for_panel: Dict[str, nn.Module] = {k: v.to(device).eval() for k, v in trained_models.items()}
		for model_name, model_inst in models_for_panel.items():
			panel_dir = os.path.join(out_root, "panels", model_name)
			ensure_dir(panel_dir)
			for digit in range(10):
				if digit not in samples_by_digit:
					continue
				base_img = samples_by_digit[digit]

				def mk(kind: str, _b=base_img):
					return make_variant_from(_b, kind).to(device)

				layer_labels = {}
				out_path = os.path.join(panel_dir, f"digit_{digit}.png")
				compose_three_row_panel(
					model=model_inst,
					make_variant=mk,
					row_labels=("Clean", "Classical BD", "Quantum BD"),
					layer_labels=layer_labels,
					out_path=out_path,
				)


if __name__ == "__main__":
	main()

