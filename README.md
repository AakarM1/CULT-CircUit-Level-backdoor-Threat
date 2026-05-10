# 𝘊𝘢𝘯 𝘘𝘶𝘢𝘯𝘵𝘶𝘮 𝘍𝘦𝘥𝘦𝘳𝘢𝘵𝘦𝘥 𝘓𝘦𝘢𝘳𝘯𝘪𝘯𝘨 𝘞𝘪𝘵𝘩𝘴𝘵𝘢𝘯𝘥 𝘊𝘪𝘳𝘤𝘶𝘪𝘵-𝘓𝘦𝘷𝘦𝘭 𝘉𝘢𝘤𝘬𝘥𝘰𝘰𝘳𝘴?

A research framework for studying **quantum-circuit-level backdoor attacks** in
federated learning, together with classical and quantum-aware defenses
(FedAvg, Krum, Multi-Krum, Median, Geometric Median, FoolsGold, MUD-HoG,
FLGuardian).

The codebase is built on **PyTorch** + **PennyLane** and supports MNIST,
Fashion-MNIST, CIFAR-10, CIFAR-100, IMDb, and Tiny-ImageNet under IID,
by-label, and Dirichlet non-IID partitioning.

---

## Repository layout

```
.
├── parser.py                 # CLI entry point; parses args and calls _main.main
├── _main.py                  # training-loop driver
├── server.py                 # aggregation rules + per-round metrics
├── clients.py                # honest client
├── clients_attackers.py      # classical + quantum-trojan attackers
├── dataloader.py             # IID / by-label / Dirichlet partitioners
├── tasks/                    # dataset + model definitions (classical & quantum)
├── utils/                    # helpers, FLGuardian defense, backdoor utils
├── experiments/              # EXP-1..6 driver scripts (this work)
│   ├── common.py
│   ├── run_exp1.py
│   ├── run_exp2.py
│   ├── run_exp3.py
│   ├── run_exp4.py
│   ├── run_exp5.py
│   └── run_exp6.py
├── run_all_experiments.sh    # orchestrates all six experiments
├── generate_tables.py        # produces LaTeX tables + PNG plots from results/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

GPU users: see the bottom of `requirements.txt` for CUDA wheels.

---

## Quick start

A minimal classical FedAvg run on MNIST:

```bash
python parser.py \
    --dataset mnist --model_type classical --AR fedavg \
    --loader_type iid --num_clients 10 --epochs 10
```

A quantum-model FL run on MNIST under MUD-HoG, with two Grover-trojan
attackers (q = 0.20, ρ = 0.9):

```bash
python parser.py \
    --dataset mnist --model_type quantum --AR mudhog \
    --loader_type dirichlet --alpha 0.9 \
    --num_clients 10 --epochs 100 --inner_epochs 1 \
    --n_attacker_grover 2 --poison_frac 0.9 \
    --attacks quantum_grover --seed 1
```

---

## Threat model

Each malicious client implements a two-surface attack:

* **Surface S1 - Quantum circuit poisoning.** During local training, the client
  swaps in a trojan QNode (`q_layer_grover`, `q_layer_noise`, `q_layer_bitflip`,
  or `q_layer_signflip`) on a fraction `--poison_frac` (ρ) of batches.
* **Surface S2 - Update crafting.** Before transmitting, the raw delta is
  re-shaped to mimic the honest profile: nearest-neighbour blending,
  null-space projection, norm matching, Gaussian camouflage, and sparsification.

The four attack types correspond to four `--n_attacker_<type>` flags:

| Flag                          | Attack                                  |
| ----------------------------- | --------------------------------------- |
| `--n_attacker_grover`         | Grover phase-kickback trojan            |
| `--n_attacker_noise`          | Hadamard-test triggered noise trojan    |
| `--n_attacker_bitflip`        | QFT period-finding bit-flip trojan      |
| `--n_attacker_signflip`       | Phase-estimation sign-flip trojan       |

Pass `--disable_s2` to ablate the update-crafting layer (S1-only mode).

---

## Key CLI options

| Argument             | Default | Notes                                         |
| -------------------- | ------- | --------------------------------------------- |
| `--AR`               | fedavg  | aggregation rule                              |
| `--dataset`          | mnist   | mnist / cifar / cifar100 / fashion_mnist / …  |
| `--model_type`       | classical | `classical` or `quantum`                    |
| `--loader_type`      | iid     | iid / byLabel / dirichlet                     |
| `--alpha`            | 0.9     | Dirichlet concentration (non-IID)             |
| `--num_clients`      | 10      | total federated clients                       |
| `--epochs`           | 10      | global rounds                                 |
| `--inner_epochs`     | 1       | local epochs per round (E)                    |
| `--poison_frac`      | 0.80    | per-batch poisoning fraction (ρ)              |
| `--n_qubits`         | dataset-default | override quantum width             |
| `--q_depth`          | dataset-default | override entangling depth          |
| `--disable_s2`       | off     | skip update-crafting (S1 only)                |
| `--csv_log_path`     | off     | write per-round divergence CSV                |
| `--log_norm_cosine`  | off     | log per-client L2 / cosine each round         |
| `--exp_tag`          | -       | tag string written into CSV rows              |
| `--seed`             | 1       | seeds Python `random`, NumPy, and PyTorch     |

Defaults match the standard configuration used across the experiments below
(ρ = 0.9, q = 0.20, α = 0.9, E = 1).

---

## Experiments

All six experiments live in `experiments/` and write results under
`results/<exp_dir>/`. Each driver enumerates its grid, builds the appropriate
`parser.py` invocation, and **skips runs whose CSV already exists**, so they
are safely resumable. Pass `--dry-run` to print the planned commands without
launching them, and `--epochs N` to override the default 100 global rounds.

### EXP-1 - Gradient divergence logging
Logs per-client update norms ‖Δθ‖₂ and cosine alignment with the mean benign
update each round, separately for benign and malicious clients, then computes:

* **Wasserstein-1** distance between benign and malicious norm distributions
* **Jensen–Shannon** divergence between the same distributions
* **Mean ‖z_attack − z_clean‖₂** (quantum-measurement deviation) per round

**Grid.** 4 attacks × ρ ∈ {0.3, 0.5, 0.7, 0.9} × 5 seeds.
Fixed: q = 0.20, MUD-HoG, MNIST, 100 rounds.
**Output:** `results/exp1_divergence/{attack}_{aggregator}_rho{rho}_s{seed}.csv`

### EXP-2 - Qubit / depth ablation
Sweeps the quantum encoder width (`--n_qubits`) and entangling depth
(`--q_depth`).

**Grid.** MNIST: n_qubits ∈ {3, 5, 7, 9}, depth ∈ {2, 4, 6, 8}.
CIFAR-10: n_qubits ∈ {7, 9}, depth ∈ {2, 4, 6, 8}.
Fixed: q = 0.20, ρ = 0.9, MUD-HoG, all 4 attacks, 5 seeds, 100 rounds.
**Output:** `results/exp2_qubit_depth/{dataset}_{attack}_q{nq}_d{depth}_s{seed}.csv`

### EXP-3 - Non-IID α ablation
Varies the Dirichlet concentration α together with the malicious fraction q
across three aggregators.

**Grid.** α ∈ {0.1, 0.5, 0.9, 1.0, 5.0} × q ∈ {0.0, 0.05, 0.20, 0.50}
× aggregators {FedAvg, MUD-HoG, MKrum} × 4 attacks × 5 seeds × {MNIST, CIFAR-10}.
**Output:** `results/exp3_alpha/{dataset}_{attack}_{agg}_alpha{a}_q{q}_s{seed}.csv`
A summary CSV with mean ± std across seeds is written by `generate_tables.py`.

### EXP-4 - Local-epoch (E) ablation  *(priority)*
Tests how the number of local epochs interacts with the poisoning rate under
the strongest defense.

**Grid.** Aggregator MUD-HoG, E ∈ {1, 3, 5}, q ∈ {0.05, 0.20}, 4 attacks,
{MNIST, CIFAR-10}, 5 seeds, ρ = 0.9, 100 rounds.
**Output:** `results/exp4_epochs/{dataset}_{attack}_E{E}_q{q}_s{seed}.csv`

### EXP-5 - ρ × q × aggregator sensitivity
Maps how attacker capacity (q), poisoning intensity (ρ), and aggregator
robustness interact.

**Grid.** ρ ∈ {0.3, 0.5, 0.7, 0.9} × q ∈ {0.05, 0.20, 0.50}
× aggregators {MUD-HoG, MKrum, FoolsGold} × 4 attacks × MNIST × 5 seeds.
**Output:** `results/exp5_sensitivity/{attack}_{agg}_rho{rho}_q{q}_s{seed}.csv`

### EXP-6 - S1-only ablation
Disables Surface S2 (update crafting) so only the circuit-level poisoning is
transmitted raw. Demonstrates the contribution of update crafting to evasion.

**Grid.** 4 attacks × q ∈ {0.05, 0.20, 0.50}
× aggregators {FedAvg, MUD-HoG, MKrum} × MNIST × 5 seeds × 100 rounds, ρ = 0.9.
Adds the `--disable_s2` flag to every run.
**Output:** `results/exp6_s1only/{attack}_{agg}_q{q}_s{seed}.csv`

---

## Running the full suite

### Linux / macOS (or Git Bash / WSL on Windows)

```bash
chmod +x run_all_experiments.sh

./run_all_experiments.sh                      # all six, sequential
./run_all_experiments.sh --parallel           # background each experiment
./run_all_experiments.sh --skip-exp2 --skip-exp3   # subset
./run_all_experiments.sh --dry-run            # print plan only
./run_all_experiments.sh --epochs 50          # override global rounds
```

Skip flags: `--skip-exp1` … `--skip-exp6`.
All other flags after `--parallel` / `--skip-*` are forwarded to every driver.

### Windows (native PowerShell)

```powershell
python experiments\run_exp4.py        # priority
python experiments\run_exp1.py
python experiments\run_exp2.py
python experiments\run_exp3.py
python experiments\run_exp5.py
python experiments\run_exp6.py
```

### Tables and plots

```bash
python generate_tables.py             # all
python generate_tables.py --only 4,6  # subset
```

Outputs:

* `results/tables/exp4_epoch_ablation.tex`, `exp4_summary.csv`
* `results/tables/exp6_s1_vs_s1s2.tex`, `exp6_summary.csv`,
* `results/tables/exp3_summary_mean_std.csv`
* `results/plots/exp1_divergence.png`
* `results/plots/exp2_<dataset>_<attack>.png`
* `results/plots/exp3_<dataset>_<attack>.png`
* `results/plots/exp5_<attack>_<agg>.png`

---

## CSV schema (per round)

```
round, exp_tag,
test_accuracy, test_loss, backdoor_acc,
wasserstein_norms, jensen_shannon_norms, mean_z_deviation,
mean_norm_benign, mean_norm_mal,
mean_cos_benign,  mean_cos_mal,
n_benign, n_mal,
per_client_norms_json, per_client_cos_json
```

`per_client_norms_json` and `per_client_cos_json` are JSON-encoded dicts
keyed by client ID, useful for downstream analysis.

---

## Reproducibility

* `--seed S` seeds Python `random`, NumPy, and PyTorch.
* The Dirichlet partitioner uses its own internal seed (1) for partition
  reproducibility across runs, matching the original framework.
* Each experiment driver writes one CSV per (config, seed). If the file
  already exists, the run is skipped - delete the file to force a re-run.

---

## Adding a new attack or defense

1. Add a new attacker class in `clients_attackers.py` (mirroring
   `Attacker_QuantumTrojan`) and wire it in `_main.py`.
2. Add a new aggregator in `server.py` and register it in `Server.set_AR`,
   then extend the `--AR` choices in `parser.py`.

The experiment drivers are pure Python and only depend on `parser.py`, so
new attacks/defenses are picked up automatically once the CLI flags exist.

---

## License

Standard `MIT` License, open to use with proper citation.
