"""Shared helpers for EXP-1..6 experiment drivers.

Every driver enumerates configs, builds an argv list for parser.py,
checks resumability (skip if CSV already exists), and shells out
to `python parser.py ...` so the existing entry point is untouched.
"""
from __future__ import annotations

import os
import subprocess
import sys
from itertools import product

ATTACKS = ['grover', 'noise', 'bitflip', 'signflip']
N_ATTACKER_FLAG = {
    'grover':   '--n_attacker_grover',
    'noise':    '--n_attacker_noise',
    'bitflip':  '--n_attacker_bitflip',
    'signflip': '--n_attacker_signflip',
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSER_PY = os.path.join(REPO_ROOT, 'parser.py')
RESULTS_ROOT = os.path.join(REPO_ROOT, 'results')


def num_clients_for(q: float, n: int = 10) -> int:
    """Default num_clients=10 — fixed across experiments per paper."""
    return n


def n_attackers_for(q: float, n: int = 10) -> int:
    """Round q*n to nearest int, min 0."""
    k = int(round(q * n))
    if q > 0 and k == 0:
        k = 1
    return k


def csv_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def build_argv(*,
               dataset: str,
               attack: str,
               aggregator: str,
               q: float,
               rho: float,
               alpha: float,
               n_qubits: int | None,
               q_depth: int | None,
               inner_epochs: int,
               epochs: int,
               seed: int,
               csv_path: str,
               disable_s2: bool = False,
               exp_tag: str = '',
               loader_type: str = 'dirichlet',
               num_clients: int = 10) -> list[str]:
    n_atk = n_attackers_for(q, num_clients)
    argv = [
        sys.executable, PARSER_PY,
        '--dataset', dataset,
        '--AR', aggregator,
        '--model_type', 'quantum',
        '--loader_type', loader_type,
        '--alpha', str(alpha),
        '--num_clients', str(num_clients),
        '--epochs', str(epochs),
        '--inner_epochs', str(inner_epochs),
        '--seed', str(seed),
        '--poison_frac', str(rho),
        '--csv_log_path', csv_path,
        '--log_norm_cosine',
        '--exp_tag', exp_tag,
        '--attacks', f'quantum_{attack}',
        N_ATTACKER_FLAG[attack], str(n_atk),
    ]
    if n_qubits is not None:
        argv += ['--n_qubits', str(n_qubits)]
    if q_depth is not None:
        argv += ['--q_depth', str(q_depth)]
    if disable_s2:
        argv.append('--disable_s2')
    return argv


def run_one(csv_path: str, argv: list[str], dry_run: bool = False) -> int:
    if csv_exists(csv_path):
        print(f"[SKIP] {csv_path}")
        return 0
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    print(f"[RUN ] {csv_path}")
    if dry_run:
        print('       ' + ' '.join(argv))
        return 0
    return subprocess.call(argv, cwd=REPO_ROOT)


SEEDS = [1, 2, 3, 4, 5]
