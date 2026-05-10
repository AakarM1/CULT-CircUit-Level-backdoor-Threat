"""EXP-2: Qubit/depth ablation.
MNIST grid: n_qubits in {3,5,7,9}, depth in {2,4,6,8}.
CIFAR-10:   n_qubits in {7,9},     depth in {2,4,6,8}.
Fixed q=0.20, rho=0.9, MUD-HoG, all 4 attacks, 5 seeds, 100 rounds.
"""
from __future__ import annotations
import argparse, os
from common import (ATTACKS, SEEDS, RESULTS_ROOT, build_argv, run_one)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    out_dir = os.path.join(RESULTS_ROOT, 'exp2_qubit_depth')
    grids = {
        'mnist': {'qubits': [3, 5, 7, 9], 'depths': [2, 4, 6, 8]},
        'cifar': {'qubits': [7, 9],       'depths': [2, 4, 6, 8]},
    }
    aggregator = 'mudhog'
    q, rho = 0.20, 0.9

    for dataset, g in grids.items():
        for attack in ATTACKS:
            for nq in g['qubits']:
                for d in g['depths']:
                    for seed in SEEDS:
                        csv_path = os.path.join(
                            out_dir,
                            f'{dataset}_{attack}_q{nq}_d{d}_s{seed}.csv')
                        tag = (f'exp2|{dataset}|{attack}|nq={nq}|d={d}'
                               f'|q={q}|rho={rho}|seed={seed}')
                        argv = build_argv(
                            dataset=dataset, attack=attack,
                            aggregator=aggregator,
                            q=q, rho=rho, alpha=0.9,
                            n_qubits=nq, q_depth=d,
                            inner_epochs=1, epochs=args.epochs,
                            seed=seed, csv_path=csv_path, exp_tag=tag,
                        )
                        run_one(csv_path, argv, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
