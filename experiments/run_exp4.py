"""EXP-4: Local-epoch (E) ablation. PRIORITY.
Aggregator MUD-HoG. E in {1,3,5}. q in {0.05,0.20}. 4 attacks. Both datasets.
5 seeds. 100 global rounds. rho=0.9.
"""
from __future__ import annotations
import argparse, os
from common import (ATTACKS, SEEDS, RESULTS_ROOT, build_argv, run_one)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    out_dir = os.path.join(RESULTS_ROOT, 'exp4_epochs')
    Es = [1, 3, 5]
    qs = [0.05, 0.20]
    datasets = ['mnist', 'cifar']
    aggregator = 'mudhog'

    for dataset in datasets:
        for attack in ATTACKS:
            for E in Es:
                for q in qs:
                    for seed in SEEDS:
                        csv_path = os.path.join(
                            out_dir,
                            f'{dataset}_{attack}_E{E}_q{q}_s{seed}.csv')
                        tag = (f'exp4|{dataset}|{attack}|E={E}|q={q}'
                               f'|seed={seed}')
                        argv = build_argv(
                            dataset=dataset, attack=attack,
                            aggregator=aggregator,
                            q=q, rho=0.9, alpha=0.9,
                            n_qubits=None, q_depth=None,
                            inner_epochs=E, epochs=args.epochs,
                            seed=seed, csv_path=csv_path, exp_tag=tag,
                        )
                        run_one(csv_path, argv, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
