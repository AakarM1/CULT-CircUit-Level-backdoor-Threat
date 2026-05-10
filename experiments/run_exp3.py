"""EXP-3: Non-IID alpha ablation.
alpha in {0.1,0.5,0.9,1.0,5.0} x q in {0.0,0.05,0.20,0.50}
x aggregators {fedavg,mudhog,mkrum} x 4 attacks x 5 seeds x both datasets.
"""
from __future__ import annotations
import argparse, os
from common import (ATTACKS, SEEDS, RESULTS_ROOT, build_argv, run_one)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    out_dir = os.path.join(RESULTS_ROOT, 'exp3_alpha')
    alphas = [0.1, 0.5, 0.9, 1.0, 5.0]
    qs = [0.0, 0.05, 0.20, 0.50]
    aggregators = ['fedavg', 'mudhog', 'mkrum']
    datasets = ['mnist', 'cifar']

    for dataset in datasets:
        for attack in ATTACKS:
            for agg in aggregators:
                for a in alphas:
                    for q in qs:
                        for seed in SEEDS:
                            csv_path = os.path.join(
                                out_dir,
                                f'{dataset}_{attack}_{agg}_alpha{a}_q{q}_s{seed}.csv')
                            tag = (f'exp3|{dataset}|{attack}|{agg}|alpha={a}'
                                   f'|q={q}|seed={seed}')
                            argv = build_argv(
                                dataset=dataset, attack=attack,
                                aggregator=agg,
                                q=q, rho=0.9, alpha=a,
                                n_qubits=None, q_depth=None,
                                inner_epochs=1, epochs=args.epochs,
                                seed=seed, csv_path=csv_path, exp_tag=tag,
                            )
                            run_one(csv_path, argv, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
