"""EXP-6: S1-only ablation (no update crafting; --disable_s2).
4 attacks x q in {0.05,0.20,0.50} x aggregators {fedavg,mudhog,mkrum}
x MNIST x 5 seeds x 100 rounds, rho=0.9.
"""
from __future__ import annotations
import argparse, os
from common import (ATTACKS, SEEDS, RESULTS_ROOT, build_argv, run_one)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    out_dir = os.path.join(RESULTS_ROOT, 'exp6_s1only')
    qs = [0.05, 0.20, 0.50]
    aggregators = ['fedavg', 'mudhog', 'mkrum']

    for attack in ATTACKS:
        for agg in aggregators:
            for q in qs:
                for seed in SEEDS:
                    csv_path = os.path.join(
                        out_dir,
                        f'{attack}_{agg}_q{q}_s{seed}.csv')
                    tag = (f'exp6|mnist|{attack}|{agg}|q={q}|seed={seed}'
                           '|s1_only')
                    argv = build_argv(
                        dataset='mnist', attack=attack,
                        aggregator=agg,
                        q=q, rho=0.9, alpha=0.9,
                        n_qubits=None, q_depth=None,
                        inner_epochs=1, epochs=args.epochs,
                        seed=seed, csv_path=csv_path, exp_tag=tag,
                        disable_s2=True,
                    )
                    run_one(csv_path, argv, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
