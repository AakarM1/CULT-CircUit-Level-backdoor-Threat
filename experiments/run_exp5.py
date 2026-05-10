"""EXP-5: rho x q x aggregator sensitivity.
rho in {0.3,0.5,0.7,0.9} x q in {0.05,0.20,0.50}
x aggregators {mudhog,mkrum,foolsgold} x 4 attacks x MNIST x 5 seeds.
"""
from __future__ import annotations
import argparse, os
from common import (ATTACKS, SEEDS, RESULTS_ROOT, build_argv, run_one)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    out_dir = os.path.join(RESULTS_ROOT, 'exp5_sensitivity')
    rhos = [0.3, 0.5, 0.7, 0.9]
    qs = [0.05, 0.20, 0.50]
    aggregators = ['mudhog', 'mkrum', 'foolsgold']

    for attack in ATTACKS:
        for agg in aggregators:
            for rho in rhos:
                for q in qs:
                    for seed in SEEDS:
                        csv_path = os.path.join(
                            out_dir,
                            f'{attack}_{agg}_rho{rho}_q{q}_s{seed}.csv')
                        tag = (f'exp5|mnist|{attack}|{agg}|rho={rho}'
                               f'|q={q}|seed={seed}')
                        argv = build_argv(
                            dataset='mnist', attack=attack,
                            aggregator=agg,
                            q=q, rho=rho, alpha=0.9,
                            n_qubits=None, q_depth=None,
                            inner_epochs=1, epochs=args.epochs,
                            seed=seed, csv_path=csv_path, exp_tag=tag,
                        )
                        run_one(csv_path, argv, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
