"""EXP-1: Gradient divergence logging.
4 attacks x rho in {0.3,0.5,0.7,0.9} x 5 seeds, q=0.20, MUD-HoG, MNIST, 100 rounds.
"""
from __future__ import annotations
import argparse, os
from common import (ATTACKS, SEEDS, RESULTS_ROOT, build_argv, run_one)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    out_dir = os.path.join(RESULTS_ROOT, 'exp1_divergence')
    rhos = [0.3, 0.5, 0.7, 0.9]
    aggregator = 'mudhog'
    q = 0.20

    for attack in ATTACKS:
        for rho in rhos:
            for seed in SEEDS:
                csv_path = os.path.join(
                    out_dir, f'{attack}_{aggregator}_rho{rho}_s{seed}.csv')
                tag = f'exp1|{attack}|{aggregator}|rho={rho}|q={q}|seed={seed}'
                argv = build_argv(
                    dataset='mnist', attack=attack, aggregator=aggregator,
                    q=q, rho=rho, alpha=0.9,
                    n_qubits=None, q_depth=None,
                    inner_epochs=1, epochs=args.epochs, seed=seed,
                    csv_path=csv_path, exp_tag=tag,
                )
                run_one(csv_path, argv, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
