from __future__ import annotations

import os
import time
import random
import argparse
from itertools import product
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

# Import the federated experiment runner
try:
    from testing import qfl_mnist as qfl
except Exception as e:
    raise SystemExit(f"Failed to import testing.qfl_mnist: {e}\nMake sure you run this from the 'Quantum-Federated-Learning' folder.")


def _run_job(index: int, kwargs: dict, ngpus: int) -> bool:
    """Top-level worker for ProcessPoolExecutor.

    - Optionally pins the process to a specific GPU by setting CUDA_VISIBLE_DEVICES
      before importing the experiment module.
    - Runs a single experiment via qfl.run_experiment(**kwargs).
    """
    import os as _os
    # Assign a GPU if available; simple round-robin over visible devices
    if ngpus and ngpus > 0:
        dev_id = (int(index) - 1) % int(ngpus)
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_id)
    # Import inside worker after potential CUDA env set
    from testing import qfl_mnist as _qfl
    _qfl.run_experiment(**kwargs)
    return True


def csv_list(s: str | None) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(',') if x.strip()]


def main():
    p = argparse.ArgumentParser(description="Grid runner for Federated MNIST (classical/hybrid/quantum)")
    # Global experiment knobs (defaults match your requested setup)
    p.add_argument('--clients', type=int, default=15)
    p.add_argument('--rounds', type=int, default=50)
    p.add_argument('--local-epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--subset-train-total', type=int, default=9000)
    p.add_argument('--subset-test', type=int, default=5000)
    p.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha when using non-iid partition')
    p.add_argument('--data-root', type=str, default=os.path.join('..','data'))
    p.add_argument('--out', type=str, default='runs')
    p.add_argument('--tag', type=str, default='', help='Optional tag added to output folder name')
    p.add_argument('--from-index', type=int, default=1, help='1-based start index of jobs to run (after grid composition)')
    p.add_argument('--to-index', type=int, default=0, help='1-based end index of jobs to run (inclusive). 0 = run to the end')
    p.add_argument('--workers', type=int, default=1, help='Number of parallel workers (processes). Use with care on single GPU.')

    # Grid dimensions (comma-separated lists). Leave empty to use sensible defaults.
    p.add_argument('--partitions', type=str, default='iid,dirichlet', help="Partitioning: 'iid', 'dirichlet' or both (comma-separated)")
    p.add_argument('--modes', type=str, default='no-mix,mix', help="Federation modes: 'no-mix', 'mix'")
    p.add_argument('--models', type=str, default='classical,hybrid,quantum', help="Models for no-mix runs (comma-separated)")
    p.add_argument('--server-types', type=str, default='fedavg,quantum', help="Server types: 'fedavg', 'quantum'")
    p.add_argument('--server-gradients', type=str, default='classical,quantum', help="Server gradient mode: 'classical', 'quantum', 'roundrobin'")
    p.add_argument('--server-send-modes', type=str, default='classical,quantum', help="Outgoing gradient channel to clients: 'classical' or 'quantum'")

    # Convenience flag to quickly sanity-check the pipeline
    p.add_argument('--fast', action='store_true', help='Use tiny settings (rounds=2, subset_train_total=3000, subset_test=1000) for a quick sanity run')

    args = p.parse_args()

    # Seeds for repeatability
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if args.fast:
        args.rounds = min(args.rounds, 2)
        args.subset_train_total = min(args.subset_train_total, 3000)
        args.subset_test = min(args.subset_test, 1000)

    partitions = csv_list(args.partitions) or ['iid', 'dirichlet']
    modes = csv_list(args.modes) or ['no-mix', 'mix']
    models = csv_list(args.models) or ['classical', 'hybrid', 'quantum']
    server_types = csv_list(args.server_types) or ['fedavg', 'quantum']
    server_grads = csv_list(args.server_gradients) or ['classical', 'quantum']
    send_modes = csv_list(args.server_send_modes) or ['classical', 'quantum']

    # Build a base folder per grid launch
    stamp = time.strftime('%Y%m%d_%H%M%S')
    base_out = os.path.join(args.out, f"grid_{stamp}{('_'+args.tag) if args.tag else ''}")
    os.makedirs(base_out, exist_ok=True)

    print("\n=== Grid configuration ===")
    print(f"Base out: {base_out}")
    print(f"clients={args.clients}, rounds={args.rounds}, local_epochs={args.local_epochs}, batch_size={args.batch_size}")
    print(f"subset_train_total={args.subset_train_total}, subset_test={args.subset_test}, alpha={args.alpha}")
    print(f"partitions={partitions}")
    print(f"modes={modes}")
    print(f"no-mix models={models}")
    print(f"server_types={server_types}")
    print(f"server_gradients={server_grads}")
    print(f"server_send_modes={send_modes}\n")

    # Compose jobs
    jobs = []
    for partition, server_type, grad_mode, send_mode in product(partitions, server_types, server_grads, send_modes):
        # mix job (single)
        if 'mix' in modes:
            jobs.append(dict(
                mode='mix', model_kind=None,
                partition=partition, server_type=server_type, server_gradients=grad_mode, server_send_mode=send_mode,
            ))
        # no-mix jobs for each model
        if 'no-mix' in modes:
            for mk in models:
                jobs.append(dict(
                    mode='no-mix', model_kind=mk,
                    partition=partition, server_type=server_type, server_gradients=grad_mode, server_send_mode=send_mode,
                ))

    # Prepare execution set with labels and out dirs
    prepared = []
    for i, cfg in enumerate(jobs, start=1):
        sub = cfg['mode']
        if cfg['mode'] == 'no-mix':
            sub += f"_{cfg['model_kind']}"
        sub += f"_{cfg['partition']}_{cfg['server_type']}_{cfg['server_gradients']}_{cfg['server_send_mode']}"
        out_dir = os.path.join(base_out, sub)
        label = (
            f"[{i}/{len(jobs)}] {cfg['mode']} "
            + (f"model={cfg['model_kind']} " if cfg['mode']=='no-mix' else "")
            + f"part={cfg['partition']} srv={cfg['server_type']} grad={cfg['server_gradients']} send={cfg['server_send_mode']}"
        )
        kwargs = dict(
            mode=cfg['mode'],
            model_kind=cfg['model_kind'],
            clients=args.clients,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            partition=cfg['partition'],
            alpha=args.alpha,
            server_type=cfg['server_type'],
            server_gradients=cfg['server_gradients'],
            server_send_mode=cfg['server_send_mode'],
            data_root=args.data_root,
            subset_train_total=args.subset_train_total,
            subset_test=args.subset_test,
            out_dir=out_dir,
        )
        prepared.append((i, label, kwargs))

    # Filter by index range if requested
    start_idx = max(1, int(args.from_index))
    end_idx = int(args.to_index) if int(args.to_index) > 0 else len(prepared)
    exec_list = [(i, label, kw) for (i, label, kw) in prepared if start_idx <= i <= end_idx]
    if not exec_list:
        print(f"No jobs to execute in range [{start_idx}, {end_idx}].")
        return

    print(f"\nExecuting {len(exec_list)} job(s) out of {len(jobs)} in range [{start_idx}, {end_idx}] with workers={args.workers}.")

    if args.workers and args.workers > 1:
        # Parallel execution using processes
        ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut2info = {}
            for i, label, kw in exec_list:
                print(f"\n=== Submitting {label} ===")
                fut = ex.submit(_run_job, i, kw, ngpus)
                fut2info[fut] = (i, label)
            done_ok = 0
            for fut in as_completed(fut2info.keys()):
                i, label = fut2info[fut]
                try:
                    _ = fut.result()
                    print(f"=== Completed {label}")
                    done_ok += 1
                except Exception as e:
                    print(f"[SKIP] {label} failed with error: {e}")
        print(f"\nFinished {done_ok}/{len(exec_list)} jobs. Check outputs under: {base_out}")
    else:
        # Sequential execution
        for i, label, kw in exec_list:
            print("\n=== Running", label, '===')
            try:
                qfl.run_experiment(**kw)
            except Exception as e:
                print(f"[SKIP] {label} failed with error: {e}")
        print("\nAll selected grid jobs completed. Check outputs under:", base_out)


if __name__ == '__main__':
    main()
