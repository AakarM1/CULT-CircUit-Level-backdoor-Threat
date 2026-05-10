"""Read results/ CSVs, emit LaTeX tables (EXP-4, EXP-6) and PNG plots
(EXP-1 line, EXP-2 heatmap, EXP-3 errbars, EXP-5 heatmap).

Robust to missing files: skips silently per experiment if dir is empty.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(REPO_ROOT, 'results')
TABLES_OUT = os.path.join(RESULTS, 'tables')
PLOTS_OUT = os.path.join(RESULTS, 'plots')


def _final_acc(df: pd.DataFrame) -> float:
    if 'test_accuracy' not in df or len(df) == 0:
        return float('nan')
    last10 = df.dropna(subset=['test_accuracy']).tail(10)
    return float(last10['test_accuracy'].mean()) if len(last10) else float('nan')


def _load_dir(subdir: str) -> list[tuple[str, pd.DataFrame]]:
    pattern = os.path.join(RESULTS, subdir, '*.csv')
    out = []
    for p in sorted(glob.glob(pattern)):
        try:
            df = pd.read_csv(p)
            out.append((os.path.basename(p), df))
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")
    return out


# ----------------------------- EXP-1 plot -----------------------------
def plot_exp1():
    files = _load_dir('exp1_divergence')
    if not files:
        print('[EXP-1] no data')
        return
    pat = re.compile(r'(?P<attack>\w+)_(?P<agg>\w+)_rho(?P<rho>[0-9.]+)_s(?P<seed>\d+)\.csv')
    rows = []
    for name, df in files:
        m = pat.match(name)
        if not m:
            continue
        d = m.groupdict()
        rows.append((d['attack'], float(d['rho']), int(d['seed']), df))

    by_attack = defaultdict(list)
    for attack, rho, seed, df in rows:
        by_attack[attack].append((rho, seed, df))

    os.makedirs(PLOTS_OUT, exist_ok=True)
    fig, axes = plt.subplots(2, len(by_attack), figsize=(4 * len(by_attack), 6),
                             squeeze=False)
    for col, (attack, lst) in enumerate(sorted(by_attack.items())):
        rho_groups = defaultdict(list)
        for rho, seed, df in lst:
            rho_groups[rho].append(df)
        for rho, dfs in sorted(rho_groups.items()):
            stk_w = pd.concat([d[['round', 'wasserstein_norms']] for d in dfs])
            stk_j = pd.concat([d[['round', 'jensen_shannon_norms']] for d in dfs])
            mw = stk_w.groupby('round')['wasserstein_norms'].mean()
            mj = stk_j.groupby('round')['jensen_shannon_norms'].mean()
            axes[0, col].plot(mw.index, mw.values, label=f'rho={rho}')
            axes[1, col].plot(mj.index, mj.values, label=f'rho={rho}')
        axes[0, col].set_title(f'{attack} — Wasserstein-1')
        axes[1, col].set_title(f'{attack} — JS divergence')
        axes[0, col].set_xlabel('round'); axes[1, col].set_xlabel('round')
        axes[0, col].legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(PLOTS_OUT, 'exp1_divergence.png')
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f'[EXP-1] wrote {out}')


# ----------------------------- EXP-2 heatmap --------------------------
def plot_exp2():
    files = _load_dir('exp2_qubit_depth')
    if not files:
        print('[EXP-2] no data')
        return
    pat = re.compile(r'(?P<ds>mnist|cifar)_(?P<attack>\w+)_q(?P<nq>\d+)_d(?P<d>\d+)_s(?P<seed>\d+)\.csv')
    rows = []
    for name, df in files:
        m = pat.match(name)
        if not m:
            continue
        d = m.groupdict()
        acc = _final_acc(df)
        rows.append((d['ds'], d['attack'], int(d['nq']), int(d['d']),
                     int(d['seed']), acc))
    if not rows:
        return
    df = pd.DataFrame(rows, columns=['ds', 'attack', 'nq', 'd', 'seed', 'acc'])
    os.makedirs(PLOTS_OUT, exist_ok=True)
    for ds in df['ds'].unique():
        for attack in df['attack'].unique():
            sub = df[(df.ds == ds) & (df.attack == attack)]
            if sub.empty:
                continue
            piv = sub.groupby(['nq', 'd'])['acc'].mean().unstack()
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(piv.values, aspect='auto', origin='lower', cmap='viridis')
            ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
            ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
            ax.set_xlabel('depth'); ax.set_ylabel('n_qubits')
            ax.set_title(f'EXP-2 {ds} {attack} — final acc')
            fig.colorbar(im, ax=ax)
            out = os.path.join(PLOTS_OUT, f'exp2_{ds}_{attack}.png')
            fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
            print(f'[EXP-2] wrote {out}')


# ----------------------------- EXP-3 errbars --------------------------
def plot_exp3():
    files = _load_dir('exp3_alpha')
    if not files:
        print('[EXP-3] no data')
        return
    pat = re.compile(r'(?P<ds>mnist|cifar)_(?P<attack>\w+)_(?P<agg>\w+)_alpha(?P<a>[0-9.]+)_q(?P<q>[0-9.]+)_s(?P<seed>\d+)\.csv')
    rows = []
    for name, df in files:
        m = pat.match(name)
        if not m:
            continue
        d = m.groupdict()
        rows.append((d['ds'], d['attack'], d['agg'], float(d['a']),
                     float(d['q']), int(d['seed']), _final_acc(df)))
    if not rows:
        return
    df = pd.DataFrame(rows, columns=['ds', 'attack', 'agg', 'alpha', 'q', 'seed', 'acc'])
    os.makedirs(PLOTS_OUT, exist_ok=True)
    g = df.groupby(['ds', 'attack', 'agg', 'alpha', 'q'])['acc'].agg(['mean', 'std']).reset_index()
    csv_path = os.path.join(RESULTS, 'tables', 'exp3_summary_mean_std.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    g.to_csv(csv_path, index=False)
    print(f'[EXP-3] summary -> {csv_path}')

    for ds in df['ds'].unique():
        for attack in df['attack'].unique():
            sub = g[(g.ds == ds) & (g.attack == attack)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            for agg in sub['agg'].unique():
                for q in sorted(sub['q'].unique()):
                    s = sub[(sub.agg == agg) & (sub['q'] == q)].sort_values('alpha')
                    ax.errorbar(s['alpha'], s['mean'], yerr=s['std'],
                                label=f'{agg} q={q}', marker='o', capsize=2)
            ax.set_xscale('log')
            ax.set_xlabel('alpha (Dirichlet)'); ax.set_ylabel('final acc (%)')
            ax.set_title(f'EXP-3 {ds} {attack}')
            ax.legend(fontsize=7)
            out = os.path.join(PLOTS_OUT, f'exp3_{ds}_{attack}.png')
            fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
            print(f'[EXP-3] wrote {out}')


# ----------------------------- EXP-4 LaTeX ----------------------------
def table_exp4():
    files = _load_dir('exp4_epochs')
    if not files:
        print('[EXP-4] no data')
        return
    pat = re.compile(r'(?P<ds>mnist|cifar)_(?P<attack>\w+)_E(?P<E>\d+)_q(?P<q>[0-9.]+)_s(?P<seed>\d+)\.csv')
    rows = []
    for name, df in files:
        m = pat.match(name)
        if not m:
            continue
        d = m.groupdict()
        rows.append((d['ds'], d['attack'], int(d['E']), float(d['q']),
                     int(d['seed']), _final_acc(df)))
    if not rows:
        return
    df = pd.DataFrame(rows, columns=['ds', 'attack', 'E', 'q', 'seed', 'acc'])
    g = df.groupby(['ds', 'attack', 'E', 'q'])['acc'].agg(['mean', 'std']).reset_index()
    os.makedirs(TABLES_OUT, exist_ok=True)
    g.to_csv(os.path.join(TABLES_OUT, 'exp4_summary.csv'), index=False)

    lines = [r'\begin{tabular}{llrrll}',
             r'\hline',
             r'Dataset & Attack & E & q & Acc mean & Acc std \\',
             r'\hline']
    for _, r in g.iterrows():
        lines.append(f"{r['ds']} & {r['attack']} & {r['E']} & {r['q']:.2f} "
                     f"& {r['mean']:.2f} & {r['std']:.2f} \\\\")
    lines += [r'\hline', r'\end{tabular}']
    out = os.path.join(TABLES_OUT, 'exp4_epoch_ablation.tex')
    open(out, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print(f'[EXP-4] wrote {out}')


# ----------------------------- EXP-5 heatmap --------------------------
def plot_exp5():
    files = _load_dir('exp5_sensitivity')
    if not files:
        print('[EXP-5] no data')
        return
    pat = re.compile(r'(?P<attack>\w+)_(?P<agg>\w+)_rho(?P<rho>[0-9.]+)_q(?P<q>[0-9.]+)_s(?P<seed>\d+)\.csv')
    rows = []
    for name, df in files:
        m = pat.match(name)
        if not m:
            continue
        d = m.groupdict()
        rows.append((d['attack'], d['agg'], float(d['rho']), float(d['q']),
                     int(d['seed']), _final_acc(df)))
    if not rows:
        return
    df = pd.DataFrame(rows, columns=['attack', 'agg', 'rho', 'q', 'seed', 'acc'])
    os.makedirs(PLOTS_OUT, exist_ok=True)
    for attack in df['attack'].unique():
        for agg in df['agg'].unique():
            sub = df[(df.attack == attack) & (df.agg == agg)]
            if sub.empty:
                continue
            piv = sub.groupby(['rho', 'q'])['acc'].mean().unstack()
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(piv.values, aspect='auto', origin='lower', cmap='magma')
            ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
            ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
            ax.set_xlabel('q'); ax.set_ylabel('rho')
            ax.set_title(f'EXP-5 {attack} {agg}')
            fig.colorbar(im, ax=ax)
            out = os.path.join(PLOTS_OUT, f'exp5_{attack}_{agg}.png')
            fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
            print(f'[EXP-5] wrote {out}')


# ----------------------------- EXP-6 LaTeX ----------------------------
def table_exp6():
    files = _load_dir('exp6_s1only')
    if not files:
        print('[EXP-6] no data')
        return
    pat = re.compile(r'(?P<attack>\w+)_(?P<agg>\w+)_q(?P<q>[0-9.]+)_s(?P<seed>\d+)\.csv')
    rows_s1 = []
    for name, df in files:
        m = pat.match(name)
        if not m:
            continue
        d = m.groupdict()
        rows_s1.append((d['attack'], d['agg'], float(d['q']),
                        int(d['seed']), _final_acc(df)))

    # Pull S1+S2 baseline numbers from EXP-3 results matching q,agg,attack at alpha=0.9
    files3 = _load_dir('exp3_alpha')
    pat3 = re.compile(r'(?P<ds>mnist|cifar)_(?P<attack>\w+)_(?P<agg>\w+)_alpha(?P<a>[0-9.]+)_q(?P<q>[0-9.]+)_s(?P<seed>\d+)\.csv')
    rows_s12 = []
    for name, df in files3:
        m = pat3.match(name)
        if not m:
            continue
        d = m.groupdict()
        if d['ds'] != 'mnist' or float(d['a']) != 0.9:
            continue
        rows_s12.append((d['attack'], d['agg'], float(d['q']),
                         int(d['seed']), _final_acc(df)))

    df1 = pd.DataFrame(rows_s1, columns=['attack', 'agg', 'q', 'seed', 'acc_s1'])
    df12 = pd.DataFrame(rows_s12, columns=['attack', 'agg', 'q', 'seed', 'acc_s1s2'])
    os.makedirs(TABLES_OUT, exist_ok=True)

    g1 = df1.groupby(['attack', 'agg', 'q'])['acc_s1'].agg(['mean', 'std']).reset_index()
    g12 = df12.groupby(['attack', 'agg', 'q'])['acc_s1s2'].agg(['mean', 'std']).reset_index() if not df12.empty else pd.DataFrame()
    if not g12.empty:
        merged = g1.merge(g12, on=['attack', 'agg', 'q'], suffixes=('_s1', '_s1s2'),
                          how='outer')
    else:
        merged = g1.copy()
        merged['mean_s1s2'] = float('nan'); merged['std_s1s2'] = float('nan')
        merged = merged.rename(columns={'mean': 'mean_s1', 'std': 'std_s1'})

    merged.to_csv(os.path.join(TABLES_OUT, 'exp6_summary.csv'), index=False)

    lines = [r'\begin{tabular}{lllrrrr}',
             r'\hline',
             r'Attack & Aggregator & q & S1 only mean & S1 only std & S1+S2 mean & S1+S2 std \\',
             r'\hline']
    for _, r in merged.iterrows():
        lines.append(
            f"{r['attack']} & {r['agg']} & {r['q']:.2f} "
            f"& {r.get('mean_s1', float('nan')):.2f} & {r.get('std_s1', float('nan')):.2f} "
            f"& {r.get('mean_s1s2', float('nan')):.2f} & {r.get('std_s1s2', float('nan')):.2f} \\\\")
    lines += [r'\hline', r'\end{tabular}']
    out = os.path.join(TABLES_OUT, 'exp6_s1_vs_s1s2.tex')
    open(out, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print(f'[EXP-6] wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=str, default=None,
                    help='subset: comma list of {1,2,3,4,5,6}')
    args = ap.parse_args()
    pick = set(args.only.split(',')) if args.only else {'1', '2', '3', '4', '5', '6'}
    if '1' in pick: plot_exp1()
    if '2' in pick: plot_exp2()
    if '3' in pick: plot_exp3()
    if '4' in pick: table_exp4()
    if '5' in pick: plot_exp5()
    if '6' in pick: table_exp6()


if __name__ == '__main__':
    main()
