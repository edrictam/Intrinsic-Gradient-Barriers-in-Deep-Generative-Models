"""Recalculate and plot all 80 cases in the paper's separation sweep.

The figure is drawn directly from rows calculated in this invocation. Saved
tables, plots, checkpoints, and training traces are never loaded.
"""
import csv
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'annals-intrinsic-gradients-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'annals-intrinsic-gradients-cache'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from isoperimetry import Iso
from targets1d import two_gauss
from plot_style import RED, BLUE, apply_style


CONFIG = {
    'sigmas': [0.1, 0.5],
    'delta_range': [0.25, 10.0],
    'delta_count': 40,
    'mode_offset_in_sigmas': 3.0,
    'mean_nodes_per_segment': 2000,
    'valley_nodes': 4000,
    'inequality_relative_tolerance': 1e-9,
}


def run():
    rows = []
    for sigma in (0.1, 0.5):
        for delta in np.linspace(0.25, 10, 40):
            target, _ = two_gauss(delta, sigma)
            left, right = -delta / 2, delta / 2
            mass = min(target.cdf(left), 1 - target.cdf(right))
            mean = target.mean_gprime_via_profile(breakpoints=[left, right])
            profile = target.profile_integral_y(left, right)
            bound = delta * Iso(mass)
            assert mean >= profile - 1e-9 * abs(profile)
            assert profile >= bound - 1e-9 * max(bound, 1)
            rows.append(dict(sigma=sigma, delta=float(delta), mean_derivative=mean,
                             valley_profile_integral=profile, lower_bound=bound))
    output = ROOT / 'results'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'multimodal_bounds.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot(data):
    apply_style()
    (ROOT / 'figures').mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True, layout='constrained')
    for ax, sigma in zip(axes, [.1, .5]):
        rows = sorted((r for r in data if float(r['sigma']) == sigma), key=lambda r:float(r['delta']))
        x = [float(r['delta']) for r in rows]
        ax.plot(x, [float(r['mean_derivative']) for r in rows], 'o-',
                color=RED, ms=2.3, lw=1.2, label='Transport map $T$')
        ax.plot(x, [float(r['lower_bound']) for r in rows], '--',
                color=BLUE, lw=1.2, label='Lower bound')
        ax.set(title=rf'$\sigma={sigma:g}$', xlabel=r'Width $\Delta$', xlim=(0,10))
    axes[0].set_ylabel('Mean transport derivative')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=2, frameon=False)
    fig.savefig(ROOT / 'figures/mm_comparison_delta.pdf')
    plt.close(fig)
    print('Plotted the two retained quantities from all 80 freshly computed cases.', flush=True)


def main():
    plot(run())


if __name__ == '__main__':
    main()
