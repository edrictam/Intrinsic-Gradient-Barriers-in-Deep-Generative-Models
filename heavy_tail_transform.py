#!/usr/bin/env python3
"""Recalculate and plot the paper's exact heavy-tail transformations.

Every integral is evaluated from its target and transport definitions.
The figure uses only rows computed in the current invocation.
"""
from pathlib import Path
import csv
import sys
import os
import tempfile

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'annals-intrinsic-gradients-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'annals-intrinsic-gradients-cache'))
import numpy as np
from scipy import special, stats
from heavytail_targets import StudentT
from quadrature import gl_integrate_log, gl_nodes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_style import RED, BLUE, JACOBIAN_LABEL, apply_style, parameter_label

CONFIG = {'families': ['Student', 'Pareto', 'Frechet'], 'alphas': [1, 1.25, 1.5, 1.75, 2, 2.025, 2.05, 2.1, 2.2, 2.3, 2.5, 2.75, 3, 3.5, 4, 5, 6, 8, 10], 'heavy_dimension': 2, 'asinh_scale': 'sqrt(alpha) for Student; 1 for Pareto and Frechet', 'affine_control': 'x / asinh_scale; recorded in data, not headline plots', 'quadrature_nodes_per_segment': 320, 'standardization': 'none; transformations and scale rule declared above'}
LN2PI = np.log(2 * np.pi)


def log_profile(log_small_probability):
    """log phi(Phi^-1(p)) using log p, without CDF clipping."""
    z = special.ndtri_exp(log_small_probability)
    return -.5 * z * z - .5 * LN2PI


def scale(family, alpha):
    return np.sqrt(alpha) if family == 'Student' else 1.


def log_transport(family, alpha, z):
    """Return log|T(z)| and log T'(z), for increasing exact maps."""
    z = np.asarray(z, float)
    if family == 'Student':
        u = StudentT.log_quantile_from_lsf(stats.norm.logsf(np.abs(z)), alpha)
        ld = stats.norm.logpdf(z) - StudentT.log_f_from_logy(u, alpha)
    elif family == 'Pareto':
        ls = stats.norm.logsf(z)
        u = -ls / alpha
        ld = -np.log(alpha) + stats.norm.logpdf(z) - (1 + 1/alpha) * ls
    else:
        # log(-log Phi(z)); preserve a nonzero value in the far right tail.
        lower = z < 5
        L = np.empty_like(z)
        L[lower] = np.log(-special.log_ndtr(z[lower]))
        ls = stats.norm.logsf(z[~lower])
        p = np.exp(ls)
        correction = np.log1p(p/2 + p*p/3 + p**3/4)
        L[~lower] = ls + correction
        u = -L / alpha
        ld = -np.log(alpha) - (1 + 1/alpha)*L + stats.norm.logpdf(z) - special.log_ndtr(z)
    return u, ld


def transformed_value(family, alpha, z):
    u, _ = log_transport(family, alpha, z)
    v = u - np.log(scale(family, alpha))
    # asinh(exp(v)) = v + log(1 + sqrt(1 + exp(-2v))) in the upper branch.
    ans = np.empty_like(v)
    upper = v > 20
    ans[upper] = v[upper] + np.log(2.)
    ans[~upper] = np.arcsinh(np.exp(v[~upper]))
    return ans * np.sign(z) if family == 'Student' else ans


def source_energy(family, alpha, transformed, n, extend=False):
    if not transformed and alpha <= 2:
        return np.inf
    hi = 12. if transformed else max(14., np.sqrt(100 / ((alpha-2)/(2*alpha))))
    lo = 0. if family == 'Student' else -12.
    if extend:
        hi *= 1.25
        lo *= 1.25
    cuts = sorted(set([lo, hi] + [x for x in
        [-8,-4,-2,-1,0,1,2,4,8,12,hi/4,hi/2,hi*.75] if lo < x < hi]))

    def integrand(z):
        u, ld = log_transport(family, alpha, z)
        if transformed:
            ld -= .5*np.logaddexp(2*np.log(scale(family, alpha)), 2*u)
        return 2*ld + stats.norm.logpdf(z)

    val, _ = gl_integrate_log(integrand, cuts, n)
    return val * (2 if family == 'Student' else 1)






def variance(family, a):
    if a <= 2:
        return np.inf
    if family == 'Student':
        return a/(a-2)
    if family == 'Pareto':
        return a/((a-1)**2*(a-2))
    return special.gamma(1-2/a)-special.gamma(1-1/a)**2


def heavy_experiments():
    rows = []
    for family in CONFIG['families']:
        for a in CONFIG['alphas']:
            row = {'family': family, 'alpha': a, 'asinh_scale': scale(family, a)}
            for transformed, name in [(False, 'direct'), (True, 'asinh')]:
                value = (np.inf if not transformed and a <= 2 else
                         2 * source_energy(family, a, transformed, 320))
                row[name + '_energy_2d'] = value
            raw, fixed = row['direct_energy_2d'], row['asinh_energy_2d']
            row['affine_only_energy_2d'] = raw / row['asinh_scale']**2
            row['direct_over_asinh'] = raw / fixed
            row['affine_over_asinh'] = row['affine_only_energy_2d'] / fixed
            row['target_trace_covariance'] = 2 * variance(family, a)
            rows.append(row)
        print(f"{family}: {len(CONFIG['alphas'])} tail indices", flush=True)
    return rows



def run():
    output = ROOT / 'results'
    output.mkdir(parents=True, exist_ok=True)
    rows = heavy_experiments()
    with (output / 'transformed_tails_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot(data):
    apply_style()
    (ROOT/'figures').mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(6.6,2.5),sharey=True,layout='constrained')
    for ax,family,title in zip(axes,['Student','Pareto','Frechet'],['Student-t','Pareto','Fréchet']):
        r=[x for x in data if x['family']==family]
        x=np.array([float(v['alpha']) for v in r]);direct=np.array([float(v['direct_energy_2d']) for v in r])
        fixed=np.array([float(v['asinh_energy_2d']) for v in r])
        ax.axvspan(1,2,color='#FADEDF',alpha=.85,zorder=0);ax.axvline(2,color=RED,ls=':',lw=.7)
        ax.plot(x,np.where(np.isfinite(direct),direct,np.nan),'o-',color=RED,ms=2.3,lw=1.2,label=r'Original map $T$')
        ax.plot(x,fixed,'s-',color=BLUE,ms=2.3,lw=1.2,label=r'Transformed map $S$')
        key={'Student':'t','Pareto':'pareto','Frechet':'frechet'}[family]
        ax.set_yscale('log');ax.set_xlim(1,10);ax.set_xticks([1,2,5,10]);ax.set_xlabel(parameter_label(key));ax.set_title(title)
    axes[0].set_ylabel(JACOBIAN_LABEL)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False)
    fig.savefig(ROOT/'figures/exact_transformed_tails.pdf')
    plt.close(fig)
    print('Plotted the freshly computed transformed tails.', flush=True)


def main():
    plot(run())


if __name__ == '__main__':
    main()
