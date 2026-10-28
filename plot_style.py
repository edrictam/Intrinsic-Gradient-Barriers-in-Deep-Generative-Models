"""Shared presentation conventions for the manuscript's numerical figures."""
import matplotlib

RED = '#B33235'
BLUE = '#176BA0'
SHADE = '#FADEDF'
JACOBIAN_LABEL = 'Expected squared\nJacobian norm'
PARAMETERS = {'t': r'\kappa', 'pareto': r'\alpha', 'frechet': r'\zeta'}


def apply_style():
    matplotlib.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.5,
        'axes.titlesize': 9, 'axes.labelsize': 8.5,
        'legend.fontsize': 8, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True,
        'grid.alpha': .18, 'grid.linewidth': .5,
        'pdf.fonttype': 42, 'savefig.facecolor': 'white',
    })


def parameter_label(family):
    name = 'Degrees of freedom' if family == 't' else 'Shape parameter'
    return name + ' $' + PARAMETERS[family] + '$'
