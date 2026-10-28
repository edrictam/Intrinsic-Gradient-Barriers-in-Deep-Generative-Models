"""Generate the heavy-tail comparison plots and heatmaps from nine notebooks."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = tuple(
    family + "_family" + suffix + ".ipynb"
    for family in ("t", "pareto", "frechet")
    for suffix in ("", "_logconcave", "_correlated")
)


def render(family, variant, data, comparison_only=False):
    """Preserve the latest manuscript's styling, using this worker's arrays."""
    if variant == "correlated":
        return render_correlated(family, data)
    import matplotlib.pyplot as plt
    from plot_style import (
        RED, BLUE, JACOBIAN_LABEL, PARAMETERS, apply_style, parameter_label,
    )

    destination = ROOT / "figures"
    destination.mkdir(exist_ok=True)
    plt.close("all")
    apply_style()
    suffix = "" if variant == "gaussian" else "_" + variant
    if variant == "gaussian":
        fig, ax = plt.subplots(figsize=(2.2, 2.5), layout="constrained")
    else:
        # A 2-inch source panel prints at 1.8 inches in the three-column row.
        # Keep the larger text and the original legend style inside each axes.
        fig, ax = plt.subplots(figsize=(2.0, 2.3))
        fig.subplots_adjust(left=.34, right=.965, bottom=.23, top=.98)
    x = data["parameter"]
    ax.plot(x, data["jacobian"], "o-", color=RED, ms=2.3, lw=1.2,
            label="Transport map $T$")
    ax.plot(x, data["bound"], "--", color=BLUE, lw=1.2, label="Lower bound")
    ax.set(xlabel=parameter_label(family), ylabel=JACOBIAN_LABEL, yscale="log")
    if variant == "gaussian":
        ax.legend(frameon=False, fontsize=7.2)
    else:
        ax.xaxis.label.set_fontsize(8.5)
        ax.yaxis.label.set_fontsize(8.5)
        ax.tick_params(axis="both", which="both", labelsize=8.0)
        ax.xaxis.get_offset_text().set_fontsize(8.0)
        ax.yaxis.get_offset_text().set_fontsize(8.0)
        ax.legend(loc="upper right", ncol=1, frameon=False, fontsize=7.0,
                  handlelength=1.35, handletextpad=.35, labelspacing=.3)
    names = [f"poincare_comparison_{family}{suffix}.png"]
    fig.savefig(destination / names[0], dpi=300)
    plt.close(fig)

    # The correlated heatmaps are notebook diagnostics, not manuscript assets.
    if variant != "correlated" and not comparison_only:
        fig, axes = plt.subplots(
            1, len(data["heat_parameters"]), figsize=(8.0, 1.72),
            layout="constrained",
        )
        for i, (ax, parameter, values) in enumerate(zip(
                axes, data["heat_parameters"], data["heat_values"])):
            im = ax.imshow(
                values, origin="lower", extent=[-4, 4, -4, 4],
                cmap="coolwarm", vmin=data["heat_limits"][0],
                vmax=data["heat_limits"][1], interpolation="nearest",
                aspect="equal",
            )
            ax.set_title("$" + PARAMETERS[family] + f"={parameter:g}$", fontsize=9)
            ax.set_xlabel("$z_1$")
            if i == 0:
                ax.set_ylabel("$z_2$")
            ax.set_xticks([-4, 0, 4])
            ax.set_yticks([-4, 0, 4])
            ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=list(axes), shrink=.9,
                     label=r"$\log|\det J_T(z)|$")
        names.append(f"logdet_heatmaps_{family}{suffix}.png")
        fig.savefig(destination / names[-1], dpi=300)
        plt.close(fig)
    return names



def render_correlated(family, data):
    """Figure S3: one panel for each displayed correlation, with common axes."""
    import matplotlib.pyplot as plt
    from plot_style import RED, BLUE, JACOBIAN_LABEL, apply_style, parameter_label

    destination = ROOT / "figures"
    destination.mkdir(exist_ok=True)
    plt.close("all")
    apply_style()
    names, limits = [], None
    for rho, suffix in ((.5, ""), (.9, "_rho09")):
        fig, ax = plt.subplots(figsize=(2.0, 2.3))
        fig.subplots_adjust(left=.34, right=.965, bottom=.23, top=.98)
        ax.plot(data["parameter"], data[f"jacobian_rho_{rho:g}"],
                color=RED, marker="o", ms=2.3, lw=1.2,
                label="Transport map $T$")
        ax.plot(data["parameter"], data["bound"], "--", color=BLUE,
                lw=1.2, label="Lower bound")
        ax.set(xlabel=parameter_label(family), ylabel=JACOBIAN_LABEL, yscale="log")
        ax.xaxis.label.set_fontsize(8.5)
        ax.yaxis.label.set_fontsize(8.5)
        ax.tick_params(axis="both", which="both", labelsize=8.0)
        ax.xaxis.get_offset_text().set_fontsize(8.0)
        ax.yaxis.get_offset_text().set_fontsize(8.0)
        ax.legend(loc="upper right", ncol=1, frameon=False, fontsize=7.0,
                  handlelength=1.35, handletextpad=.35, labelspacing=.3)
        if limits is None:
            limits = (ax.get_xlim(), ax.get_ylim())
        else:
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
        name = f"poincare_comparison_{family}_correlated{suffix}.png"
        fig.savefig(destination / name, dpi=300)
        plt.close(fig)
        names.append(name)
    return names


def execute_notebook(name):
    """Execute a notebook in full and plot its newly computed arrays."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    notebook = ROOT / name
    document = json.loads(notebook.read_text())
    namespace = {"__name__": "__main__", "__file__": str(notebook)}
    completed = 0
    with contextlib.redirect_stdout(io.StringIO()):
        for index, cell in enumerate(document["cells"], 1):
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), f"{name}:cell-{index}", "exec"), namespace)
                completed += 1
    family = name.split("_family")[0]
    variant = ("correlated" if "correlated" in name else
               "logconcave" if "logconcave" in name else "gaussian")
    data = {"parameter": np.asarray(namespace[
        "nu_grid" if family == "t" else "alpha_grid"])}
    if variant == "correlated":
        for rho, values in namespace["rhs_per_rho"].items():
            data[f"jacobian_rho_{rho:g}"] = np.asarray(values)
    else:
        data["jacobian"] = np.asarray(namespace["rhs_est"])
    data["bound"] = np.asarray(namespace.get("scaled_lower_bound", namespace["lhs_var_target"]))
    data["heat_parameters"] = np.asarray(namespace["nus" if family == "t" else "alphas"])
    data["heat_values"] = np.asarray(namespace["vals"])
    data["heat_limits"] = np.asarray([namespace["vmin"], namespace["vmax"]])
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError(f"{name}: nonfinite numerical result")
    (ROOT / "results").mkdir(exist_ok=True)
    np.savez_compressed(ROOT / "results" / f"heavy_tail_bounds_{notebook.stem}.npz", **data)
    render(family, variant, data)
    print(f"{name}: completed {completed} cells", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Notebook names; default: all nine")
    parser.add_argument("--worker", choices=NOTEBOOKS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if set(args.notebooks) - set(NOTEBOOKS):
        parser.error("Unknown notebook name")
    if args.worker:
        execute_notebook(args.worker)
        return
    for name in args.notebooks or NOTEBOOKS:
        with tempfile.TemporaryDirectory(prefix="heavy-tail-notebook-") as directory:
            environment = {
                **os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1",
                "MPLCONFIGDIR": str(Path(directory) / "matplotlib"),
                "ANNALS_NOTEBOOK_FIGURE_DIR": str(Path(directory) / "notebook_figures"),
            }
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                            "--worker", name], cwd=directory, env=environment, check=True)


if __name__ == "__main__":
    main()
