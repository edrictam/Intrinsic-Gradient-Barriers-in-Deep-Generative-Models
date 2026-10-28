"""Recompute the paper's experiments and save their figures and numerical results."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
SCRIPTS = ("density_fitting.py", "heavy_tail_bounds.py", "multimodal_bounds.py",
           "heavy_tail_transform.py", "component_means.py")
TABLES = ("density_metrics.csv", "density_metrics.json", "multimodal_bounds.csv",
          "transformed_tails_metrics.csv", "component_means_metrics.csv")
FIGURES = {
    "density_comparison_2x2.pdf", "mm_comparison_delta.pdf",
    "exact_transformed_tails.pdf", "component_means_gradient_growth.pdf",
} | {
    f"poincare_comparison_{family}{suffix}.png"
    for family in ("t", "pareto", "frechet")
    for suffix in ("", "_logconcave", "_correlated", "_correlated_rho09")
} | {
    f"logdet_heatmaps_{family}{suffix}.png"
    for family in ("t", "pareto", "frechet")
    for suffix in ("", "_logconcave")
}


def main():
    # Only source files enter this directory; prior outputs are never inputs.
    with tempfile.TemporaryDirectory(prefix="intrinsic-gradients-") as directory:
        work = Path(directory)
        for source in (*ROOT.glob("*.py"), *ROOT.glob("*.ipynb")):
            shutil.copy2(source, work / source.name)
        for notebook in work.glob("*.ipynb"):
            document = json.loads(notebook.read_text())
            for cell in document["cells"]:
                if cell["cell_type"] == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
            notebook.write_text(json.dumps(document, indent=1) + "\n")
        environment = {
            **os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "MPLCONFIGDIR": str(work / ".cache" / "matplotlib"),
            "XDG_CACHE_HOME": str(work / ".cache"),
        }
        for script in SCRIPTS:
            print(f"Running {script}", flush=True)
            subprocess.run([sys.executable, "-B", str(work / script)],
                           cwd=work, env=environment, check=True)

        figures = {path.name for path in (work / "figures").iterdir() if path.is_file()}
        if figures != FIGURES:
            raise RuntimeError("The generated figure set differs from the paper.")
        arrays = sorted((work / "results").glob("heavy_tail_bounds_*_family*.npz"))
        if len(arrays) != 9:
            raise RuntimeError("Expected numerical results from all nine notebooks.")

        # Save the numerical results and figures from this run.
        for folder in ("figures", "results"):
            (ROOT / folder).mkdir(exist_ok=True)
        for name in sorted(FIGURES):
            shutil.copy2(work / "figures" / name, ROOT / "figures" / name)
        for source in [*(work / "results" / name for name in TABLES), *arrays]:
            shutil.copy2(source, ROOT / "results" / source.name)
    print("Completed 4 full fits, 9 notebooks (56 cells), and 22 figures.", flush=True)
    print("Outputs: figures/ and results/", flush=True)


if __name__ == "__main__":
    main()
