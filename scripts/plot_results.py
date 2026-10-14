"""Create a four-row overview from saved inversion outputs; no GPU required."""
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("FlatVel-B", "FlatFault-B", "CurveVel-B", "CurveFault-B")
METHODS = ("fwi", "tv_fwi", "tikhonov_fwi", "red_diffeq", "dlo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/demo_results.png")
    args = parser.parse_args()
    titles = ["Ground truth", "Initial", "FWI", "TV-FWI", "Tikhonov-FWI", "RED-DiffEq", "DLO"]
    fig, axes = plt.subplots(4, 7, figsize=(17.5, 10), layout="constrained")
    for row, name in enumerate(DATASETS):
        with np.load(ROOT / "examples" / f"{name}.npz") as sample:
            panels = [sample["velocity_true"], sample["velocity_init"]]
        panels += [np.load(args.input / name / f"{m}_velocity.npy") for m in METHODS]
        for col, velocity in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(velocity, cmap="jet", vmin=1500, vmax=4500)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col])
            if col == 0:
                ax.set_ylabel(name)
    fig.colorbar(im, ax=list(axes.flat), label="Velocity (m/s)", shrink=0.75)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
