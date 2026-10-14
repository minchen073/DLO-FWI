"""Run one or all supplied OpenFWI samples with any of the five methods."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASETS = ("FlatVel-B", "FlatFault-B", "CurveVel-B", "CurveFault-B")
METHODS = ("fwi", "tv_fwi", "tikhonov_fwi", "red_diffeq", "dlo")


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def plot_result(truth, initial, results, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"fwi": "FWI", "tv_fwi": "TV-FWI", "tikhonov_fwi": "Tikhonov-FWI",
             "red_diffeq": "RED-DiffEq", "dlo": "DLO"}
    panels = [("Ground truth", truth), ("Initial", initial)]
    panels += [(names[name], result["velocity_pred"]) for name, result in results.items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.5 * len(panels), 3), layout="constrained")
    for ax, (name, velocity) in zip(axes, panels):
        im = ax.imshow(velocity, cmap="jet", vmin=1500, vmax=4500)
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=list(axes), label="Velocity (m/s)", shrink=0.8)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--methods", default="all", help="all or comma-separated: " + ",".join(METHODS))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/openfwi.json")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, help="Override 300 iterations (short smoke runs only)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu", default="0", help="One CUDA_VISIBLE_DEVICES GPU index or UUID")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic PyTorch operations")
    args = parser.parse_args()
    methods = list(METHODS) if args.methods == "all" else args.methods.split(",")
    if not methods or any(m not in METHODS for m in methods) or len(set(methods)) != len(methods):
        parser.error("--methods must contain unique names from " + ",".join(METHODS))
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be positive")
    if "," in args.gpu:
        parser.error("The fixed acquisition operator supports one visible GPU per process")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from dlo import invert
    from dlo.io import load_prior, sha256

    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required by the CuPy acoustic operator")
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    config = json.loads(args.config.read_text())
    # Acquisition and normalization are fixed in this release, not tunable JSON options.
    canonical = json.loads((ROOT / "configs/openfwi.json").read_text())
    for key in ("acquisition", "normalization", "initialization_sigma", "diffusion"):
        if config[key] != canonical[key]:
            parser.error(f"Custom {key} is not supported by the supplied fixed operator/data")
    seed = config["seed"] if args.seed is None else args.seed
    data_manifest = json.loads((ROOT / "examples/manifest.json").read_text())["datasets"]
    weight_manifest = json.loads((ROOT / "configs/weights.json").read_text())["models"]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    need_prior = any(m in ("red_diffeq", "dlo") for m in methods)
    # Check every input before creating outputs or starting the first inversion.
    for name in datasets:
        p = ROOT / "examples" / data_manifest[name]["file"]
        if sha256(p) != data_manifest[name]["sha256"]:
            raise ValueError(f"Sample checksum mismatch: {p}")
        if need_prior:
            p = args.weights_dir / weight_manifest[name]["path"]
            if not p.is_file() or sha256(p) != weight_manifest[name]["sha256"]:
                raise ValueError(f"Missing or invalid weights: {p}; run scripts/download_weights.py")
    out = args.output or ROOT / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    versions = {p: importlib.metadata.version(p) for p in
                ["torch", "diffusers", "cupy-cuda12x", "numpy", "scipy", "matplotlib"]}
    save_json(out / "environment.json", {"packages": versions, "python": sys.version,
              "gpu": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda,
              "deterministic": args.deterministic, "seed": seed,
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "dlo").glob("*.py"))}})
    from dlo.wave_solver import seismic_master_forward_modeling as forward
    summary = []
    for name in datasets:
        folder = out / name
        folder.mkdir()
        with np.load(ROOT / "examples" / data_manifest[name]["file"], allow_pickle=False) as sample:
            truth, initial, seismic = [sample[k].copy() for k in ("velocity_true", "velocity_init", "seismic_obs")]
        tensors = [torch.from_numpy(a).to("cuda") for a in (truth, initial, seismic)]
        prior = load_prior(args.weights_dir / weight_manifest[name]["path"], torch.device("cuda"),
                           config["diffusion"], weight_manifest[name]["sha256"]) if need_prior else None
        results = {}
        for method in methods:
            params = dict(config["methods"][method])
            if args.iterations is not None:
                params["n_iters"] = args.iterations
            torch.manual_seed(seed)
            def progress(row):
                if row["step"] % 50 == 0 or row["step"] == params["n_iters"]:
                    print(f"{name:12} {method:13} {row['step']:3}/{params['n_iters']} "
                          f"MAE={row['mae']:.5f} RMSE={row['rmse']:.5f} SSIM={row['ssim']:.5f}", flush=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = invert(method, tensors[2], tensors[1], forward, params,
                            prior=prior, velocity_true=tensors[0], callback=progress)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            results[method] = result
            np.save(folder / f"{method}_velocity.npy", result["velocity_pred"])
            save_json(folder / f"{method}_history.json", result["history"])
            save_json(folder / f"{method}_config.json", {"params": params, "seed": seed,
                "sample": data_manifest[name], "checkpoint": weight_manifest[name] if method in ("dlo", "red_diffeq") else None,
                "dlo_update_order": "v_then_z" if method == "dlo" else None,
                "time_s": elapsed, "timing_scope": "solver including metrics, progress and first-use CUDA compilation; not a paper timing benchmark"})
            summary.append({"dataset": name, "method": method,
                            **{k: result["history"][-1][k] for k in ("mae", "rmse", "ssim")},
                            "time_s": elapsed, "iterations": params["n_iters"]})
            save_json(out / "summary.json", summary)
        plot_result(truth, initial, results, folder / "comparison.png")
        del prior, tensors
        torch.cuda.empty_cache()
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
