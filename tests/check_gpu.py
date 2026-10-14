"""Optional real-operator checks; run on an otherwise idle CUDA GPU.

python tests/check_gpu.py --output validation/gpu_checks.json
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/gpu_checks.json")
    args = parser.parse_args()
    import numpy as np
    import torch
    from dlo import invert
    from dlo.io import load_prior
    from dlo.wave_solver import seismic_master_forward_modeling as forward
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    config = json.loads((ROOT / "configs/openfwi.json").read_text())
    weights = json.loads((ROOT / "configs/weights.json").read_text())["models"]["FlatVel-B"]
    prior = load_prior(ROOT / "checkpoints" / weights["path"], torch.device("cuda"),
                       config["diffusion"], weights["sha256"])
    with np.load(ROOT / "examples/FlatVel-B.npz") as s:
        initial = torch.from_numpy(s["velocity_init"]).cuda()
        obs = torch.from_numpy(s["seismic_obs"]).cuda()
    params = {**config["methods"]["dlo"], "n_iters": 4}
    torch.manual_seed(42)
    a = invert("dlo", obs, initial, forward, params, prior=prior)
    torch.manual_seed(42)
    b = invert("dlo", obs, initial, forward, params, prior=prior, reuse_generator=False)
    delta = float(np.max(np.abs(a["velocity_pred"] - b["velocity_pred"])))
    assert delta == 0, delta
    assert a["history"] == b["history"]

    # Direction has zero boundary support, so the min-velocity damping parameter
    # and edge extension are unchanged in the finite difference.
    v = torch.linspace(2200, 3500, 70*70, device="cuda").reshape(70,70).requires_grad_(True)
    torch.manual_seed(101)
    direction = torch.zeros_like(v)
    direction[10:-10, 10:-10] = torch.randn(50,50,device="cuda")
    output = forward(v)
    cotangent = torch.randn_like(output)
    gradient, = torch.autograd.grad((output * cotangent).sum(), v)
    analytical = (gradient * direction).sum().item()
    checks = []
    for epsilon in [0.1, 0.3, 1.0]:
        with torch.no_grad():
            plus = forward(v.detach() + epsilon*direction).clone()
            minus = forward(v.detach() - epsilon*direction).clone()
            finite_difference = ((plus-minus).double()*cotangent.double()).sum().item() / (2*epsilon)
        relative = abs(analytical-finite_difference) / max(abs(analytical), abs(finite_difference), 1e-12)
        checks.append({"epsilon_m_s": epsilon, "finite_difference": finite_difference,
                       "adjoint": analytical, "relative_error": relative})
    assert min(c["relative_error"] for c in checks) < 0.03, checks
    report = {"real_prior_cached_vs_recomputed_max_abs_m_s": delta,
              "real_prior_histories_identical": True, "adjoint_directional_checks": checks}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
