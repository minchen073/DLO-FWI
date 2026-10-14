"""FWI, TV-FWI, Tikhonov-FWI, RED-DiffEq and DLO.

The reference velocity is optional and is used only for reporting metrics.
All updates operate on normalized velocities; the forward operator receives m/s.
"""
import torch
import torch.nn.functional as F
from .metrics import normalize, physical, evaluate
from .prior import _ddim_sample, _red_diffeq_grad_field, _tikhonov_reg, _tv_reg

METHODS = ("fwi", "tv_fwi", "tikhonov_fwi", "red_diffeq", "dlo")


def invert(method, seismic_obs, velocity_init, forward_fn, params, *,
           prior=None, velocity_true=None, callback=None, reuse_generator=True):
    """Run one method. Returned velocities are numpy arrays in m/s.

    DLO follows the manuscript's v-then-z order with a fixed coupling weight.
    The DDIM graph used by the v-step is reused by the following z-step.
    ``reuse_generator=False`` is a numerical verification switch.
    History includes the initial state followed by each completed update.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if velocity_init.ndim != 2 or not torch.isfinite(velocity_init).all():
        raise ValueError("velocity_init must be a finite 2D tensor in m/s")
    if not torch.isfinite(seismic_obs).all():
        raise ValueError("seismic_obs must contain only finite values")
    if seismic_obs.device != velocity_init.device:
        raise ValueError("observations and initial velocity must be on the same device")
    iterations = int(params["n_iters"])
    if iterations < 1:
        raise ValueError("n_iters must be positive")
    if method in ("red_diffeq", "dlo"):
        if prior is None:
            raise ValueError(f"{method} requires a pretrained diffusion prior")
        if any(p.requires_grad for p in prior.wrapper.parameters()) or any(m.training for m in prior.wrapper.modules()):
            raise ValueError("The prior must be frozen and in evaluation mode")
    v = normalize(velocity_init.detach()).reshape(1, 1, *velocity_init.shape).clamp(-1, 1).clone().requires_grad_(True)
    truth = normalize(velocity_true.detach()).reshape_as(v) if velocity_true is not None else None
    obs = seismic_obs.detach()
    optimizer = torch.optim.Adam([v], lr=float(params["lr_v"]))
    schedule = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min=0)
                if params.get("cosine_decay", False) else None)
    rows = []

    def record(step, **losses):
        row = {"step": step, **losses}
        if truth is not None:
            row.update(evaluate(v.detach(), truth))
        rows.append(row)
        if callback is not None:
            callback(row)

    record(0)
    z = None
    if method == "dlo":
        z = torch.randn_like(v, requires_grad=True)
        z_optimizer = torch.optim.Adam([z], lr=float(params["lr_z"]))

    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        if method == "red_diffeq":
            x = v + torch.randn_like(v) * float(params["sigma_x0"])
        else:
            x = v
        predicted = forward_fn(physical(x.squeeze(0).squeeze(0)).clamp(1500, 4500))
        if predicted.shape != obs.shape:
            raise ValueError(f"Forward output {predicted.shape} differs from observations {obs.shape}")
        loss_obs = F.l1_loss(predicted, obs)
        if method == "tikhonov_fwi":
            regularization = _tikhonov_reg(x)
        elif method == "tv_fwi":
            regularization = _tv_reg(x)
        elif method == "red_diffeq":
            regularization, _ = _red_diffeq_grad_field(x, prior, None, False)
        elif method == "dlo":
            generated = _ddim_sample(z, prior, int(params["ddim_steps"]), eta=0,
                                     require_grad=reuse_generator)
            regularization = F.mse_loss(v, generated.detach())
        else:
            regularization = v.new_zeros(())
        weight = float(params["reg_lambda"])
        total = loss_obs + weight * regularization
        if not torch.isfinite(total):
            raise FloatingPointError(f"Nonfinite objective at iteration {step}")
        total.backward()
        optimizer.step()
        with torch.no_grad():
            v.clamp_(-1, 1)
        if schedule is not None:
            schedule.step()
        losses = {"obs_loss": loss_obs.item(), "reg_loss": regularization.item(),
                  "total_loss": total.item()}
        if method == "dlo":
            z_optimizer.zero_grad(set_to_none=True)
            if not reuse_generator:
                generated = _ddim_sample(z, prior, int(params["ddim_steps"]), eta=0, require_grad=True)
            latent_loss = F.mse_loss(generated, v.detach())
            latent_loss.backward()
            z_optimizer.step()
            losses["latent_loss"] = latent_loss.item()
        record(step + 1, **losses)

    return {"velocity_pred": physical(v.detach()).squeeze().cpu().numpy(),
            "history": rows, "params": dict(params), "method": method}
