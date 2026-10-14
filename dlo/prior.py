"""Diffusion prior, deterministic DDIM and RED-DiffEq regularization."""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

@dataclass
class DiffusionPrior:
    """扩散先验包装：DDPM 网络 + DDPM 噪声调度表（alphas_cumprod）。

    与 ``OpenFWIUNetWrapper``/``DDPMScheduler`` 兼容；调用 ``denoise(x_t, t)``
    会得到模型直接预测的噪声 :math:`\\epsilon_\\theta(x_t,t)`。
    """

    wrapper: torch.nn.Module
    alphas_cumprod: torch.Tensor          # (T,) DDPM 累积 alpha
    num_train_timesteps: int = 1000
    final_alpha_cumprod: float = 1.0      # DDIM 边界，t=-1 处的 ᾱ，用于反向最末步

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        """ε_θ(x_t, t)。``t`` 可为 int 或 (B,) long tensor。"""
        if isinstance(t, int):
            t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        else:
            t_tensor = t.to(device=x_t.device, dtype=torch.long)
        return self.wrapper(x_t, t_tensor).sample

    def alpha_bar(self, t: int) -> torch.Tensor:
        """ᾱ_t (scalar)；t==-1 → final_alpha_cumprod。"""
        if t < 0:
            return torch.tensor(self.final_alpha_cumprod, dtype=torch.float32)
        return self.alphas_cumprod[int(t)].float()

def _tikhonov_reg(mu: torch.Tensor) -> torch.Tensor:
    """:math:`\\|\\nabla\\mu\\|_2^2`，对相邻像素 L2 差。"""
    dx = mu[..., 1:] - mu[..., :-1]
    dy = mu[..., 1:, :] - mu[..., :-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()

def _tv_reg(mu: torch.Tensor) -> torch.Tensor:
    """:math:`\\|\\nabla\\mu\\|_1`，anisotropic TV (separate horizontal and vertical differences)。"""
    dx = (mu[..., 1:] - mu[..., :-1]).abs()
    dy = (mu[..., 1:, :] - mu[..., :-1, :]).abs()
    return dx.mean() + dy.mean()

def _red_diffeq_grad_field(
    x0_pred: torch.Tensor,
    diffusion: DiffusionPrior,
    fixed_timestep: Optional[int],
    use_time_weight: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RED-DiffEq 梯度场 :math:`\\epsilon_\\theta(x_t,t) - \\epsilon`，detach 后乘 x0_pred。

    ``x0_pred`` 已在外层主循环中由 μ + σ·η 构造（与官方 ``red_diffeq``
    ``core/inversion.py`` 行为一致）。返回 (reg_loss_for_backward, time_t)。
    """
    B = x0_pred.shape[0]
    device = x0_pred.device
    T_max = diffusion.num_train_timesteps if fixed_timestep is None else int(fixed_timestep)
    t = torch.randint(0, T_max, (B,), device=device, dtype=torch.long)

    noise = torch.randn_like(x0_pred)
    abar = diffusion.alphas_cumprod[t].to(device).float().view(B, 1, 1, 1)
    x_t = abar.sqrt() * x0_pred + (1.0 - abar).sqrt() * noise

    eps_raw = diffusion.denoise(x_t, t)

    pred_x0 = ((x_t - (1.0 - abar).sqrt() * eps_raw) / abar.sqrt()).clamp(-1.0, 1.0)
    eps_pred = (x_t - abar.sqrt() * pred_x0) / (1.0 - abar).sqrt()

    grad_field = (eps_pred - noise).detach()

    if use_time_weight:
        w_t = ((1.0 - abar) / abar).sqrt()
        grad_field = grad_field * w_t

    reg_field = grad_field * x0_pred
    reg_loss = reg_field.mean()
    return reg_loss, t

def _build_ddim_timesteps_full(num_steps: int, num_train: int) -> List[int]:
    """Rounded training-time grid used by the experiments: [667, 333, 0] for 3 steps."""
    if num_steps <= 1:
        return [num_train - 1]
    step = num_train / num_steps
    return [int(round((num_steps - 1 - i) * step)) for i in range(num_steps)]

def _ddim_sample(
    z: torch.Tensor,
    diffusion: DiffusionPrior,
    num_steps: int,
    eta: float = 0.0,
    require_grad: bool = True,
    clip_sample: bool = True,
    clip_sample_range: float = 1.0,
    use_clipped_model_output: bool = True,
) -> torch.Tensor:
    """DDIM 反向采样 z(噪声) → x_0；require_grad=False 时禁用梯度。

    与 diffusers ``DDIMScheduler.step`` 行为等价。两个关键开关（DMPlug 风格，
    默认开启）：

    * ``clip_sample`` (default ``True``)：每步对预测的 ``x_0`` clamp 到
      ``[-clip_sample_range, +clip_sample_range]``，避免越界累积。
    * ``use_clipped_model_output`` (default ``True``)：用 clipped ``x_0``
      反推 ``ε``，让 "direction pointing to x_t" 与 clipped ``x_0`` 自洽 ——
      在用 DDIM 链做潜变量优化（梯度反传穿过整个链）时尤其重要，提升稳定性
      与细节恢复。
    """
    timesteps = _build_ddim_timesteps_full(num_steps, diffusion.num_train_timesteps)
    abar = diffusion.alphas_cumprod.to(z.device).float()

    def _step():
        x = z
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            eps = diffusion.denoise(x, int(t))
            abar_t = abar[int(t)]
            abar_prev = abar[int(t_prev)] if t_prev >= 0 else torch.tensor(diffusion.final_alpha_cumprod, device=z.device, dtype=torch.float32)

            # (3) 从预测噪声解出 x̂_0
            pred_x0 = (x - (1.0 - abar_t).sqrt() * eps) / abar_t.sqrt()
            # (4) 可选 clip x̂_0
            if clip_sample:
                pred_x0 = pred_x0.clamp(-clip_sample_range, clip_sample_range)

            # (5) 方差 / sigma
            variance = ((1.0 - abar_prev) / (1.0 - abar_t) * (1.0 - abar_t / abar_prev)).clamp(min=0.0)
            sigma = eta * variance.sqrt()

            # (5') 可选：用 clipped x̂_0 反推 ε，使 direction 与 x̂_0 自洽
            eps_used = eps
            if use_clipped_model_output:
                eps_used = (x - abar_t.sqrt() * pred_x0) / (1.0 - abar_t).sqrt()

            # (6) direction pointing to x_t
            coef_dir = (1.0 - abar_prev - sigma ** 2).clamp(min=0.0).sqrt()
            # (7) x_{t-1}
            x = abar_prev.sqrt() * pred_x0 + coef_dir * eps_used
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x

    if require_grad:
        return _step()
    with torch.no_grad():
        return _step()
