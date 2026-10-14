"""Velocity normalization and SSIM with the experiment's window convention."""
import math
import torch
import torch.nn.functional as F


def normalize(velocity):
    return (velocity - 3000.0) / 1500.0


def physical(velocity):
    return velocity * 1500.0 + 3000.0


@torch.no_grad()
def evaluate(prediction, reference):
    """Inputs are normalized velocities. MAE/RMSE use unclipped inputs.

    SSIM uses clipped [0,1] inputs, an 11x11 Gaussian (sigma=1.5),
    zero padding of 5, population moments, and K1=0.01, K2=0.03.
    The SSIM score itself is not restricted to [0,1].
    """
    difference = prediction.float() - reference.float()
    a = (prediction.float().reshape(1, 1, *prediction.shape[-2:]).clamp(-1, 1) + 1) / 2
    b = (reference.float().reshape_as(a).clamp(-1, 1) + 1) / 2
    kernel = torch.tensor([math.exp(-(i - 5)**2 / (2 * 1.5**2)) for i in range(11)],
                          dtype=torch.float32, device=a.device)
    kernel /= kernel.sum()
    window = (kernel[:, None] @ kernel[None, :]).reshape(1, 1, 11, 11)
    average = lambda x: F.conv2d(x, window, padding=5)
    ma, mb = average(a), average(b)
    va, vb = average(a*a) - ma*ma, average(b*b) - mb*mb
    cov = average(a*b) - ma*mb
    score = ((2*ma*mb + 0.01**2) * (2*cov + 0.03**2)
             / ((ma*ma + mb*mb + 0.01**2) * (va + vb + 0.03**2))).mean()
    return {"mae": difference.abs().mean().item(),
            "rmse": difference.square().mean().sqrt().item(), "ssim": score.item()}
