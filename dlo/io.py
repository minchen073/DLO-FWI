"""Checksummed inputs and model loading."""
import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prior(path, device, schedule, expected_sha256):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}. Run scripts/download_weights.py first.")
    if sha256(path) != expected_sha256:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    from diffusers import DDPMScheduler
    from .unet import load_openfwi_checkpoint
    from .prior import DiffusionPrior
    wrapper = load_openfwi_checkpoint(path).to(device).eval().requires_grad_(False)
    scheduler = DDPMScheduler(**schedule)
    return DiffusionPrior(wrapper, scheduler.alphas_cumprod.to(device).float(),
                          schedule["num_train_timesteps"])
