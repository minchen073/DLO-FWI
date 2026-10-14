# Sources and attribution

- OpenFWI example data are distributed under CC BY-NC-SA 4.0.
  See [examples/LICENSE.md](examples/LICENSE.md).
- The pretrained UNet uses [Hugging Face diffusers](https://github.com/huggingface/diffusers),
  distributed under Apache-2.0.
- DDIM follows Song et al., *Denoising Diffusion Implicit Models*:
  https://arxiv.org/abs/2010.02502 .
- RED-DiffEq follows Shan, Zhu, Lin and Lu, *Regularization by denoising diffusion
  models for solving inverse PDE problems with application to full waveform
  inversion*, Communications Physics (2026).
- SSIM follows Wang et al., *Image Quality Assessment: From Error Visibility to
  Structural Similarity* (2004). It is implemented in PyTorch with the window
  and padding conventions of [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim).
- PyTorch, CuPy, NumPy, SciPy and Matplotlib are external dependencies and retain
  their respective licenses.
