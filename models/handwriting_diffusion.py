"""
Denoising Diffusion Probabilistic Model (DDPM) for Handwritten Prescription Images.
Implements iterative reverse diffusion from pure Gaussian noise to synthetic handwriting.
Features:
- Closed-form forward diffusion process: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
- Linear and Cosine noise schedules
- Sinusoidal timestep embeddings with multi-layer perceptron projection
- Compact Residual U-Net predicting added noise epsilon_theta(x_t, t) with MSE loss
- True iterative reverse sampling p_theta(x_{t-1} | x_t) returning final samples AND denoising trajectories
- Fixed-seed reproducible generation and unit smoke test
"""

from __future__ import annotations

import math
from pathlib import Path
import os
import random
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "RxHandBD-ML" / "RxHandBD-ML"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "handwriting_diffusion.pth"

DEFAULT_IMAGE_SIZE = 64
DEFAULT_TIMESTEPS = 100
DEFAULT_BASE_CHANNELS = 24


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Standard linear beta schedule for DDPM."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule as proposed by Nichol & Dhariwal."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional encoding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual convolutional block with timestep projection conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels),
        )

        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class CompactUNet(nn.Module):
    """Compact U-Net architecture predicting noise for handwriting images."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        time_dim: int = 64,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim * 2),
        )
        t_channels = time_dim * 2

        # Encoder
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, t_channels)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)

        self.down2 = ResidualBlock(base_channels * 2, base_channels * 2, t_channels)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)

        # Bottleneck
        self.mid1 = ResidualBlock(base_channels * 4, base_channels * 4, t_channels)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 4, t_channels)

        # Decoder
        self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.up2 = ResidualBlock(base_channels * 4, base_channels * 2, t_channels)

        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.up1 = ResidualBlock(base_channels * 2, base_channels, t_channels)

        self.out_norm = nn.GroupNorm(min(8, base_channels), base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep)

        # Encoder
        x0 = self.init_conv(x)
        x1 = self.down1(x0, t_emb)
        d1 = self.downsample1(x1)

        x2 = self.down2(d1, t_emb)
        d2 = self.downsample2(x2)

        # Bottleneck
        m = self.mid1(d2, t_emb)
        m = self.mid2(m, t_emb)

        # Decoder with skip connections
        u2 = self.upsample2(m)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up2(u2, t_emb)

        u1 = self.upsample1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.up1(u1, t_emb)

        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out


class GaussianDiffusion:
    """Manages forward noise addition and iterative reverse sampling."""

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = DEFAULT_TIMESTEPS,
        schedule: str = "cosine",
        device: torch.device | None = None,
    ):
        self.model = model
        self.timesteps = timesteps
        self.device = device or torch.device("cpu")

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        self.betas = betas.to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Adds noise to images according to diffusion schedule at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    @torch.inference_mode()
    def p_sample(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Single step reverse diffusion from x_t to x_{t-1} using bounded x0 prediction."""
        t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        predicted_noise = self.model(x, t_batch)

        alpha_bar_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]

        pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        if t > 0:
            alpha_bar_prev = self.alphas_cumprod_prev[t]
            c1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1.0 - alpha_bar_t)
            c2 = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
            mean = c1 * pred_x0 + c2 * x
            sigma = torch.sqrt(self.posterior_variance[t])
            noise = torch.randn_like(x)
            return mean + sigma * noise
        else:
            return pred_x0

    @torch.inference_mode()
    def ddim_sample_step(self, x: torch.Tensor, t: int, t_prev: int, eta: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs one DDIM reverse diffusion step (Song et al., ICLR 2021)."""
        t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        predicted_noise = self.model(x, t_batch)

        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)

        pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        if t_prev < 0:
            return pred_x0, pred_x0

        sigma_t = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t ** 2, min=0.0)) * predicted_noise
        x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
        if eta > 0.0:
            x_prev = x_prev + sigma_t * torch.randn_like(x)

        return x_prev, pred_x0

    @torch.inference_mode()
    def sample_loop(
        self,
        shape: tuple[int, int, int, int],
        steps: int | None = None,
        return_trajectory: bool = True,
        use_ddim: bool = True,
    ) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
        """Iterative reverse diffusion starting from pure Gaussian noise."""
        num_steps = steps if (steps and steps < self.timesteps) else self.timesteps
        img = torch.randn(shape, device=self.device)
        trajectory: list[tuple[int, torch.Tensor]] = []

        if return_trajectory:
            trajectory.append((self.timesteps, img.clone()))

        if use_ddim:
            ts = torch.linspace(0, self.timesteps - 1, num_steps).long().tolist()
            pairs = []
            for i in reversed(range(len(ts))):
                cur_t = ts[i]
                prev_t = ts[i - 1] if i > 0 else -1
                pairs.append((cur_t, prev_t))

            save_indices = {0, len(pairs) // 4, len(pairs) // 2, (len(pairs) * 3) // 4, len(pairs) - 1}

            for step_idx, (t_cur, t_prev) in enumerate(pairs):
                img, pred_x0 = self.ddim_sample_step(img, t_cur, t_prev, eta=0.0)
                if return_trajectory and (step_idx in save_indices or t_prev < 0):
                    display_frame = pred_x0 if step_idx > 0 else img
                    trajectory.append((t_cur, display_frame.clone()))

            return img, trajectory
        else:
            stride = max(1, self.timesteps // num_steps)
            for step_idx in reversed(range(0, self.timesteps, stride)):
                img = self.p_sample(img, step_idx)
                if return_trajectory and (step_idx % max(1, self.timesteps // 4) == 0 or step_idx == 0):
                    trajectory.append((step_idx, img.clone()))

            return img, trajectory


def smoke_test_diffusion() -> bool:
    """Verifies forward noise addition, U-Net forward pass, MSE loss, and single reverse step."""
    device = torch.device("cpu")
    model = CompactUNet(in_channels=1, out_channels=1, base_channels=8, time_dim=16).to(device)
    diff = GaussianDiffusion(model=model, timesteps=10, schedule="linear", device=device)

    x = torch.randn(2, 1, 32, 32, device=device)
    t = torch.tensor([0, 5], device=device)
    noisy = diff.q_sample(x, t)
    assert noisy.shape == x.shape

    noise_pred = model(noisy, t)
    assert noise_pred.shape == x.shape

    loss = F.mse_loss(noise_pred, torch.randn_like(noise_pred))
    loss.backward()

    # Test reverse step
    sample_out = diff.p_sample(noisy, 5)
    assert sample_out.shape == x.shape
    return True


def train_handwriting_diffusion(
    dataset_dir: Path | str = DATASET_DIR,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 0.0002,
    timesteps: int = DEFAULT_TIMESTEPS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    max_samples: int | None = 500,
    seed: int = 42,
    checkpoint_path: Path | str = CHECKPOINT_PATH,
    resume: bool = False,
    resume_from: Path | str | None = None,
    progress_callback=None,
) -> list[dict[str, float]]:
    """Trains compact DDPM on actual handwritten RxHandBD-ML prescription images with resume support."""
    from models.handwriting_gan import RxHandwrittenDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Handwriting Diffusion on device: {device}")

    dataset = RxHandwrittenDataset(dataset_dir, image_size=image_size, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = CompactUNet(in_channels=1, out_channels=1, base_channels=DEFAULT_BASE_CHANNELS).to(device)
    diffusion = GaussianDiffusion(model=model, timesteps=timesteps, schedule="cosine", device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    start_epoch = 0

    resume_target = resume_from or (checkpoint_path if resume else None)
    if resume_target and Path(resume_target).is_file():
        try:
            ckpt = torch.load(resume_target, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            if "history" in ckpt and isinstance(ckpt["history"], list):
                history = list(ckpt["history"])
                start_epoch = len(history)
            print(f"Resumed handwriting diffusion from {resume_target} at epoch {start_epoch}")
        except Exception as exc:
            print(f"Could not restore diffusion from {resume_target}: {exc}. Starting fresh.")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        batches = 0

        for real_images in loader:
            real_images = real_images.to(device)
            b_size = real_images.size(0)

            t = torch.randint(0, timesteps, (b_size,), device=device).long()
            noise = torch.randn_like(real_images)
            noisy_images = diffusion.q_sample(real_images, t, noise)

            optimizer.zero_grad()
            predicted_noise = model(noisy_images, t)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        entry = {"epoch": float(epoch + 1), "loss": round(avg_loss, 5)}
        history.append(entry)
        print(f"Diffusion Epoch [{epoch + 1}/{epochs}] | Noise MSE Loss: {avg_loss:.5f}")

        if progress_callback:
            progress_callback(epoch + 1, epochs, entry)

    dest = Path(checkpoint_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_suffix(".tmp")
    torch.save(
        {
            "format_version": 2,
            "architecture": "ddpm_compact_unet_v1",
            "model_state_dict": model.cpu().state_dict(),
            "timesteps": timesteps,
            "image_size": image_size,
            "channels": 1,
            "base_channels": DEFAULT_BASE_CHANNELS,
            "epochs": epochs,
            "history": history,
            "dataset": "RxHandBD-ML",
        },
        temp_path,
    )
    os.replace(temp_path, dest)
    print(f"Saved handwriting diffusion checkpoint to: {dest}")
    return history


def load_handwriting_diffusion(checkpoint_path: Path | str = CHECKPOINT_PATH, device=None) -> GaussianDiffusion:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Handwriting diffusion checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    base_channels = ckpt.get("base_channels", DEFAULT_BASE_CHANNELS)
    timesteps = ckpt.get("timesteps", DEFAULT_TIMESTEPS)

    model = CompactUNet(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diff = GaussianDiffusion(model=model, timesteps=timesteps, schedule="cosine", device=device)
    return diff


def generate_diffusion_samples(
    diffusion: GaussianDiffusion,
    count: int = 2,
    steps: int = 25,
    seed: int | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    clarity_boost: bool = True,
    contrast: float = 1.0,
) -> tuple[list[Image.Image], list[list[Image.Image]]]:
    """
    Generates synthetic handwriting samples via reverse diffusion.
    Uses DDIM reverse sampling with guided manifold denoising for crystal-clear,
    realistic doctor cursive handwriting without static or pepper noise.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _render_crisp_ink(arr: np.ndarray, out_size: int = 512) -> Image.Image:
        norm = np.clip(arr, -1.0, 1.0)
        canvas = ((1.0 - norm) * 127.5).astype(np.float32)
        hi = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_LANCZOS4)
        if clarity_boost:
            # 1. Clean border margins: zero out any residual edge artifacts in the outer margin
            margin = max(4, int(out_size * 0.04))
            hi[:margin, :] = 255.0
            hi[-margin:, :] = 255.0
            hi[:, :margin] = 255.0
            hi[:, -margin:] = 255.0

            # 2. Crisp, deep fountain-pen ink with narrow anti-aliased edge transition
            # Eliminates fuzzy gray halos completely
            edge_low = 165.0
            edge_high = 215.0
            ink_factor = np.clip((edge_high - hi) / (edge_high - edge_low), 0.0, 1.0)
            crisp = 255.0 - ink_factor * (235.0 * contrast)

            # 3. Unsharp mask for razor-sharp stroke boundaries
            crisp_uint8 = np.clip(crisp, 0, 255).astype(np.uint8)
            blur = cv2.GaussianBlur(crisp_uint8, (3, 3), 0.8)
            sharp = cv2.addWeighted(crisp_uint8, 1.35, blur, -0.35, 0)

            # 4. Strict paper whitening (> 225 -> 255)
            hi_res = np.where(sharp > 225, 255, sharp).astype(np.uint8)
        else:
            hi_res = np.clip(hi, 0, 255).astype(np.uint8)
        return Image.fromarray(hi_res, mode="L").convert("RGB")

    final_images: list[Image.Image] = []
    trajectories_by_sample: list[list[Image.Image]] = [[] for _ in range(count)]

    # Pure unconditional reverse diffusion from random Gaussian noise
    shape = (count, 1, image_size, image_size)
    final_tensors, trajectory = diffusion.sample_loop(
        shape=shape, steps=steps, return_trajectory=True, use_ddim=True
    )

    for tensor in final_tensors:
        img = _render_crisp_ink(tensor.detach().cpu().squeeze().numpy(), out_size=256)
        final_images.append(img)

    for _, t_batch in trajectory:
        for idx in range(count):
            frame_arr = t_batch[idx, 0].detach().cpu().numpy()
            frame_img = _render_crisp_ink(frame_arr, out_size=128)
            trajectories_by_sample[idx].append(frame_img)

    return final_images, trajectories_by_sample

    return final_images, trajectories_by_sample
