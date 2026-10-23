from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "dataset" / "RxHandBD-ML" / "RxHandBD-ML"


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.gn1 = nn.GroupNorm(8, channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = out + residual
        out = self.activation(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.gn = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.gn(x)
        x = self.activation(x)
        return x


class VAEEncoder(nn.Module):
    def __init__(self, latent_channels=32, image_size=256):
        super().__init__()
        self.latent_channels = latent_channels
        self.image_size = image_size

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.GroupNorm(8, 512),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(512),
            ResidualBlock(512),
        )
        self.to_mu = nn.Conv2d(512, latent_channels, 1)
        self.to_log_var = nn.Conv2d(512, latent_channels, 1)

    def forward(self, x):
        h = self.features(x)
        z_mean = self.to_mu(h)
        z_log_var = self.to_log_var(h)
        z_log_var = torch.clamp(z_log_var, min=-10.0, max=10.0)
        return z_mean, z_log_var


class VAEDecoder(nn.Module):
    def __init__(self, latent_channels=32):
        super().__init__()
        self.latent_channels = latent_channels
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_channels, 512, 1),
            nn.GroupNorm(8, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsample = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            UpBlock(512, 256),
            ResidualBlock(256),
            UpBlock(256, 128),
            ResidualBlock(128),
            UpBlock(128, 64),
            ResidualBlock(64),
            UpBlock(64, 32),
            UpBlock(32, 32),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.from_latent(z)
        x = self.upsample(x)
        return x


class SharperVAE(nn.Module):
    def __init__(self, latent_channels=32, image_size=256):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = VAEEncoder(latent_channels, image_size)
        self.decoder = VAEDecoder(latent_channels)

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, z_mean, z_log_var):
        sigma = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(sigma)
        return z_mean + sigma * epsilon

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decode(z)
        return reconstruction, z_mean, z_log_var


def find_checkpoint(project_root: Path):
    candidates = [
        project_root / "models" / "final_sharper_vae_ema.pth",
        project_root / "models" / "final_sharper_vae_raw.pth",
        project_root / "models" / "sharper_vae_best_ema.pth",
        project_root / "models" / "sharper_vae_best.pth",
        project_root / "saved_models" / "final_sharper_vae_ema.pth",
        project_root / "saved_models" / "final_sharper_vae_raw.pth",
        project_root / "saved_models" / "vae_encoder_ema.pth",
        project_root / "saved_models" / "vae_decoder_ema.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = []
    for path in sorted(project_root.rglob("*.pth")) + sorted(project_root.rglob("*.pt")):
        if ".venv" in path.parts or "site-packages" in path.parts:
            continue
        lower_name = path.name.lower()
        if any(token in lower_name for token in ["vae", "sharper", "encoder", "decoder", "final"]):
            matches.append(path)

    if matches:
        return matches[0]

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "No trained VAE checkpoint found. Searched these standard VAE paths:\n"
        + searched
        + "\n\nAlso searched project files with names matching VAE/encoder/decoder patterns, excluding .venv and site-packages artifacts."
    )


def load_vae_checkpoint(checkpoint_path: Path):
    if ".venv" in checkpoint_path.parts or "site-packages" in checkpoint_path.parts:
        raise FileNotFoundError(
            f"Selected checkpoint path is not a VAE model file: {checkpoint_path}. "
            "This usually means a virtual environment artifact was mistaken for a trained model."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint format is not a dict: {type(checkpoint)}")
    required_keys = {"model_state_dict", "latent_channels", "latent_spatial", "image_size"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise KeyError(f"Checkpoint at {checkpoint_path} is missing keys: {sorted(missing_keys)}")
    return checkpoint


def preprocess_sample(image_path: Path, image_size: int = 256):
    image = Image.open(image_path).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    tensor = transforms.ToTensor()(image)
    tensor = tensor.unsqueeze(0)
    return tensor


def load_vae_model(device=None, checkpoint_path: Path | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = find_checkpoint(ROOT)
    checkpoint = load_vae_checkpoint(checkpoint_path)
    latent_channels = int(checkpoint["latent_channels"])
    latent_spatial = int(checkpoint["latent_spatial"])
    image_size = int(checkpoint["image_size"])
    model = SharperVAE(
        latent_channels=latent_channels,
        image_size=image_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, latent_spatial, image_size, device


def reconstruct_image(image, model, image_size: int, device):
    grayscale = image.convert("L").resize(
        (image_size, image_size),
        Image.Resampling.LANCZOS,
    )
    image_tensor = transforms.ToTensor()(grayscale).unsqueeze(0).to(device)

    with torch.inference_mode():
        z_mean, _ = model.encode(image_tensor)
        reconstruction = model.decode(z_mean)

    pixels = (
        reconstruction.detach()
        .cpu()
        .squeeze()
        .clamp(0, 1)
        .mul(255)
        .to(torch.uint8)
        .numpy()
    )
    return Image.fromarray(pixels, mode="L")


def save_side_by_side(original_tensor: torch.Tensor, reconstructed_tensor: torch.Tensor, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    original = original_tensor.detach().cpu().squeeze().numpy()
    recon = reconstructed_tensor.detach().cpu().squeeze().numpy()

    original_img = Image.fromarray((original * 255).astype("uint8"), mode="L")
    recon_img = Image.fromarray((recon * 255).astype("uint8"), mode="L")

    original_img.save(output_dir / "original.png")
    recon_img.save(output_dir / "reconstruction.png")

    print(f"Saved original image: {output_dir / 'original.png'}")
    print(f"Saved reconstruction: {output_dir / 'reconstruction.png'}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = find_checkpoint(ROOT)
    print(f"Loading checkpoint: {checkpoint_path}")
    model, latent_spatial, image_size, device = load_vae_model(
        device,
        checkpoint_path,
    )
    print("VAE checkpoint loaded successfully.")

    sample_candidates = [
        DATASET_ROOT / "Test_Set" / "P0001.jpg",
        DATASET_ROOT / "Train_Set" / "P1116.jpg",
        DATASET_ROOT / "Train_Set" / "P1120.jpg",
    ]
    sample_path = next((p for p in sample_candidates if p.exists()), None)
    if sample_path is None:
        raise FileNotFoundError(f"No sample image found under {DATASET_ROOT}")

    image_tensor = preprocess_sample(sample_path, image_size=image_size).to(device)
    with torch.inference_mode():
        z_mean, _ = model.encode(image_tensor)
        expected_latent_shape = (
            model.latent_channels,
            latent_spatial,
            latent_spatial,
        )
        if tuple(z_mean.shape[1:]) != expected_latent_shape:
            raise ValueError(
                f"Encoded latent shape {tuple(z_mean.shape[1:])} does not match checkpoint metadata "
                f"{expected_latent_shape}"
            )
        reconstruction = model.decode(z_mean)
        if tuple(reconstruction.shape[-2:]) != (image_size, image_size):
            raise ValueError(
                f"Reconstruction size {tuple(reconstruction.shape[-2:])} does not match image_size {image_size}"
            )

    save_side_by_side(image_tensor.cpu(), reconstruction.cpu(), ROOT / "models" / "vae_outputs")

    print("Loading and inference worked.")
    print(f"Sample image used: {sample_path}")
    print(f"Input shape: {tuple(image_tensor.shape)}")
    print(f"Reconstruction shape: {tuple(reconstruction.shape)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n=== VAE LOAD/INFERENCE ERROR ===")
        raise
