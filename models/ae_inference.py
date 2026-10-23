from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "dataset" / "RxHandBD-ML" / "RxHandBD-ML"
CHECKPOINT_PATH = ROOT / "models" / "ae_best_model.pth"
OUTPUT_DIR = ROOT / "models" / "ae_outputs"
MAX_IMAGE_SIDE = 1024
MODEL_DOWNSAMPLE_FACTOR = 8
LATENT_CHANNELS = 64


class SkipAutoencoder(nn.Module):
    def __init__(self, latent_channels=LATENT_CHANNELS):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.to_latent = nn.Conv2d(256, latent_channels, kernel_size=1)
        self.from_latent = nn.Conv2d(latent_channels, 256, kernel_size=1)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        z = self.to_latent(e3)
        x = self.from_latent(z)
        x = self.dec3(x)
        x = self.dec2(torch.cat([x, e2], dim=1))
        reconstruction = self.dec1(torch.cat([x, e1], dim=1))
        return reconstruction, z


def load_autoencoder(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SkipAutoencoder().to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device


def _prepare_image(image):
    image = image.convert("RGB")
    width, height = image.size

    longest_side = max(width, height)
    if longest_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest_side
        resized_size = (round(width * scale), round(height * scale))
        image = image.resize(resized_size, Image.Resampling.LANCZOS)

    input_tensor = transforms.ToTensor()(image).unsqueeze(0)
    height, width = input_tensor.shape[-2:]
    pad_height = (-height) % MODEL_DOWNSAMPLE_FACTOR
    pad_width = (-width) % MODEL_DOWNSAMPLE_FACTOR
    if pad_height or pad_width:
        input_tensor = torch.nn.functional.pad(
            input_tensor,
            (0, pad_width, 0, pad_height),
            mode="replicate",
        )

    return input_tensor, (width, height)


def reconstruct_image(image, model, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor, (width, height) = _prepare_image(image)
    input_tensor = input_tensor.to(device)

    with torch.inference_mode():
        reconstruction, _ = model(input_tensor)

    reconstruction = reconstruction[..., :height, :width]
    reconstruction = reconstruction.squeeze(0).clamp(0, 1).cpu()
    return transforms.ToPILImage()(reconstruction)


def enhance_reconstruction(image):
    rgb = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, chroma_a, chroma_b = cv2.split(lab)

    denoised = cv2.fastNlMeansDenoising(
        lightness,
        None,
        h=2,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    contrast = cv2.createCLAHE(
        clipLimit=1.25,
        tileGridSize=(8, 8),
    ).apply(denoised)
    soft_edges = cv2.GaussianBlur(contrast, (0, 0), sigmaX=0.7)
    sharpened = cv2.addWeighted(contrast, 1.5, soft_edges, -0.5, 0)

    enhanced_lab = cv2.merge((sharpened, chroma_a, chroma_b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)


def adaptive_threshold_preview(image):
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return Image.fromarray(threshold)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device = load_autoencoder(device)
    print(f"Loaded AE checkpoint: {CHECKPOINT_PATH}")

    test_dir = DATASET_ROOT / "Test_Set"
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        path for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported test images found in {test_dir}")
    sample_path = image_paths[0]

    with Image.open(sample_path) as image:
        input_image = image.convert("RGB")
        input_tensor, (width, height) = _prepare_image(input_image)
        input_tensor = input_tensor.to(device)
        reconstruction_image = reconstruct_image(input_image, model, device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transforms.ToPILImage()(
        input_tensor[..., :height, :width].squeeze(0).cpu()
    ).save(OUTPUT_DIR / "original.png")
    reconstruction_image.save(OUTPUT_DIR / "reconstruction.png")

    print(f"Test image: {sample_path}")
    print(f"Input shape: {tuple(input_tensor.shape)}")
    print(f"Reconstruction size: {reconstruction_image.size}")
    print(f"Saved original: {OUTPUT_DIR / 'original.png'}")
    print(f"Saved reconstruction: {OUTPUT_DIR / 'reconstruction.png'}")


if __name__ == "__main__":
    main()