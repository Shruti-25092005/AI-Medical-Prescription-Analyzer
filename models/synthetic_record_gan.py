from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.diffusion_inference import render_synthetic_form


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "synthetic_record_gan.pth"
IMAGE_SIZE = 128
OUTPUT_SIZE = 512
LATENT_DIMENSION = 64
FEATURES = 16
CHECKPOINT_FORMAT_VERSION = 6
GAN_ARCHITECTURE = "dcgan_gray128_synthetic_document_v6"
SYNTHETIC_OUTPUT_LABEL = (
    "AI-GENERATED SYNTHETIC DATA — NOT A REAL PRESCRIPTION — "
    "NOT FOR MEDICAL USE"
)
SYNTHETIC_IMAGE_BANNER = (
    "AI-GENERATED - NOT A PRESCRIPTION - NOT FOR MEDICAL USE"
)


@dataclass
class FictionalRecord:
    record_id: str
    content: str
    status: str
    patient_name: str = "Synthetic Patient"
    patient_age: int = 40
    patient_gender: str = "Unspecified"
    date: str = "2026-10-26"
    doctor_name: str = "Dr. Medical Staff, MD"
    doctor_reg: str = "PMC-000000"
    medicines: list[dict[str, str]] | None = None


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIMENSION, FEATURES * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(FEATURES * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURES * 8, FEATURES * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURES * 4, FEATURES * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURES * 2, FEATURES, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURES, FEATURES // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURES // 2, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.layers(noise)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, FEATURES, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(FEATURES, FEATURES * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(FEATURES * 2, FEATURES * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(FEATURES * 4, FEATURES * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(FEATURES * 8, FEATURES * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(FEATURES * 16, 1, 4, 1, 0, bias=False),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image).view(-1)


def _add_abstract_strokes(image: Image.Image, seed: int) -> Image.Image:
    """Add non-linguistic ink-like curves to a blank synthetic form."""
    rng = random.Random(seed)
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    ink = rng.randrange(20, 85)
    stroke_width = max(1, width // 128)
    for _ in range(rng.randint(2, 5)):
        left = rng.randint(width // 5, width // 3)
        right = rng.randint(width * 2 // 3, width * 4 // 5)
        baseline = rng.randint(height * 2 // 5, height * 4 // 5)
        amplitude = rng.randint(max(2, height // 100), max(3, height // 35))
        cycles = rng.uniform(1.0, 2.8)
        point_count = rng.randint(12, 25)
        points = []
        for point in range(point_count):
            progress = point / (point_count - 1)
            x = left + progress * (right - left)
            y = (
                baseline
                + np.sin(progress * cycles * 2 * np.pi) * amplitude
                + rng.uniform(-1.2, 1.2)
            )
            points.append((round(x), round(y)))
        draw.line(points, fill=ink, width=stroke_width, joint="curve")
    return image


class SyntheticDocumentDataset(Dataset):
    def __init__(self, count: int, seed: int = 42) -> None:
        if count < 2:
            raise ValueError("Synthetic dataset must contain at least two images.")
        self.count = count
        self.seed = seed

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> torch.Tensor:
        image = render_synthetic_form(index, IMAGE_SIZE, self.seed)
        image = _add_abstract_strokes(image, self.seed + index * 104_729)
        pixels = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(pixels).unsqueeze(0)
        return tensor.div(127.5).sub(1)


def _initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def train_gan(
    epochs: int = 40,
    record_count: int = 512,
    batch_size: int = 32,
    seed: int = 42,
    progress_callback=None,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> list[dict[str, float]]:
    if epochs < 1:
        raise ValueError("Epoch count must be at least 1.")
    if batch_size < 2:
        raise ValueError("Batch size must be at least 2.")
    if record_count < batch_size:
        raise ValueError("Record count must be at least the batch size.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticDocumentDataset(record_count, seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(_initialize_weights)
    discriminator.apply(_initialize_weights)
    loss_function = nn.MSELoss()
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=0.0001,
        betas=(0.5, 0.999),
    )

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        generator_loss_total = 0.0
        discriminator_loss_total = 0.0
        structure_loss_total = 0.0
        diversity_loss_total = 0.0
        batch_count = 0
        instance_noise = 0.05 * (1 - epoch / max(epochs - 1, 1))
        for real_images in loader:
            real_images = real_images.to(device, non_blocking=True)
            size = real_images.size(0)
            real_targets = torch.full((size,), 0.9, device=device)
            fake_targets = torch.zeros(size, device=device)
            noise = torch.randn(size, LATENT_DIMENSION, 1, 1, device=device)
            fake_images = generator(noise)

            discriminator_optimizer.zero_grad(set_to_none=True)
            real_input = real_images + torch.randn_like(real_images) * instance_noise
            fake_input = fake_images.detach() + (
                torch.randn_like(fake_images) * instance_noise
            )
            real_loss = loss_function(discriminator(real_input), real_targets)
            fake_loss = loss_function(
                discriminator(fake_input),
                fake_targets,
            )
            discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
            discriminator_optimizer.step()

            discriminator_optimizer.zero_grad(set_to_none=True)
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            generator_optimizer.zero_grad(set_to_none=True)
            adversarial_loss = loss_function(
                discriminator(
                    fake_images + torch.randn_like(fake_images) * instance_noise
                ),
                torch.full((size,), 0.9, device=device),
            )
            real_structure = nn.functional.adaptive_avg_pool2d(
                real_images,
                output_size=(16, 16),
            )
            fake_structure = nn.functional.adaptive_avg_pool2d(
                fake_images,
                output_size=(16, 16),
            )
            structure_loss = nn.functional.l1_loss(
                fake_structure.mean(dim=0),
                real_structure.mean(dim=0),
            )
            real_pair_distance = torch.abs(
                real_structure - real_structure.roll(1, dims=0)
            ).mean()
            fake_pair_distance = torch.abs(
                fake_structure - fake_structure.roll(1, dims=0)
            ).mean()
            diversity_loss = nn.functional.l1_loss(
                fake_pair_distance,
                real_pair_distance,
            )
            generator_loss = (
                adversarial_loss
                + 2.0 * diversity_loss
                + 3.0 * structure_loss
            )
            generator_loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            generator_optimizer.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

            generator_loss_total += generator_loss.item()
            discriminator_loss_total += discriminator_loss.item()
            structure_loss_total += structure_loss.item()
            diversity_loss_total += diversity_loss.item()
            batch_count += 1

        metrics = {
            "epoch": float(epoch + 1),
            "generator_loss": generator_loss_total / batch_count,
            "discriminator_loss": discriminator_loss_total / batch_count,
            "structure_loss": structure_loss_total / batch_count,
            "diversity_loss": diversity_loss_total / batch_count,
        }
        history.append(metrics)
        if progress_callback is not None:
            progress_callback(epoch + 1, epochs, metrics)

    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".tmp")
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "architecture": GAN_ARCHITECTURE,
            "generator_state_dict": generator.cpu().state_dict(),
            "image_size": IMAGE_SIZE,
            "image_channels": 1,
            "latent_dimension": LATENT_DIMENSION,
            "features": FEATURES,
            "epochs": epochs,
            "training_record_count": record_count,
            "training_data": "procedural_blank_forms_with_abstract_strokes_v1",
            "seed": seed,
            "history": history,
        },
        temporary_path,
    )
    os.replace(temporary_path, destination)
    return history


def _upscale_generated_image(image: Image.Image) -> Image.Image:
    return image.convert("L").resize(
        (OUTPUT_SIZE, OUTPUT_SIZE),
        Image.Resampling.LANCZOS,
    ).convert("RGB")


def _tensor_to_grayscale_image(image: torch.Tensor) -> Image.Image:
    pixels = (
        ((image.detach().cpu().squeeze().clamp(-1, 1) + 1) * 127.5)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    if pixels.ndim != 2:
        pixels = np.squeeze(pixels)
    return Image.fromarray(pixels, mode="L")


def _watermark_image(image: Image.Image) -> Image.Image:
    image = _upscale_generated_image(image)
    draw = ImageDraw.Draw(image)
    banner_top = OUTPUT_SIZE - 58
    draw.rectangle((0, banner_top, OUTPUT_SIZE, OUTPUT_SIZE), fill=(145, 28, 38))
    font = ImageFont.load_default()
    text = SYNTHETIC_IMAGE_BANNER
    bounds = draw.textbbox((0, 0), text, font=font)
    text_width = bounds[2] - bounds[0]
    if text_width > OUTPUT_SIZE - 12:
        text = "SYNTHETIC - NOT FOR MEDICAL USE"
        bounds = draw.textbbox((0, 0), text, font=font)
        text_width = bounds[2] - bounds[0]
    draw.text(
        ((OUTPUT_SIZE - text_width) // 2, banner_top + 24),
        text,
        fill=(255, 255, 255),
        font=font,
    )
    return image


def generate_records(
    count: int = 4,
    seed: int | None = None,
) -> list[tuple[FictionalRecord, Image.Image]]:
    if count < 1:
        raise ValueError("Record count must be at least 1.")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"GAN checkpoint not found: {CHECKPOINT_PATH}. "
            "Run notebooks/GAN_Training.ipynb to train it."
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION
        or checkpoint.get("architecture") != GAN_ARCHITECTURE
    ):
        raise ValueError(
            "The existing GAN checkpoint was trained on the legacy card "
            "dataset and is incompatible. Retrain it with "
            "notebooks/GAN_Training.ipynb."
        )
    if (
        checkpoint.get("image_size") != IMAGE_SIZE
        or checkpoint.get("image_channels") != 1
        or checkpoint.get("latent_dimension") != LATENT_DIMENSION
        or checkpoint.get("features") != FEATURES
    ):
        raise ValueError("GAN checkpoint architecture does not match this model.")

    generator = Generator()
    generator.load_state_dict(checkpoint["generator_state_dict"], strict=True)
    generator.eval()

    noise_generator = (
        torch.Generator(device="cpu").manual_seed(seed)
        if seed is not None
        else None
    )
    noise = torch.randn(
        count,
        LATENT_DIMENSION,
        1,
        1,
        generator=noise_generator,
    )
    with torch.inference_mode():
        generated_images = generator(noise)

    id_generator = random.Random(seed)
    results = []
    for generated in generated_images:
        image = _watermark_image(_tensor_to_grayscale_image(generated))
        record_id = id_generator.randrange(1, 1_000_000)
        record = FictionalRecord(
            record_id=f"SYNTHETIC-{record_id:06d}",
            content="GAN-generated text-free form",
            status="Synthetic — not for medical use",
        )
        results.append((record, image))
    return results


def records_as_dicts(records: list[FictionalRecord]) -> list[dict[str, Any]]:
    return [
        {
            "SampleID": record.record_id,
            "Patient": getattr(record, "patient_name", "Synthetic Patient"),
            "Age": getattr(record, "patient_age", 40),
            "Gender": getattr(record, "patient_gender", "Unspecified"),
            "Doctor": getattr(record, "doctor_name", "Dr. Staff, MD"),
            "content": record.content,
            "status": record.status,
        }
        for record in records
    ]
