from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "synthetic_record_gan.pth"
IMAGE_SIZE = 128
LATENT_DIMENSION = 64
FEATURES = 16
SYNTHETIC_OUTPUT_LABEL = (
    "AI-GENERATED SYNTHETIC DATA — NOT A REAL PRESCRIPTION — "
    "NOT FOR MEDICAL USE"
)
SYNTHETIC_IMAGE_BANNER = "AI-GENERATED - NOT A PRESCRIPTION"


@dataclass(frozen=True)
class FictionalRecord:
    record_id: str
    content: str
    status: str


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
            nn.ConvTranspose2d(FEATURES // 2, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.layers(noise)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, FEATURES, 4, 2, 1, bias=False),
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


def _make_record(index: int, seed: int = 0) -> FictionalRecord:
    return FictionalRecord(
        record_id=f"SYNTHETIC-{(seed + index) % 1_000_000:06d}",
        content="NO CLINICAL CONTENT",
        status="NOT A PRESCRIPTION",
    )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_record(record: FictionalRecord, size: int = IMAGE_SIZE) -> Image.Image:
    image = Image.new("RGB", (size, size), color=(250, 248, 238))
    draw = ImageDraw.Draw(image)
    margin = max(5, size // 20)
    ink = (37, 56, 65)
    muted = (98, 111, 115)
    title_font = _font(max(8, size // 12))
    body_font = _font(max(7, size // 16))
    small_font = _font(max(6, size // 20))

    draw.rounded_rectangle(
        (2, 2, size - 3, size - 3),
        radius=max(3, size // 32),
        outline=(126, 151, 151),
        width=max(1, size // 64),
        fill=(255, 254, 249),
    )
    draw.text(
        (margin, margin),
        "FICTIONAL SYNTHETIC SAMPLE",
        fill=ink,
        font=title_font,
    )
    draw.line(
        (margin, margin + size // 6, size - margin, margin + size // 6),
        fill=(90, 133, 135),
        width=max(1, size // 64),
    )
    fields = [
        ("SAMPLE ID", record.record_id),
        ("CONTENT", record.content),
        ("STATUS", record.status),
    ]
    y = margin + size // 5
    row_height = max(10, size // 9)
    for label, value in fields:
        draw.text((margin, y), label, fill=muted, font=small_font)
        draw.text((margin, y + max(7, size // 28)), value, fill=ink, font=body_font)
        y += row_height

    banner_top = size - max(15, size // 7)
    draw.rectangle((0, banner_top, size, size), fill=(153, 39, 45))
    draw.text(
        (margin, banner_top + max(2, size // 48)),
        SYNTHETIC_IMAGE_BANNER,
        fill=(255, 255, 255),
        font=small_font,
    )
    return image


class SyntheticRecordDataset(Dataset):
    def __init__(self, count: int, seed: int = 42) -> None:
        if count < 2:
            raise ValueError("Synthetic dataset must contain at least two records.")
        self.records = [
            _make_record(index + 1, seed)
            for index in range(count)
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = render_record(self.records[index])
        tensor = torch.from_numpy(np.asarray(image, dtype="float32").copy())
        tensor = tensor.permute(2, 0, 1)
        return tensor.div(127.5).sub(1)


def _initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def train_gan(
    epochs: int = 5,
    record_count: int = 512,
    batch_size: int = 32,
    seed: int = 42,
    progress_callback=None,
) -> list[dict[str, float]]:
    if epochs < 1:
        raise ValueError("Epoch count must be at least 1.")
    if batch_size < 2:
        raise ValueError("Batch size must be at least 2.")
    if record_count < batch_size:
        raise ValueError("Record count must be at least the batch size.")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticRecordDataset(record_count, seed)
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
    loss_function = nn.BCEWithLogitsLoss()
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

    history = []
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        generator_loss_total = 0.0
        discriminator_loss_total = 0.0
        batch_count = 0
        for real_images in loader:
            real_images = real_images.to(device, non_blocking=True)
            size = real_images.size(0)
            real_targets = torch.full((size,), 0.9, device=device)
            fake_targets = torch.zeros(size, device=device)
            noise = torch.randn(size, LATENT_DIMENSION, 1, 1, device=device)
            fake_images = generator(noise)

            discriminator_optimizer.zero_grad(set_to_none=True)
            real_loss = loss_function(discriminator(real_images), real_targets)
            fake_loss = loss_function(
                discriminator(fake_images.detach()),
                fake_targets,
            )
            discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss = loss_function(discriminator(fake_images), real_targets)
            generator_loss.backward()
            generator_optimizer.step()

            generator_loss_total += generator_loss.item()
            discriminator_loss_total += discriminator_loss.item()
            batch_count += 1

        metrics = {
            "epoch": float(epoch + 1),
            "generator_loss": generator_loss_total / batch_count,
            "discriminator_loss": discriminator_loss_total / batch_count,
        }
        history.append(metrics)
        if progress_callback is not None:
            progress_callback(epoch + 1, epochs, metrics)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")
    torch.save(
        {
            "generator_state_dict": generator.cpu().state_dict(),
            "image_size": IMAGE_SIZE,
            "latent_dimension": LATENT_DIMENSION,
            "features": FEATURES,
            "epochs": epochs,
            "training_record_count": record_count,
            "seed": seed,
        },
        temporary_path,
    )
    os.replace(temporary_path, CHECKPOINT_PATH)
    return history


def generate_records(
    count: int = 4,
    seed: int | None = None,
) -> list[tuple[FictionalRecord, Image.Image]]:
    if count < 1:
        raise ValueError("Record count must be at least 1.")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"GAN checkpoint not found: {CHECKPOINT_PATH}. "
            "Run notebooks/Synthetic_Record_GAN.ipynb to train it."
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("image_size") != IMAGE_SIZE
        or checkpoint.get("latent_dimension") != LATENT_DIMENSION
        or checkpoint.get("features") != FEATURES
    ):
        raise ValueError("GAN checkpoint architecture does not match this model.")

    generator = Generator()
    generator.load_state_dict(checkpoint["generator_state_dict"], strict=True)
    generator.eval()
    random_generator = random.Random(seed)
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

    results = []
    for generated in generated_images:
        rgb = ((generated.clamp(-1, 1) + 1) * 127.5).byte()
        generated_image = Image.fromarray(
            rgb.permute(1, 2, 0).numpy(),
            mode="RGB",
        ).resize((512, 512), Image.Resampling.BICUBIC)
        record = _make_record(
            random_generator.randrange(1, 1_000_000),
        )
        results.append((record, _compose_card(generated_image, record)))
    return results


def _compose_card(
    generated_background: Image.Image,
    record: FictionalRecord,
) -> Image.Image:
    image = generated_background.convert("RGB").filter(
        ImageFilter.GaussianBlur(radius=16)
    ).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        (20, 20, 492, 492),
        radius=18,
        fill=(255, 254, 249, 255),
        outline=(75, 116, 119, 255),
        width=3,
    )
    title_font = _font(27)
    label_font = _font(18)
    body_font = _font(22)
    draw.text(
        (42, 42),
        "FICTIONAL SYNTHETIC SAMPLE",
        fill=(31, 61, 64),
        font=title_font,
    )
    draw.line((42, 88, 470, 88), fill=(72, 126, 128, 255), width=3)
    rows = [
        ("SAMPLE ID", record.record_id),
        ("CONTENT", record.content),
        ("STATUS", record.status),
    ]
    y = 108
    for label, value in rows:
        draw.text((44, y), label, fill=(100, 111, 115), font=label_font)
        draw.text((205, y), value, fill=(34, 52, 60), font=body_font)
        y += 48
    draw.rectangle((0, 438, 512, 512), fill=(153, 39, 45, 255))
    draw.text(
        (35, 458),
        SYNTHETIC_IMAGE_BANNER,
        fill=(255, 255, 255),
        font=_font(22),
    )
    return Image.alpha_composite(image, overlay).convert("RGB")


def records_as_dicts(records: list[FictionalRecord]) -> list[dict[str, str]]:
    return [asdict(record) for record in records]
