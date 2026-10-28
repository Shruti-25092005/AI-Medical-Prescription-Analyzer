import argparse
import math
from pathlib import Path
import os
import pickle
import random

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "diffusion_model.pth"
IMAGE_SIZE = 128
TIMESTEPS = 100
BASE_CHANNELS = 16
CHECKPOINT_FORMAT_VERSION = 4
IMAGE_NORMALIZATION = "grayscale[-1,1]"
NOISE_SCHEDULE = "cosine_alpha_bar_s0.008_v1"
MODEL_ARCHITECTURE = "residual_unet_sinusoidal_time_v3"
PREDICTION_TYPE = "velocity_v1"


def cosine_noise_schedule(
    timesteps: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if timesteps < 2:
        raise ValueError("Diffusion requires at least 2 timesteps.")
    positions = torch.linspace(0, timesteps, timesteps + 1, device=device)
    positions = positions / timesteps
    offset = 0.008
    alpha_curve = torch.cos(
        ((positions + offset) / (1 + offset)) * (math.pi / 2)
    ).square()
    alpha_curve = alpha_curve / alpha_curve[0]
    betas = (1 - alpha_curve[1:] / alpha_curve[:-1]).clamp(
        min=1e-8,
        max=0.999,
    )
    alphas = 1 - betas
    return betas, alphas, torch.cumprod(alphas, dim=0)


def velocity_to_clean_and_noise(
    noisy_image: torch.Tensor,
    predicted_velocity: torch.Tensor,
    alpha_cumulative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = alpha_cumulative.to(
        device=noisy_image.device,
        dtype=noisy_image.dtype,
    )
    while alpha.ndim < noisy_image.ndim:
        alpha = alpha.unsqueeze(-1)
    predicted_clean = (
        alpha.sqrt() * noisy_image - (1 - alpha).sqrt() * predicted_velocity
    )
    predicted_noise = (
        (1 - alpha).sqrt() * noisy_image + alpha.sqrt() * predicted_velocity
    )
    return predicted_clean, predicted_noise
SYNTHETIC_OUTPUT_LABEL = (
    "AI-GENERATED SYNTHETIC — NOT A REAL PRESCRIPTION — "
    "NOT FOR MEDICAL USE"
)


def _setting(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st
        try:
            secret = st.secrets.get(name)
        except Exception:
            return None
    except Exception:
        return None

    return str(secret) if secret else None


def configured_checkpoint_path() -> Path:
    configured_path = _setting("DIFFUSION_MODEL_PATH")
    if configured_path:
        path = Path(configured_path).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    return DEFAULT_CHECKPOINT_PATH


def checkpoint_is_configured() -> bool:
    return (
        configured_checkpoint_path().is_file()
        or bool(_setting("DIFFUSION_MODEL_ID"))
    )


def configured_checkpoint_description() -> str:
    path = configured_checkpoint_path()
    if path.is_file():
        return str(path)
    model_id = _setting("DIFFUSION_MODEL_ID")
    if model_id:
        filename = _setting("DIFFUSION_MODEL_FILENAME") or "diffusion_model.pth"
        return f"{model_id}/{filename}"
    return str(path)


def _configured_model_id() -> str | None:
    return _setting("DIFFUSION_MODEL_ID")


def _resolve_checkpoint_path() -> Path:
    local_path = configured_checkpoint_path()
    if local_path.is_file():
        return local_path

    model_id = _configured_model_id()
    if not model_id:
        raise FileNotFoundError(
            "Diffusion checkpoint is not available. Train it with "
            "models/train_diffusion.py or configure DIFFUSION_MODEL_PATH."
        )

    from huggingface_hub import snapshot_download

    filename = _setting("DIFFUSION_MODEL_FILENAME") or "diffusion_model.pth"
    checkpoint_dir = snapshot_download(
        repo_id=model_id,
        token=_setting("HF_TOKEN"),
        allow_patterns=[filename],
    )
    checkpoint_path = Path(checkpoint_dir) / filename
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "The configured private model repository does not contain the "
            "diffusion checkpoint file."
        )
    return checkpoint_path


def _group_count(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_channels: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(input_channels), input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_channels, output_channels)
        self.norm2 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, image: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(functional.silu(self.norm1(image)))
        hidden = hidden + self.time_projection(functional.silu(time))[:, :, None, None]
        hidden = self.conv2(functional.silu(self.norm2(hidden)))
        return hidden + self.skip(image)


class DiffusionDenoiser(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = BASE_CHANNELS,
        timesteps: int = TIMESTEPS,
    ) -> None:
        super().__init__()
        if image_channels < 1 or base_channels < 4 or timesteps < 2:
            raise ValueError("Diffusion model dimensions and timesteps are invalid.")

        time_channels = base_channels * 4
        self.timesteps = timesteps
        self.time_embedding = nn.Sequential(
            nn.Linear(time_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.input = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, time_channels)
        self.downsample1 = nn.Conv2d(
            base_channels, base_channels * 2, 4, stride=2, padding=1
        )
        self.down2 = ResidualBlock(
            base_channels * 2, base_channels * 2, time_channels
        )
        self.downsample2 = nn.Conv2d(
            base_channels * 2, base_channels * 4, 4, stride=2, padding=1
        )
        self.middle = ResidualBlock(
            base_channels * 4, base_channels * 4, time_channels
        )
        self.up2 = ResidualBlock(
            base_channels * 6, base_channels * 2, time_channels
        )
        self.up1 = ResidualBlock(
            base_channels * 3, base_channels, time_channels
        )
        self.output_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.output = nn.Conv2d(base_channels, image_channels, 3, padding=1)

    def forward(self, image: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        half_channels = self.time_embedding[0].in_features // 2
        frequencies = torch.exp(
            -torch.log(image.new_tensor(10_000.0))
            * torch.arange(half_channels, device=image.device, dtype=image.dtype)
            / max(half_channels - 1, 1)
        )
        phase = timestep.to(image.dtype)[:, None] * frequencies[None, :]
        time = self.time_embedding(torch.cat((phase.sin(), phase.cos()), dim=1))
        first = self.down1(self.input(image), time)
        second = self.down2(self.downsample1(first), time)
        hidden = self.middle(self.downsample2(second), time)
        hidden = functional.interpolate(
            hidden, size=second.shape[-2:], mode="nearest"
        )
        hidden = self.up2(torch.cat((hidden, second), dim=1), time)
        hidden = functional.interpolate(
            hidden, size=first.shape[-2:], mode="nearest"
        )
        hidden = self.up1(torch.cat((hidden, first), dim=1), time)
        return self.output(functional.silu(self.output_norm(hidden)))


def render_synthetic_form(index: int, image_size: int, seed: int) -> Image.Image:
    if image_size < 16 or image_size % 4 != 0:
        raise ValueError("Image size must be a multiple of 4 and at least 16.")

    target_size = 512
    img = Image.new("L", (target_size, target_size), color=255)
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed + index)

    top_m = 52
    bot_m = 52
    side_m = 16

    # Outer double border fitting cleanly between red watermark banners
    outer_border_ink = rng.randint(25, 45)
    inner_border_ink = rng.randint(65, 85)
    draw.rectangle(
        (side_m, top_m, target_size - side_m, target_size - bot_m),
        outline=outer_border_ink,
        width=2,
    )
    draw.rectangle(
        (side_m + 3, top_m + 3, target_size - side_m - 3, target_size - bot_m - 3),
        outline=inner_border_ink,
        width=1,
    )

    # 1. Header Box (Clinic Letterhead)
    header_top = top_m + 6
    header_bot = header_top + 48
    header_bg = rng.randint(245, 252)
    draw.rectangle(
        (side_m + 6, header_top, target_size - side_m - 6, header_bot),
        outline=outer_border_ink,
        fill=header_bg,
        width=1,
    )

    # Medical Cross Icon on left
    cx, cy = side_m + 28, header_top + 24
    cross_ink = rng.randint(30, 55)
    draw.rectangle((cx - 3, cy - 12, cx + 3, cy + 12), fill=cross_ink)
    draw.rectangle((cx - 12, cy - 3, cx + 12, cy + 3), fill=cross_ink)

    # Header rules
    name_len = rng.randint(18, 40)
    draw.line(
        (cx + 20, cy - 8, target_size - side_m - name_len, cy - 8),
        fill=outer_border_ink,
        width=2,
    )
    draw.line(
        (cx + 20, cy + 2, target_size - side_m - name_len - 50, cy + 2),
        fill=inner_border_ink,
        width=1,
    )
    draw.line(
        (cx + 20, cy + 10, target_size - side_m - name_len - 100, cy + 10),
        fill=rng.randint(100, 125),
        width=1,
    )

    # 2. Patient Demographics Subgrid (3 Columns)
    p_top = header_bot + 8
    p_bot = p_top + 44
    draw.rectangle(
        (side_m + 6, p_top, target_size - side_m - 6, p_bot),
        outline=rng.randint(55, 75),
        width=1,
    )
    inner_w = target_size - 2 * side_m - 12
    col1 = side_m + 6 + int(inner_w * 0.45)
    col2 = col1 + int(inner_w * 0.25)
    div_ink = rng.randint(80, 105)
    draw.line((col1, p_top, col1, p_bot), fill=div_ink, width=1)
    draw.line((col2, p_top, col2, p_bot), fill=div_ink, width=1)

    field_ink = rng.randint(50, 70)
    subfield_ink = rng.randint(100, 125)
    draw.line((side_m + 16, p_top + 14, col1 - 14, p_top + 14), fill=field_ink, width=1)
    draw.line((side_m + 16, p_top + 30, col1 - 14, p_top + 30), fill=subfield_ink, width=1)
    draw.line((col1 + 14, p_top + 14, col2 - 14, p_top + 14), fill=field_ink, width=1)
    draw.line((col1 + 14, p_top + 30, col2 - 14, p_top + 30), fill=subfield_ink, width=1)
    draw.line((col2 + 14, p_top + 14, target_size - side_m - 20, p_top + 14), fill=field_ink, width=1)
    draw.line((col2 + 14, p_top + 30, target_size - side_m - 20, p_top + 30), fill=subfield_ink, width=1)

    # 3. Rx Symbol
    rx_top = p_bot + 10
    rx_x, rx_y = side_m + 20, rx_top + 4
    rx_ink = rng.randint(15, 35)
    draw.line((rx_x, rx_y, rx_x, rx_y + 24), fill=rx_ink, width=3)
    draw.arc((rx_x - 3, rx_y, rx_x + 15, rx_y + 14), start=-90, end=90, fill=rx_ink, width=3)
    draw.line((rx_x + 5, rx_y + 12, rx_x + 15, rx_y + 24), fill=rx_ink, width=3)
    draw.line((rx_x + 10, rx_y + 17, rx_x + 20, rx_y + 27), fill=rx_ink, width=2)

    # 4. Medication Table Grid
    tbl_top = rx_top + 32
    tbl_bot = target_size - bot_m - 58
    table_border_ink = rng.randint(35, 55)
    draw.rectangle((side_m + 6, tbl_top, target_size - side_m - 6, tbl_bot), outline=table_border_ink, width=2)

    th_bot = tbl_top + 20
    draw.rectangle(
        (side_m + 6, tbl_top, target_size - side_m - 6, th_bot),
        outline=table_border_ink,
        fill=rng.randint(238, 246),
        width=1,
    )

    c_w = target_size - 2 * side_m - 12
    c_x1 = side_m + 6 + int(c_w * 0.08)
    c_x2 = side_m + 6 + int(c_w * 0.50)
    c_x3 = side_m + 6 + int(c_w * 0.78)
    for cx_line in (c_x1, c_x2, c_x3):
        draw.line((cx_line, tbl_top, cx_line, tbl_bot), fill=rng.randint(55, 75), width=1)

    num_rows = rng.choice([4, 5, 6])
    r_h = (tbl_bot - th_bot) / num_rows
    row_line_ink = rng.randint(70, 95)
    guide_ink = rng.randint(150, 180)
    for r in range(1, num_rows + 1):
        ry = int(th_bot + r * r_h)
        if ry < tbl_bot:
            draw.line((side_m + 6, ry, target_size - side_m - 6, ry), fill=row_line_ink, width=1)
            guide_y = ry - int(r_h * 0.35)
            for gx in range(c_x1 + 8, c_x2 - 8, 8):
                draw.line((gx, guide_y, gx + 4, guide_y), fill=guide_ink, width=1)
            for gx in range(c_x2 + 8, c_x3 - 8, 8):
                draw.line((gx, guide_y, gx + 4, guide_y), fill=guide_ink, width=1)

    # 5. Doctor Signature & Stamp Seal Box
    sig_top = tbl_bot + 8
    sig_bot = target_size - bot_m - 6
    draw.rectangle(
        (side_m + 6, sig_top, target_size - side_m - 6, sig_bot),
        outline=rng.randint(75, 95),
        fill=rng.randint(248, 253),
        width=1,
    )
    draw.line((side_m + 16, sig_top + 18, side_m + 180, sig_top + 18), fill=rng.randint(110, 135), width=1)
    draw.line((side_m + 16, sig_top + 28, side_m + 150, sig_top + 28), fill=rng.randint(130, 155), width=1)

    sig_line_x = target_size - side_m - 170
    draw.line((sig_line_x, sig_top + 30, target_size - side_m - 18, sig_top + 30), fill=rng.randint(35, 55), width=2)
    draw.ellipse(
        (target_size - side_m - 75, sig_top + 6, target_size - side_m - 25, sig_top + 38),
        outline=rng.randint(90, 120),
        width=1,
    )

    if image_size != target_size:
        return img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return img


def _watermark_image(image: Image.Image) -> Image.Image:
    image = image.convert("L").resize((512, 512), Image.Resampling.BICUBIC).convert("RGB")
    draw = ImageDraw.Draw(image)
    font_candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
    ]
    font_path = next((candidate for candidate in font_candidates if candidate.is_file()), None)

    def fitting_font(text: str, max_width: int, size: int) -> ImageFont.ImageFont:
        for font_size in range(size, 9, -1):
            try:
                font = (
                    ImageFont.truetype(str(font_path), font_size)
                    if font_path
                    else ImageFont.load_default()
                )
            except OSError:
                return ImageFont.load_default()
            bounds = draw.textbbox((0, 0), text, font=font)
            if bounds[2] - bounds[0] <= max_width:
                return font
        return ImageFont.load_default()

    header = "AI-GENERATED SYNTHETIC — NOT A PRESCRIPTION"
    footer_first = "NO REAL IDENTITY OR CLINICAL CONTENT"
    footer_second = "NOT A PRESCRIPTION — NOT FOR MEDICAL USE"

    draw.rectangle((0, 0, 512, 48), fill=(150, 25, 35))
    draw.text(
        (256, 23),
        header,
        fill=(255, 255, 255),
        font=fitting_font(header, 488, 19),
        anchor="mm",
    )
    draw.rectangle((0, 458, 512, 512), fill=(150, 25, 35))
    draw.text(
        (256, 471),
        footer_first,
        fill=(255, 255, 255),
        font=fitting_font(footer_first, 488, 16),
        anchor="mm",
    )
    draw.text(
        (256, 496),
        footer_second,
        fill=(255, 255, 255),
        font=fitting_font(footer_second, 488, 16),
        anchor="mm",
    )

    return image


def load_diffusion_model(
    checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else _resolve_checkpoint_path()
    )
    if not path.is_file():
        raise FileNotFoundError("The configured diffusion checkpoint was not found.")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "Diffusion checkpoint could not be read. Verify the checkpoint "
            "file and retrain if it is incomplete."
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("Diffusion checkpoint has an unsupported data format.")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Diffusion checkpoint is outdated or incompatible. Retrain it "
            "with `python -m models.train_diffusion`."
        )
    if (
        checkpoint.get("architecture") != MODEL_ARCHITECTURE
        or checkpoint.get("normalization") != IMAGE_NORMALIZATION
        or checkpoint.get("noise_schedule") != NOISE_SCHEDULE
        or checkpoint.get("prediction_type") != PREDICTION_TYPE
    ):
        raise ValueError(
            "Diffusion checkpoint architecture or preprocessing does not "
            "match inference. Retrain the checkpoint."
        )
    image_size = int(checkpoint["image_size"])
    timesteps = int(checkpoint["timesteps"])
    base_channels = int(checkpoint["base_channels"])
    if image_size < 16 or image_size % 4 or timesteps < 2 or base_channels < 4:
        raise ValueError("Diffusion checkpoint configuration is invalid.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionDenoiser(
        image_channels=1,
        base_channels=base_channels,
        timesteps=timesteps,
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, RuntimeError, TypeError) as exc:
        raise ValueError(
            "Diffusion checkpoint weights do not match the configured "
            "architecture. Retrain the checkpoint."
        ) from exc
    model.to(device).eval()
    return {
        "model": model,
        "image_size": image_size,
        "timesteps": timesteps,
        "device": device,
        "training_epochs": int(checkpoint.get("training_epochs", 0)),
        "training_sample_count": int(
            checkpoint.get("training_sample_count", 0)
        ),
        "validation_velocity_mse": float(
            checkpoint.get("validation_velocity_mse", float("nan"))
        ),
    }


def generate_image_comparisons(
    count: int = 4,
    sampling_steps: int = TIMESTEPS,
    seed: int | None = None,
    checkpoint_path: str | Path | None = None,
    model_bundle: dict[str, object] | None = None,
    return_trajectories: bool = False,
) -> list[tuple[Image.Image, Image.Image]] | tuple[list[tuple[Image.Image, Image.Image]], list[list[Image.Image]]]:
    if count < 1 or count > 8:
        raise ValueError("Sample count must be between 1 and 8.")
    if sampling_steps < 1:
        raise ValueError("Sampling steps must be at least 1.")

    bundle = model_bundle or load_diffusion_model(checkpoint_path)
    model = bundle["model"]
    image_size = int(bundle["image_size"])
    timesteps = int(bundle["timesteps"])
    device = torch.device(bundle["device"])
    if not isinstance(model, DiffusionDenoiser):
        raise TypeError("The diffusion model bundle is invalid.")

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()
    initial_noise = torch.randn(
        count,
        1,
        image_size,
        image_size,
        generator=generator,
        device=device,
    )
    image = initial_noise.clone()
    _, _, alpha_cumulative = cosine_noise_schedule(timesteps, device)
    step_count = min(sampling_steps, timesteps)
    selected_timesteps = torch.linspace(
        0,
        timesteps - 1,
        step_count,
        device=device,
    ).long().unique()

    trajectories_by_sample: list[list[Image.Image]] = [[] for _ in range(count)]
    save_indices = {0, len(selected_timesteps) // 4, len(selected_timesteps) // 2, (len(selected_timesteps) * 3) // 4, len(selected_timesteps) - 1}

    def _render_form_frame(sample_arr: np.ndarray, out_size: int = 512, watermark: bool = True) -> Image.Image:
        img_norm = np.clip((sample_arr + 1.0) * 127.5, 0.0, 255.0).astype(np.float32)
        if out_size != sample_arr.shape[0]:
            img_res = cv2.resize(img_norm, (out_size, out_size), interpolation=cv2.INTER_LANCZOS4)
        else:
            img_res = img_norm
        crisp = np.where(img_res > 228.0, 255.0, img_res)
        crisp = np.where(crisp < 160.0, crisp * 0.75, crisp)
        raw_img = Image.fromarray(np.clip(crisp, 0, 255).astype(np.uint8), mode="L")
        return _watermark_image(raw_img) if watermark else raw_img.convert("RGB")

    # Record initial noise frame
    if return_trajectories:
        for idx in range(count):
            n_px = ((initial_noise[idx, 0].clamp(-2.5, 2.5) + 2.5) * (255 / 5.0)).byte().cpu().numpy()
            trajectories_by_sample[idx].append(Image.fromarray(n_px, mode="L").resize((128, 128), Image.Resampling.NEAREST).convert("RGB"))

    with torch.inference_mode():
        for index in range(len(selected_timesteps) - 1, -1, -1):
            current_timestep = selected_timesteps[index]
            timestep_batch = current_timestep.expand(count)
            alpha_current = alpha_cumulative[current_timestep]
            predicted_velocity = model(image, timestep_batch)
            predicted_clean, predicted_noise = velocity_to_clean_and_noise(
                image,
                predicted_velocity,
                alpha_current,
            )
            predicted_clean = predicted_clean.clamp(-1, 1)

            if index == 0:
                image = predicted_clean
            else:
                predicted_noise = (
                    image - alpha_current.sqrt() * predicted_clean
                ) / (1 - alpha_current).sqrt().clamp_min(1e-8)
                previous_timestep = selected_timesteps[index - 1]
                alpha_previous = alpha_cumulative[previous_timestep]
                image = (
                    alpha_previous.sqrt() * predicted_clean
                    + (1 - alpha_previous).sqrt() * predicted_noise
                )

            if return_trajectories and (index in save_indices or index == 0):
                for idx in range(count):
                    display_sample = (predicted_clean if index > 0 else image)[idx, 0].cpu().numpy()
                    frame_img = _render_form_frame(display_sample, out_size=128, watermark=False)
                    trajectories_by_sample[idx].append(frame_img)

    comparisons = []
    for idx, (starting_noise, sample) in enumerate(zip(initial_noise, image.clamp(-1, 1))):
        noise_pixels = (
            (starting_noise[0].clamp(-3, 3) + 3) * (255 / 6)
        ).byte().cpu().numpy()
        noise_image = Image.fromarray(noise_pixels, mode="L").resize(
            (512, 512),
            Image.Resampling.NEAREST,
        ).convert("RGB")

        sample_arr = sample[0].cpu().numpy()
        generated_image = _render_form_frame(sample_arr, out_size=512, watermark=True)
        comparisons.append((noise_image, generated_image))

    if return_trajectories:
        return comparisons, trajectories_by_sample
    return comparisons


def generate_images(
    count: int = 4,
    sampling_steps: int = TIMESTEPS,
    seed: int | None = None,
    checkpoint_path: str | Path | None = None,
    model_bundle: dict[str, object] | None = None,
) -> list[Image.Image]:
    return [
        generated
        for _, generated in generate_image_comparisons(
            count=count,
            sampling_steps=sampling_steps,
            seed=seed,
            checkpoint_path=checkpoint_path,
            model_bundle=model_bundle,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clearly labeled synthetic form images."
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--steps", type=int, default=TIMESTEPS)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "generated_images" / "diffusion",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = generate_images(args.count, args.steps, args.seed)
    for index, image in enumerate(images, start=1):
        output_path = args.output_dir / f"synthetic_form_{index:02d}.png"
        image.save(output_path)
        print(f"Saved synthetic sample: {output_path}")


if __name__ == "__main__":
    main()
