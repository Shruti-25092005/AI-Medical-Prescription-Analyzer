import argparse
import csv
import copy
import errno
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from models.diffusion_inference import (
    BASE_CHANNELS,
    DEFAULT_CHECKPOINT_PATH,
    IMAGE_SIZE,
    CHECKPOINT_FORMAT_VERSION,
    IMAGE_NORMALIZATION,
    MODEL_ARCHITECTURE,
    NOISE_SCHEDULE,
    PREDICTION_TYPE,
    TIMESTEPS,
    DiffusionDenoiser,
    cosine_noise_schedule,
    render_synthetic_form,
)


TRAINING_DATA_VERSION = "procedural_blank_forms_v2"


class SyntheticFormDataset(Dataset):
    def __init__(self, count: int, image_size: int, seed: int) -> None:
        if count < 2:
            raise ValueError("Training sample count must be at least 2.")
        self.count = count
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> torch.Tensor:
        image = render_synthetic_form(index, self.image_size, self.seed)
        pixels = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(pixels).unsqueeze(0)
        return tensor.div(127.5).sub(1)


def _write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    "epoch",
                    "train_velocity_mse",
                    "validation_velocity_mse",
                    "best_validation_velocity_mse",
                ),
            )
            writer.writeheader()
            writer.writerows(history)
        _replace_with_retry(temporary_path, path)
    except OSError:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def _replace_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(60):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if (
                exc.errno != errno.EACCES
                and getattr(exc, "winerror", None) not in (5, 32)
            ) or attempt == 59:
                raise
            time.sleep(0.5)


def _save_comparison_grid(sample_directory: Path) -> Path | None:
    samples = sorted(sample_directory.glob("epoch_*_sample_01.png"))
    if not samples:
        return None

    columns = min(3, len(samples))
    cell_width = 512
    label_height = 32
    cell_height = 512 + label_height
    rows = (len(samples) + columns - 1) // columns
    grid = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        color=(238, 242, 242),
    )
    draw = ImageDraw.Draw(grid)
    for index, path in enumerate(samples):
        row, column = divmod(index, columns)
        x = column * cell_width
        y = row * cell_height
        with Image.open(path) as source:
            grid.paste(source.convert("RGB").resize((cell_width, 512)), (x, y + label_height))
        draw.text(
            (x + 8, y + 8),
            f"Training epoch {path.stem.split('_')[1]}",
            fill=(25, 45, 50),
        )

    grid_path = sample_directory / "training_comparison_grid.png"
    grid.save(grid_path)
    return grid_path


def train(
    epochs: int = 100,
    sample_count: int = 512,
    batch_size: int = 16,
    image_size: int = IMAGE_SIZE,
    timesteps: int = TIMESTEPS,
    base_channels: int = BASE_CHANNELS,
    learning_rate: float = 2e-4,
    seed: int = 42,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    resume: bool = False,
    resume_from: str | Path | None = None,
    checkpoint_interval: int = 5,
    sample_interval: int = 10,
    samples_per_checkpoint: int = 2,
    sample_directory: str | Path | None = None,
) -> Path:
    if epochs < 1 or batch_size < 2:
        raise ValueError("Epochs must be positive and batch size at least 2.")
    if image_size < 16 or image_size % 4:
        raise ValueError("Image size must be a multiple of 4 and at least 16.")
    if (
        timesteps < 2
        or base_channels < 4
        or learning_rate <= 0
        or checkpoint_interval < 1
        or sample_interval < 0
        or samples_per_checkpoint < 1
    ):
        raise ValueError("Diffusion model training parameters are invalid.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    dataset = SyntheticFormDataset(sample_count, image_size, seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionDenoiser(
        image_channels=1,
        base_channels=base_channels,
        timesteps=timesteps,
    ).to(device)
    ema_model = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    _, _, alpha_cumulative = cosine_noise_schedule(timesteps, device)
    validation_dataset = SyntheticFormDataset(
        count=max(32, min(128, sample_count // 10)),
        image_size=image_size,
        seed=seed + 1_000_003,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(batch_size, 32),
        shuffle=False,
        num_workers=0,
    )
    validation_generator = torch.Generator(device=device).manual_seed(seed + 1)
    validation_inputs = [
        (
            images.to(device),
            torch.randint(
                0,
                timesteps,
                (images.size(0),),
                generator=validation_generator,
                device=device,
            ),
            torch.randn(
                images.size(0),
                1,
                image_size,
                image_size,
                generator=validation_generator,
                device=device,
            ),
        )
        for images in validation_loader
    ]
    destination = Path(checkpoint_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_samples = (
        Path(sample_directory)
        if sample_directory is not None
        else (
            DEFAULT_CHECKPOINT_PATH.parents[1]
            / "generated_images"
            / "diffusion"
            / f"{destination.stem}_training_progress"
        )
    )
    history_path = destination.with_name(
        f"{destination.stem}_training_history.csv"
    )
    history: list[dict[str, float | int]] = []
    start_epoch = 0
    optimizer_steps = 0
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    resume_path = Path(resume_from).expanduser() if resume_from else destination
    warm_start_epochs = 0
    same_training_resume = False
    if resume or resume_from is not None:
        if not resume_path.is_file():
            if resume_from is not None:
                raise FileNotFoundError(
                    f"Warm-start checkpoint not found: {resume_path}"
                )
        else:
            checkpoint = torch.load(
                resume_path,
                map_location="cpu",
                weights_only=True,
            )
            if (
                not isinstance(checkpoint, dict)
                or checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION
                or checkpoint.get("architecture") != MODEL_ARCHITECTURE
                or checkpoint.get("normalization") != IMAGE_NORMALIZATION
                or checkpoint.get("noise_schedule") != NOISE_SCHEDULE
                or checkpoint.get("prediction_type") != PREDICTION_TYPE
                or checkpoint.get("timesteps") != timesteps
                or checkpoint.get("base_channels") != base_channels
            ):
                raise ValueError(
                    "Warm-start checkpoint is incompatible with the requested "
                    "diffusion model architecture or noise contract."
                )
            same_training_resume = (
                checkpoint.get("image_size") == image_size
                and checkpoint.get("training_data_version")
                == TRAINING_DATA_VERSION
                and checkpoint.get("training_sample_count") == sample_count
                and checkpoint.get("seed") == seed
            )
            training_state = checkpoint.get(
                "resume_model_state_dict",
                checkpoint.get("model_state_dict"),
            )
            if not isinstance(training_state, dict):
                raise ValueError(
                    "Warm-start checkpoint does not contain model weights."
                )
            try:
                model.load_state_dict(training_state, strict=True)
                ema_state = checkpoint.get(
                    "ema_model_state_dict",
                    checkpoint.get("model_state_dict"),
                )
                if not isinstance(ema_state, dict):
                    raise TypeError("EMA model state is missing.")
                ema_model.load_state_dict(ema_state, strict=True)
            except (RuntimeError, TypeError) as exc:
                raise ValueError(
                    "Warm-start checkpoint weights do not match this model."
                ) from exc

            warm_start_epochs = int(checkpoint.get("training_epochs", 0))
            if same_training_resume:
                start_epoch = warm_start_epochs
                best_validation_loss = float(
                    checkpoint.get("validation_velocity_mse", float("inf"))
                )
                best_state_dict = checkpoint.get(
                    "best_model_state_dict",
                    checkpoint.get("model_state_dict"),
                )
                if isinstance(best_state_dict, dict):
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in best_state_dict.items()
                        if isinstance(value, torch.Tensor)
                    }
                saved_history = checkpoint.get("training_loss_history", [])
                if isinstance(saved_history, list):
                    history = [
                        item for item in saved_history if isinstance(item, dict)
                    ]
                optimizer_state = checkpoint.get("optimizer_state_dict")
                if isinstance(optimizer_state, dict):
                    optimizer.load_state_dict(optimizer_state)
                    optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] = learning_rate
                else:
                    print(
                        "Checkpoint has weights but no optimizer state; "
                        "continuing with a fresh optimizer.",
                        flush=True,
                    )
                if start_epoch >= epochs:
                    raise ValueError(
                        f"Checkpoint already has {start_epoch} epochs; "
                        f"requested total epochs is {epochs}."
                    )
                print(
                    f"Resuming at epoch {start_epoch + 1} from {resume_path}.",
                    flush=True,
                )
            else:
                print(
                    f"Warm-starting weights from {resume_path} at "
                    f"{checkpoint.get('image_size')} px; the requested "
                    f"{image_size}-px run or changed synthetic-data "
                    "configuration starts a fresh optimizer and loss history.",
                    flush=True,
                )

    if history:
        _write_history(history_path, history)
    ema_decay = 0.995

    def save_checkpoint(epoch: int, latest_validation_loss: float) -> None:
        if best_state is None:
            raise RuntimeError("Training has no valid best checkpoint weights.")
        temporary_path = destination.with_name(
            f"{destination.name}.{os.getpid()}.tmp"
        )
        checkpoint_data = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "architecture": MODEL_ARCHITECTURE,
            "training_data_version": TRAINING_DATA_VERSION,
            "normalization": IMAGE_NORMALIZATION,
            "noise_schedule": NOISE_SCHEDULE,
            "prediction_type": PREDICTION_TYPE,
            "model_state_dict": best_state,
            "best_model_state_dict": best_state,
            "resume_model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "ema_model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in ema_model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_steps": optimizer_steps,
            "image_size": image_size,
            "timesteps": timesteps,
            "base_channels": base_channels,
            "training_sample_count": sample_count,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_epochs": epoch,
            "target_epochs": epochs,
            "warm_start_epochs": warm_start_epochs,
            "seed": seed,
            "validation_velocity_mse": best_validation_loss,
            "latest_validation_velocity_mse": latest_validation_loss,
            "training_loss_history": history,
        }
        try:
            torch.save(checkpoint_data, temporary_path)
            _replace_with_retry(temporary_path, destination)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        print(f"Saved diffusion checkpoint: {destination.resolve()}", flush=True)

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        total_loss = 0.0
        processed = 0
        for clean_images in loader:
            clean_images = clean_images.to(device, non_blocking=True)
            batch_timesteps = torch.randint(
                0,
                timesteps,
                (clean_images.size(0),),
                device=device,
            )
            noise = torch.randn_like(clean_images)
            alpha = alpha_cumulative[batch_timesteps].view(-1, 1, 1, 1)
            noisy_images = alpha.sqrt() * clean_images + (1 - alpha).sqrt() * noise
            velocity = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean_images

            optimizer.zero_grad(set_to_none=True)
            predicted_velocity = model(noisy_images, batch_timesteps)
            loss = functional.mse_loss(predicted_velocity, velocity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer_steps += 1
            current_ema_decay = min(
                ema_decay,
                (1 + optimizer_steps) / (10 + optimizer_steps),
            )
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    ema_model.parameters(),
                    model.parameters(),
                ):
                    ema_parameter.lerp_(parameter, 1 - current_ema_decay)

            total_loss += loss.item() * clean_images.size(0)
            processed += clean_images.size(0)

        ema_model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.inference_mode():
            for clean_images, batch_timesteps, noise in validation_inputs:
                alpha = alpha_cumulative[batch_timesteps].view(-1, 1, 1, 1)
                noisy_images = (
                    alpha.sqrt() * clean_images + (1 - alpha).sqrt() * noise
                )
                velocity = (
                    alpha.sqrt() * noise - (1 - alpha).sqrt() * clean_images
                )
                predicted_velocity = ema_model(noisy_images, batch_timesteps)
                validation_loss = functional.mse_loss(
                    predicted_velocity,
                    velocity,
                    reduction="sum",
                )
                validation_total += validation_loss.item()
                validation_count += noise.numel()
        mean_validation_loss = validation_total / validation_count
        if mean_validation_loss < best_validation_loss:
            best_validation_loss = mean_validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in ema_model.state_dict().items()
            }

        history.append(
            {
                "epoch": epoch,
                "train_velocity_mse": total_loss / processed,
                "validation_velocity_mse": mean_validation_loss,
                "best_validation_velocity_mse": best_validation_loss,
            }
        )
        _write_history(history_path, history)
        print(
            f"Epoch {epoch}/{epochs} - train velocity MSE "
            f"{total_loss / processed:.5f} - validation velocity MSE "
            f"{mean_validation_loss:.5f}",
            flush=True,
        )

        if epoch % checkpoint_interval == 0 or epoch == epochs:
            save_checkpoint(epoch, mean_validation_loss)

        if sample_interval and (
            epoch % sample_interval == 0 or epoch == epochs
        ):
            from models.diffusion_inference import generate_images

            run_samples.mkdir(parents=True, exist_ok=True)
            images = generate_images(
                count=samples_per_checkpoint,
                sampling_steps=timesteps,
                seed=seed + epoch,
                model_bundle={
                    "model": ema_model,
                    "image_size": image_size,
                    "timesteps": timesteps,
                    "device": device,
                },
            )
            for index, image in enumerate(images, start=1):
                image.save(
                    run_samples / f"epoch_{epoch:03d}_sample_{index:02d}.png"
                )
            grid_path = _save_comparison_grid(run_samples)
            print(
                f"Saved {len(images)} labeled samples and comparison grid "
                f"{grid_path}.",
                flush=True,
            )

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train an unconditional diffusion model on procedural, text-free "
            "synthetic form images. Handwritten prescription images and labels "
            "are deliberately excluded to avoid memorizing clinical text."
        )
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the configured checkpoint's optimizer and loss history.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Warm-start from a compatible checkpoint, including a different resolution.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--samples-per-checkpoint", type=int, default=2)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        sample_count=args.samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        timesteps=args.timesteps,
        base_channels=args.base_channels,
        learning_rate=args.learning_rate,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        resume_from=args.resume_from,
        checkpoint_interval=args.checkpoint_interval,
        sample_interval=args.sample_interval,
        samples_per_checkpoint=args.samples_per_checkpoint,
    )


if __name__ == "__main__":
    main()
