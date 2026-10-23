import argparse
import os
from pathlib import Path
import time

import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.ae_inference import (
    CHECKPOINT_PATH,
    DATASET_ROOT,
    SkipAutoencoder,
)


IMAGE_SIZE = 512
SEED = 42
EDGE_WEIGHT = 0.15
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 1
CANDIDATE_CHECKPOINT = CHECKPOINT_PATH.with_name("ae_sharp_candidate.pth")


class PrescriptionDataset(Dataset):
    def __init__(self, rows: pd.DataFrame) -> None:
        self.image_paths = rows["image_path"].tolist()
        self.transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.image_paths[index]) as image:
            return self.transform(image.convert("RGB"))


def edge_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    channels = target.shape[1]
    sobel_x = target.new_tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ).div_(8).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    sobel_y = sobel_x.transpose(-1, -2)

    target_x = functional.conv2d(target, sobel_x, padding=1, groups=channels)
    target_y = functional.conv2d(target, sobel_y, padding=1, groups=channels)
    output_x = functional.conv2d(
        reconstruction, sobel_x, padding=1, groups=channels
    )
    output_y = functional.conv2d(
        reconstruction, sobel_y, padding=1, groups=channels
    )
    return 0.5 * (
        functional.l1_loss(output_x, target_x)
        + functional.l1_loss(output_y, target_y)
    )


def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    pixel_loss = functional.l1_loss(reconstruction, target)
    return (1 - EDGE_WEIGHT) * pixel_loss + EDGE_WEIGHT * edge_loss(
        reconstruction, target
    )


def prepare_datasets() -> tuple[DataLoader, DataLoader]:
    label_path = DATASET_ROOT / "Train_Label.csv"
    image_dir = DATASET_ROOT / "Train_Set"
    if not label_path.is_file():
        raise FileNotFoundError(f"Training labels were not found: {label_path}")

    rows = pd.read_csv(label_path)
    if "Images" not in rows.columns:
        raise ValueError(f"Training labels need an Images column: {label_path}")
    rows = rows.assign(
        image_path=rows["Images"].map(lambda name: str(image_dir / str(name)))
    )
    missing = [path for path in rows["image_path"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Training images referenced by the labels are missing, "
            f"including: {missing[:3]}"
        )

    rows = rows.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    validation_size = int(0.1 * len(rows))
    validation_rows = rows.iloc[:validation_size].reset_index(drop=True)
    training_rows = rows.iloc[validation_size:].reset_index(drop=True)

    training_loader = DataLoader(
        PrescriptionDataset(training_rows),
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        PrescriptionDataset(validation_rows),
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    print(
        f"Training images: {len(training_rows)}; "
        f"validation images: {len(validation_rows)}; resolution: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        flush=True,
    )
    return training_loader, validation_loader


def run_epoch(
    model: SkipAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    for batch_index, images in enumerate(loader, start=1):
        images = images.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            reconstruction, _ = model(images)
            loss = reconstruction_loss(reconstruction, images)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        if training and batch_index % 250 == 0:
            print(
                f"  processed {batch_index}/{len(loader)} training images",
                flush=True,
            )

    return total_loss / len(loader.dataset)


def train(epochs: int) -> Path | None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Starting AE checkpoint was not found: {CHECKPOINT_PATH}"
        )
    training_loader, validation_loader = prepare_datasets()
    device = torch.device("cpu")
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    model = SkipAutoencoder().to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-6)

    with torch.inference_mode():
        baseline_validation_loss = run_epoch(
            model, validation_loader, device
        )
    best_validation_loss = baseline_validation_loss
    temporary_checkpoint = CANDIDATE_CHECKPOINT.with_name(
        f"{CANDIDATE_CHECKPOINT.stem}.{os.getpid()}.tmp"
    )
    candidate_saved = False
    print(
        f"Starting checkpoint validation loss: "
        f"{baseline_validation_loss:.6f}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        started_at = time.perf_counter()
        training_loss = run_epoch(model, training_loader, device, optimizer)
        with torch.inference_mode():
            validation_loss = run_epoch(model, validation_loader, device)

        elapsed = time.perf_counter() - started_at
        print(
            f"Epoch {epoch}/{epochs}: train={training_loss:.6f}, "
            f"validation={validation_loss:.6f}, elapsed={elapsed / 60:.1f} min",
            flush=True,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), temporary_checkpoint)
            for attempt in range(10):
                try:
                    os.replace(temporary_checkpoint, CANDIDATE_CHECKPOINT)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(min(0.25 * 2**attempt, 2))
            candidate_saved = True
            print(
                f"Saved best candidate to {CANDIDATE_CHECKPOINT}",
                flush=True,
            )

    if not candidate_saved:
        print(
            "No epoch beat the starting checkpoint on validation loss; "
            "the existing model remains unchanged.",
            flush=True,
        )
        return None
    return CANDIDATE_CHECKPOINT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a separate high-resolution, edge-aware AE candidate."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    candidate_path = train(args.epochs)
    if candidate_path is not None:
        print(f"Training complete. Candidate checkpoint: {candidate_path}", flush=True)


if __name__ == "__main__":
    main()
