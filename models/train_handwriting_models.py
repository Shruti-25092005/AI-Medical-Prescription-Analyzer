"""
Baseline Training Script for Handwriting GAN and DDPM models on RxHandBD-ML dataset.
Produces validated local checkpoints for interactive Streamlit demonstration.
"""

from pathlib import Path
import time
from models.handwriting_gan import train_handwriting_gan
from models.handwriting_diffusion import train_handwriting_diffusion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "RxHandBD-ML" / "RxHandBD-ML"
GAN_CKPT = PROJECT_ROOT / "models" / "handwriting_gan.pth"
DIFF_CKPT = PROJECT_ROOT / "models" / "handwriting_diffusion.pth"


def main():
    print("=== Training Baseline Handwriting GAN ===")
    t0 = time.time()
    gan_history = train_handwriting_gan(
        dataset_dir=DATASET_DIR,
        epochs=4,
        batch_size=16,
        lr_g=0.0002,
        lr_d=0.0001,
        max_samples=256,
        checkpoint_path=GAN_CKPT,
    )
    print(f"GAN Training finished in {time.time() - t0:.2f}s. Checkpoint: {GAN_CKPT}")

    print("\n=== Training Baseline Handwriting DDPM Diffusion ===")
    t1 = time.time()
    diff_history = train_handwriting_diffusion(
        dataset_dir=DATASET_DIR,
        epochs=3,
        batch_size=8,
        lr=0.0003,
        timesteps=50,
        max_samples=128,
        checkpoint_path=DIFF_CKPT,
    )
    print(f"Diffusion Training finished in {time.time() - t1:.2f}s. Checkpoint: {DIFF_CKPT}")
    print("\nAll baseline models trained and saved successfully.")


if __name__ == "__main__":
    main()
