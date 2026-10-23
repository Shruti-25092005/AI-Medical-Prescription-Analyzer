from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "transformer"
    / "best_trocr_model"
)

TEST_IMAGE = (
    PROJECT_ROOT
    / "dataset"
    / "RxHandBD-ML"
    / "RxHandBD-ML"
    / "Test_Set"
    / "P0001.jpg"
)

OUTPUT_FILE = PROJECT_ROOT / "models" / "transformer_prediction.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD MODEL AND PROCESSOR
# ============================================================

def load_model():
    print("Loading trained TrOCR model...")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"Device: {DEVICE}")

    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found:\n{CHECKPOINT_DIR}"
        )

    if not (CHECKPOINT_DIR / "config.json").is_file():
        raise FileNotFoundError("Checkpoint is missing config.json.")

    weight_files = [
        CHECKPOINT_DIR / "model.safetensors",
        CHECKPOINT_DIR / "pytorch_model.bin",
    ]

    if not any(path.is_file() for path in weight_files):
        raise FileNotFoundError("No model weights found in checkpoint.")

    # Load only from the local checkpoint.
    processor = TrOCRProcessor.from_pretrained(
        str(CHECKPOINT_DIR),
        local_files_only=True,
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        str(CHECKPOINT_DIR),
        local_files_only=True,
    )

    model.to(DEVICE)
    model.eval()

    print("Processor loaded.")
    print("Model loaded.")

    return processor, model


# ============================================================
# OCR PREDICTION
# ============================================================

def predict_text(image_path, processor, model):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Test image not found:\n{image_path}"
        )

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    return predict_image(image, processor, model)


def predict_image(image, processor, model):
    image = image.convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt",
    ).pixel_values.to(DEVICE)

    print("Input tensor shape:", tuple(pixel_values.shape))

    with torch.inference_mode():
        generated_ids = model.generate(
            pixel_values,
            num_beams=2,
            max_new_tokens=24,
        )

    prediction = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    return prediction


# ============================================================
# MAIN
# ============================================================

def main():
    processor, model = load_model()

    print(f"\nTesting image: {TEST_IMAGE.name}")

    prediction = predict_text(
        TEST_IMAGE,
        processor,
        model,
    )

    print("\n" + "=" * 50)
    print("TrOCR RECOGNIZED TEXT")
    print("=" * 50)
    print(prediction if prediction else "(No text recognized)")
    print("=" * 50)

    OUTPUT_FILE.write_text(
        prediction + "\n",
        encoding="utf-8",
    )

    print(f"\nResult saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()