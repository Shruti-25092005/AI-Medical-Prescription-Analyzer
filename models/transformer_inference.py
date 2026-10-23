
from pathlib import Path
import os

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = (
    PROJECT_ROOT / "models" / "transformer" / "best_trocr_model"
)

MODEL_ID = os.getenv("HF_MODEL_ID", "sh12rt/prescription-trocr")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print("Loading trained TrOCR model...")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")

    token = os.getenv("HF_TOKEN")

    # In Streamlit, read credentials from app Secrets.
    try:
        import streamlit as st
        token = st.secrets.get("HF_TOKEN", token)
        model_id = st.secrets.get("HF_MODEL_ID", MODEL_ID)
    except Exception:
        model_id = MODEL_ID

    # Download/cache the private repository if needed.
    checkpoint = snapshot_download(
        repo_id=model_id,
        token=token,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
            "pytorch_model.bin",
        ],
    )

    processor = TrOCRProcessor.from_pretrained(
        checkpoint,
        token=token,
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        checkpoint,
        token=token,
    )

    model.to(DEVICE)
    model.eval()

    print("Processor loaded.")
    print("Model loaded.")

    return processor, model


def predict_text(image_path, processor, model):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    return predict_image(image, processor, model)


def predict_image(image, processor, model):
    image = image.convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt",
    ).pixel_values.to(DEVICE)

    with torch.inference_mode():
        generated_ids = model.generate(
            pixel_values,
            num_beams=2,
            max_new_tokens=24,
        )

    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()


def main():
    test_image = (
        PROJECT_ROOT
        / "dataset"
        / "RxHandBD-ML"
        / "RxHandBD-ML"
        / "Test_Set"
        / "P0001.jpg"
    )

    output_file = PROJECT_ROOT / "models" / "transformer_prediction.txt"

    processor, model = load_model()
    prediction = predict_text(test_image, processor, model)

    print("\nTrOCR recognized text:")
    print(prediction if prediction else "(No text recognized)")

    output_file.write_text(prediction + "\n", encoding="utf-8")
    print(f"Result saved to: {output_file}")


if __name__ == "__main__":
    main()
