import os
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from transformers import (
    PreTrainedTokenizerFast,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = (
    PROJECT_ROOT / "models" / "transformer" / "best_trocr_model"
)
LEGACY_CHECKPOINT_DIR = (
    PROJECT_ROOT / "TrOCR_Checkpoints" / "best_trocr_model"
)
DEPLOY_DIR = PROJECT_ROOT / "trocr_deploy"

MODEL_ID = os.getenv("HF_MODEL_ID", "sh12rt/prescription-trocr")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LocalTrOCRProcessor:
    """Offline-compatible TrOCR Processor wrapping ViTImageProcessor and PreTrainedTokenizerFast."""

    def __init__(self, image_processor: ViTImageProcessor, tokenizer: PreTrainedTokenizerFast):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, images, return_tensors="pt", **kwargs):
        return self.image_processor(images=images, return_tensors=return_tensors, **kwargs)

    def batch_decode(self, sequences, skip_special_tokens=True, **kwargs):
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens, **kwargs)


def _load_local_checkpoint(checkpoint_path: Path):
    print(f"Loading local TrOCR checkpoint from: {checkpoint_path}")
    tok_file = checkpoint_path / "tokenizer.json"
    if not tok_file.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json in {checkpoint_path}")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_file),
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    image_processor = ViTImageProcessor(
        size={"height": 384, "width": 384},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )

    processor = LocalTrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

    model = VisionEncoderDecoderModel.from_pretrained(
        str(checkpoint_path),
        local_files_only=True,
    )
    model.to(DEVICE)
    model.eval()
    return processor, model


def load_model():
    print("Loading trained TrOCR model...")
    print(f"Device: {DEVICE}")

    # 1. Prefer local checkpoint directories
    for candidate in [LEGACY_CHECKPOINT_DIR, CHECKPOINT_DIR, DEPLOY_DIR]:
        if candidate.is_dir() and ((candidate / "model.safetensors").is_file() or (candidate / "pytorch_model.bin").is_file()):
            try:
                return _load_local_checkpoint(candidate)
            except Exception as exc:
                print(f"Failed to load from {candidate}: {exc}. Trying next...")

    # 2. Remote download fallback
    token = os.getenv("HF_TOKEN")
    try:
        import streamlit as st
        token = st.secrets.get("HF_TOKEN", token)
        model_id = st.secrets.get("HF_MODEL_ID", MODEL_ID)
    except Exception:
        model_id = MODEL_ID

    from huggingface_hub import snapshot_download
    from transformers import TrOCRProcessor

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

    processor = TrOCRProcessor.from_pretrained(checkpoint, token=token)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint, token=token)
    model.to(DEVICE)
    model.eval()
    return processor, model


def predict_text(image_path, processor, model):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Test image not found: {image_path}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    return predict_image(image, processor, model)


import numpy as np


def predict_image_detailed(
    image: Image.Image,
    processor,
    model,
    confidence_threshold: float = 0.60,
) -> dict[str, Any]:
    """
    Transcribes an image and calculates true sequence confidence from generation scores.
    Flags uncertain words when model probability is below confidence_threshold or output is degenerate.
    """
    image = image.convert("RGB")
    pixel_values = processor(
        images=image,
        return_tensors="pt",
    ).pixel_values.to(DEVICE)

    with torch.inference_mode():
        outputs = model.generate(
            pixel_values,
            num_beams=2,
            max_new_tokens=24,
            return_dict_in_generate=True,
            output_scores=True,
        )

    text = processor.batch_decode(
        outputs.sequences,
        skip_special_tokens=True,
    )[0].strip()

    confidence = 0.5
    if (
        hasattr(outputs, "sequences_scores")
        and outputs.sequences_scores is not None
        and len(outputs.sequences_scores) > 0
    ):
        seq_score = outputs.sequences_scores[0].item()
        confidence = float(np.exp(np.clip(seq_score, -10.0, 0.0)))
    elif hasattr(outputs, "scores") and outputs.scores:
        step_probs = []
        for step_idx, step_logits in enumerate(outputs.scores):
            probs = torch.softmax(step_logits[0], dim=-1)
            token_id = (
                outputs.sequences[0, step_idx + 1]
                if step_idx + 1 < outputs.sequences.shape[1]
                else torch.argmax(probs)
            )
            step_probs.append(probs[token_id].item())
        if step_probs:
            confidence = float(np.mean(step_probs))

    is_uncertain = (
        confidence < confidence_threshold
        or len(text) == 0
        or (len(text) <= 2 and not text.isalnum())
    )

    return {
        "text": text,
        "confidence": round(confidence, 3),
        "is_uncertain": is_uncertain,
    }


def predict_image(image, processor, model) -> str:
    return predict_image_detailed(image, processor, model)["text"]


def predict_crop_sequence(
    crops: Sequence[Image.Image], processor, model
) -> list[dict[str, Any]]:
    """Transcribes a sequence of cropped word images with confidence scores and uncertainty flags."""
    predictions = []
    for crop in crops:
        pred_info = predict_image_detailed(crop, processor, model)
        predictions.append(pred_info)
    return predictions


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
