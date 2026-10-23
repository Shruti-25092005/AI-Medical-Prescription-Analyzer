from dataclasses import dataclass
from pathlib import Path
import time

import pandas as pd
import streamlit as st
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "RxHandBD-ML"
    / "RxHandBD-ML"
)
EVALUATION_CSV = MODELS_DIR / "transformer_evaluation.csv"
MODEL_NAMES = {
    "trocr": "TrOCR - handwriting recognition",
    "ae": "AE - image reconstruction",
    "vae": "VAE - image reconstruction",
    "gan": "GAN - synthetic sample generation",
}


@dataclass
class InferenceResult:
    recognized_text: str | None = None
    reconstruction: Image.Image | None = None
    inference_seconds: float = 0.0
    threshold_preview: Image.Image | None = None


def model_statuses() -> list[dict[str, str | bool]]:
    checkpoints = {
        "ae": MODELS_DIR / "ae_best_model.pth",
        "vae": MODELS_DIR / "final_sharper_vae_ema.pth",
        "trocr": MODELS_DIR / "transformer" / "best_trocr_model",
        "gan": MODELS_DIR / "synthetic_record_gan.pth",
    }
    descriptions = {
        "ae": "Reconstructs an RGB image through a skip autoencoder.",
        "vae": "Encodes and reconstructs a grayscale image using the VAE mean.",
        "trocr": "Recognizes handwritten text from prescription images.",
        "gan": "Generates clearly labeled fictional sample cards.",
    }
    statuses = []
    for model_id in ("ae", "vae", "trocr", "gan"):
        checkpoint = checkpoints[model_id]
        if model_id in {"ae", "vae", "gan"}:
            available = checkpoint.is_file()
        else:
            available = (
                checkpoint.is_dir()
                and (checkpoint / "config.json").is_file()
                and (
                    (checkpoint / "model.safetensors").is_file()
                    or (checkpoint / "pytorch_model.bin").is_file()
                )
            )
        statuses.append({
            "id": model_id,
            "name": MODEL_NAMES[model_id],
            "purpose": descriptions[model_id],
            "checkpoint": str(checkpoint),
            "available": available,
        })
    return statuses


@st.cache_resource
def load_model_bundle(model_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_id == "trocr":
        from models.transformer_inference import load_model

        processor, model = load_model()
        return {
            "processor": processor,
            "model": model,
            "device": device,
        }

    if model_id == "ae":
        from models.ae_inference import load_autoencoder

        model, device = load_autoencoder(device)
        return {"model": model, "device": device}

    if model_id == "vae":
        from models.vae_inference import load_vae_model

        model, _, image_size, device = load_vae_model(device)
        return {
            "model": model,
            "device": device,
            "image_size": image_size,
        }

    raise ValueError(f"Unknown model selection: {model_id}")


def run_inference(
    model_id: str,
    image: Image.Image,
    model_bundle: dict,
) -> InferenceResult:
    started_at = time.perf_counter()

    if model_id == "trocr":
        from models.transformer_inference import predict_image

        recognized_text = predict_image(
            image,
            model_bundle["processor"],
            model_bundle["model"],
        )
        return InferenceResult(
            recognized_text=recognized_text,
            inference_seconds=time.perf_counter() - started_at,
        )

    if model_id == "ae":
        from models.ae_inference import (
            adaptive_threshold_preview,
            enhance_reconstruction,
            reconstruct_image,
        )

        raw_reconstruction = reconstruct_image(
            image,
            model_bundle["model"],
            model_bundle["device"],
        )
        reconstruction = enhance_reconstruction(raw_reconstruction)
        threshold_preview = adaptive_threshold_preview(reconstruction)
    elif model_id == "vae":
        from models.vae_inference import reconstruct_image

        reconstruction = reconstruct_image(
            image,
            model_bundle["model"],
            model_bundle["image_size"],
            model_bundle["device"],
        )
        threshold_preview = None
    else:
        raise ValueError(f"Unknown model selection: {model_id}")

    return InferenceResult(
        reconstruction=reconstruction,
        threshold_preview=threshold_preview,
        inference_seconds=time.perf_counter() - started_at,
    )


def evaluation_data() -> pd.DataFrame:
    if not EVALUATION_CSV.is_file():
        raise FileNotFoundError(
            f"Existing TrOCR evaluation results were not found: {EVALUATION_CSV}"
        )

    data = pd.read_csv(EVALUATION_CSV)
    required = {
        "image",
        "reference",
        "prediction",
        "exact_match",
        "cer",
        "wer",
        "inference_seconds",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"Evaluation CSV is missing required columns: {sorted(missing)}"
        )
    if data.empty:
        raise ValueError("The existing TrOCR evaluation CSV contains no rows.")
    return data


def evaluation_metrics(data: pd.DataFrame) -> dict[str, float | int]:
    from jiwer import cer, wer

    references = data["reference"].fillna("").astype(str).tolist()
    predictions = data["prediction"].fillna("").astype(str).tolist()
    return {
        "samples": len(data),
        "exact_match_percent": float(data["exact_match"].astype(float).mean() * 100),
        "mean_cer": float(data["cer"].astype(float).mean()),
        "mean_wer": float(data["wer"].astype(float).mean()),
        "corpus_cer": float(cer(references, predictions)),
        "corpus_wer": float(wer(references, predictions)),
        "mean_inference_seconds": float(
            data["inference_seconds"].astype(float).mean()
        ),
    }


def model_evaluation_matrix(metrics: dict[str, float | int]) -> pd.DataFrame:
    rows = []
    for model in model_statuses():
        measured = model["id"] == "trocr"
        task = (
            "Handwriting OCR"
            if measured
            else (
                "Synthetic record generation"
                if model["id"] == "gan"
                else "Image reconstruction"
            )
        )
        rows.append({
            "Model": model["name"],
            "Task": task,
            "Exact match": (
                f"{metrics['exact_match_percent']:.2f}%"
                if measured
                else "Not measured"
            ),
            "Mean CER": f"{metrics['mean_cer']:.2%}" if measured else "Not measured",
            "Mean WER": f"{metrics['mean_wer']:.2%}" if measured else "Not measured",
            "Inference time": (
                f"{metrics['mean_inference_seconds']:.2f} s/image"
                if measured
                else "Not measured"
            ),
        })
    return pd.DataFrame(rows)


def dataset_counts() -> dict[str, int | None]:
    label_paths = {
        "train": DATASET_DIR / "Train_Label.csv",
        "test": DATASET_DIR / "Test_Label.csv",
    }
    counts: dict[str, int | None] = {}
    for split, label_path in label_paths.items():
        counts[split] = len(pd.read_csv(label_path)) if label_path.is_file() else None
    return counts


def session_statistics() -> dict[str, int | float]:
    defaults: dict[str, int | float] = {
        "runs": 0,
        "pages": 0,
        "successful_pages": 0,
        "failed_pages": 0,
        "inference_seconds": 0.0,
    }
    if "processing_statistics" not in st.session_state:
        st.session_state["processing_statistics"] = defaults.copy()
    return st.session_state["processing_statistics"]


def record_processing_run(
    page_count: int,
    successful_pages: int,
    failed_pages: int,
    inference_seconds: float,
) -> None:
    stats = session_statistics()
    stats["runs"] += 1
    stats["pages"] += page_count
    stats["successful_pages"] += successful_pages
    stats["failed_pages"] += failed_pages
    stats["inference_seconds"] += inference_seconds
