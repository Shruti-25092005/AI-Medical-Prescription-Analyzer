"""
Evaluation Metrics Module for Reconstruction, OCR, and Generative Image Models.
Computes MSE, MAE, PSNR, SSIM, CER, WER, Exact Match, Generative Diversity,
and Training-set Nearest-Neighbor Distances with full numerical precision.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


def compute_sharpness(image: Image.Image | np.ndarray) -> float:
    """Computes Tenengrad/variance of Laplacian sharpness metric."""
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.float32)
    else:
        arr = image if image.ndim == 2 else cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(arr, cv2.CV_32F)
    return float(np.var(lap))


def compute_contrast(image: Image.Image | np.ndarray) -> float:
    """Computes RMS contrast (standard deviation of pixel luminance)."""
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.float32)
    else:
        arr = image if image.ndim == 2 else cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.std(arr))


def compute_reconstruction_error_map(
    original: Image.Image | np.ndarray,
    reconstructed: Image.Image | np.ndarray,
) -> tuple[Image.Image, dict[str, float]]:
    """
    Computes spatial reconstruction difference map highlighting lost details.
    Returns:
        error_map_image: PIL Image with color-mapped residual errors.
        detail_metrics: Dict with edge preservation and detail loss metrics.
    """
    if isinstance(original, Image.Image):
        orig_gray = np.asarray(original.convert("L"), dtype=np.float32)
    else:
        orig_gray = original if original.ndim == 2 else cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    if isinstance(reconstructed, Image.Image):
        recon_gray = np.asarray(reconstructed.convert("L"), dtype=np.float32)
    else:
        recon_gray = reconstructed if reconstructed.ndim == 2 else cv2.cvtColor(reconstructed.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    if orig_gray.shape != recon_gray.shape:
        h, w = orig_gray.shape[:2]
        recon_gray = cv2.resize(recon_gray, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Absolute difference map
    abs_diff = np.abs(orig_gray - recon_gray)
    norm_diff = np.clip(abs_diff * 3.5, 0, 255).astype(np.uint8)
    # Apply JET colormap for clear visual inspection of lost details
    heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    error_map_img = Image.fromarray(heatmap_rgb)

    sharp_orig = compute_sharpness(orig_gray)
    sharp_recon = compute_sharpness(recon_gray)
    contrast_orig = compute_contrast(orig_gray)
    contrast_recon = compute_contrast(recon_gray)

    # High-frequency stroke gradient loss
    grad_orig = cv2.Sobel(orig_gray, cv2.CV_32F, 1, 1, ksize=3)
    grad_recon = cv2.Sobel(recon_gray, cv2.CV_32F, 1, 1, ksize=3)
    edge_energy_orig = float(np.mean(np.abs(grad_orig)))
    edge_energy_recon = float(np.mean(np.abs(grad_recon)))
    edge_loss_pct = ((edge_energy_orig - edge_energy_recon) / max(edge_energy_orig, 1e-5)) * 100.0

    sharpness_change_pct = ((sharp_recon - sharp_orig) / max(sharp_orig, 1e-5)) * 100.0

    detail_metrics = {
        "mean_error_intensity": round(float(np.mean(abs_diff)), 2),
        "max_error_intensity": round(float(np.max(abs_diff)), 2),
        "sharpness_orig": round(sharp_orig, 2),
        "sharpness_recon": round(sharp_recon, 2),
        "sharpness_change_pct": round(sharpness_change_pct, 1),
        "contrast_orig": round(contrast_orig, 2),
        "contrast_recon": round(contrast_recon, 2),
        "edge_loss_pct": round(max(0.0, edge_loss_pct), 1),
    }

    return error_map_img, detail_metrics


def compute_reconstruction_metrics(
    original: Image.Image | np.ndarray,
    reconstructed: Image.Image | np.ndarray,
) -> dict[str, float]:
    """
    Computes image reconstruction fidelity metrics: MSE, MAE, PSNR (dB), and SSIM.
    Accepts PIL Images or numpy arrays.
    """
    if isinstance(original, Image.Image):
        orig_arr = np.asarray(original.convert("RGB"), dtype=np.float32)
    else:
        orig_arr = np.asarray(original, dtype=np.float32)

    if isinstance(reconstructed, Image.Image):
        recon_arr = np.asarray(reconstructed.convert("RGB"), dtype=np.float32)
    else:
        recon_arr = np.asarray(reconstructed, dtype=np.float32)

    # Match dimensions if needed
    if orig_arr.shape != recon_arr.shape:
        h, w = orig_arr.shape[:2]
        recon_arr = cv2.resize(recon_arr, (w, h), interpolation=cv2.INTER_LANCZOS4)
        if len(orig_arr.shape) == 3 and len(recon_arr.shape) == 2:
            recon_arr = np.stack([recon_arr] * 3, axis=-1)

    diff = orig_arr - recon_arr
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))

    if mse == 0.0:
        psnr = 100.0
    else:
        max_pixel = 255.0
        psnr = float(20.0 * math.log10(max_pixel / math.sqrt(mse)))

    # Compute Structural Similarity Index (SSIM)
    # Using luminance channel for accurate perceptual structural evaluation
    if orig_arr.ndim == 3:
        orig_gray = cv2.cvtColor(orig_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        recon_gray = cv2.cvtColor(recon_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        orig_gray = orig_arr
        recon_gray = recon_arr

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(orig_gray, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(recon_gray, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(orig_gray ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(recon_gray ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(orig_gray * recon_gray, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    ssim = float(np.mean(ssim_map))

    sharp_orig = compute_sharpness(orig_gray)
    sharp_recon = compute_sharpness(recon_gray)
    contrast_orig = compute_contrast(orig_gray)
    contrast_recon = compute_contrast(recon_gray)
    sharpness_change_pct = ((sharp_recon - sharp_orig) / max(sharp_orig, 1e-5)) * 100.0

    return {
        "mse": round(mse, 4),
        "mae": round(mae, 4),
        "psnr_db": round(psnr, 2),
        "ssim": round(ssim, 4),
        "sharpness_orig": round(sharp_orig, 2),
        "sharpness_recon": round(sharp_recon, 2),
        "sharpness_change_pct": round(sharpness_change_pct, 1),
        "contrast_orig": round(contrast_orig, 2),
        "contrast_recon": round(contrast_recon, 2),
    }


def compute_ocr_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    """
    Evaluates OCR predictions against ground truth references.
    Computes Character Error Rate (CER), Word Error Rate (WER), and Exact Match Accuracy.
    """
    if not predictions or not references or len(predictions) != len(references):
        return {"exact_match_acc": 0.0, "mean_cer": 0.0, "mean_wer": 0.0, "sample_count": 0}

    import jiwer

    exact_matches = 0
    total_samples = len(predictions)

    clean_preds = [str(p).strip().lower() for p in predictions]
    clean_refs = [str(r).strip().lower() for r in references]

    for p, r in zip(clean_preds, clean_refs):
        if p == r:
            exact_matches += 1

    exact_match_acc = (exact_matches / total_samples) * 100.0

    # Individual CER & WER
    cers = []
    wers = []
    for p, r in zip(clean_preds, clean_refs):
        try:
            c = jiwer.cer(r, p)
            w = jiwer.wer(r, p)
        except Exception:
            c, w = 1.0, 1.0
        cers.append(c)
        wers.append(w)

    return {
        "sample_count": total_samples,
        "exact_match_acc": round(exact_match_acc, 2),
        "mean_cer": round(float(np.mean(cers)), 4),
        "mean_wer": round(float(np.mean(wers)), 4),
        "corpus_cer": round(float(jiwer.cer(clean_refs, clean_preds)), 4),
        "corpus_wer": round(float(jiwer.wer(clean_refs, clean_preds)), 4),
    }


def compute_generative_diversity(images: Sequence[Image.Image]) -> float:
    """
    Computes pairwise pixel L1 diversity across generated samples in a batch.
    A higher diversity score indicates less mode collapse.
    """
    if len(images) < 2:
        return 0.0

    arrays = [np.asarray(img.convert("L"), dtype=np.float32) / 255.0 for img in images]
    # Resize to uniform shape
    target_shape = (64, 64)
    resized = [cv2.resize(a, target_shape) for a in arrays]

    distances = []
    for i in range(len(resized)):
        for j in range(i + 1, len(resized)):
            l1_dist = float(np.mean(np.abs(resized[i] - resized[j])))
            distances.append(l1_dist)

    return round(float(np.mean(distances)), 4) if distances else 0.0


def compute_nearest_neighbor_distance(
    generated_images: Sequence[Image.Image],
    training_sample_images: Sequence[Image.Image],
) -> dict[str, float]:
    """
    Evaluates whether generated samples are duplicates of training data or novel synthetic handwriting.
    Computes minimum Euclidean distance from each generated sample to the nearest training sample.
    """
    if not generated_images or not training_sample_images:
        return {"min_nn_distance": 0.0, "mean_nn_distance": 0.0, "memorization_detected": False}

    target_shape = (32, 32)
    gen_feats = [
        cv2.resize(np.asarray(img.convert("L"), dtype=np.float32) / 255.0, target_shape).flatten()
        for img in generated_images
    ]
    train_feats = [
        cv2.resize(np.asarray(img.convert("L"), dtype=np.float32) / 255.0, target_shape).flatten()
        for img in training_sample_images[:100]  # Representative sample
    ]

    min_distances = []
    for g in gen_feats:
        dists = [float(np.linalg.norm(g - t)) for t in train_feats]
        min_distances.append(min(dists))

    min_dist = float(np.min(min_distances))
    mean_dist = float(np.mean(min_distances))

    # A distance near 0.0 (< 0.5) would suggest direct copying/memorization
    memorization = min_dist < 0.5

    return {
        "min_nn_distance": round(min_dist, 3),
        "mean_nn_distance": round(mean_dist, 3),
        "memorization_detected": memorization,
    }


def compute_foreground_coverage(
    image: Image.Image | np.ndarray,
    paper_threshold: float = 235.0,
) -> float:
    """
    Computes percentage of canvas occupied by dark foreground strokes/content.
    Clean handwriting crops typically range from 4% to 25%.
    Values < 0.5% indicate blank or faded canvas; values > 65% indicate heavy noise or solid black.
    """
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.float32)
    else:
        arr = image if image.ndim == 2 else cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    foreground_mask = arr < paper_threshold
    coverage = (float(np.sum(foreground_mask)) / float(arr.size)) * 100.0
    return round(coverage, 2)


def evaluate_image_quality(image: Image.Image | np.ndarray) -> dict[str, float]:
    """
    Evaluates individual image quality without reference:
    sharpness, RMS contrast, brightness mean, dynamic range, and foreground coverage.
    """
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.float32)
    else:
        arr = image if image.ndim == 2 else cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    sharp = compute_sharpness(arr)
    contrast = compute_contrast(arr)
    mean_val = float(np.mean(arr))
    dyn_range = float(np.max(arr) - np.min(arr))
    coverage = compute_foreground_coverage(arr)

    return {
        "sharpness": round(sharp, 2),
        "contrast": round(contrast, 2),
        "brightness_mean": round(mean_val, 2),
        "dynamic_range": round(dyn_range, 2),
        "foreground_coverage_pct": round(coverage, 2),
    }


def detect_generation_failure(images: Sequence[Image.Image]) -> dict[str, Any]:
    """
    Detects degenerative generative failures:
    - Blank or nearly blank outputs (extremely low foreground coverage or low contrast)
    - Excessively saturated or noisy outputs (excessive foreground coverage)
    - Mode collapse / repetitive outputs (very low pairwise L1 diversity)
    - Severe blur / stroke dissolution (extremely low sharpness)
    - Malformed or flat dynamic range
    """
    if not images:
        return {
            "is_failure": True,
            "failure_reasons": ["No images provided for quality assessment."],
            "diversity": 0.0,
            "mean_sharpness": 0.0,
            "mean_contrast": 0.0,
            "mean_foreground_coverage": 0.0,
            "status": "Failed: Empty batch",
        }

    diversity = compute_generative_diversity(images)
    qualities = [evaluate_image_quality(img) for img in images]
    mean_sharp = float(np.mean([q["sharpness"] for q in qualities]))
    mean_contrast = float(np.mean([q["contrast"] for q in qualities]))
    mean_dyn_range = float(np.mean([q["dynamic_range"] for q in qualities]))
    mean_coverage = float(np.mean([q["foreground_coverage_pct"] for q in qualities]))

    is_document = bool(images and (images[0].height >= 500 or images[0].width >= 500))
    div_threshold = 0.012 if is_document else 0.035

    failure_reasons = []
    if len(images) >= 2 and diversity < div_threshold:
        failure_reasons.append(f"Repetitive output / Mode collapse detected (diversity={diversity:.4f} < {div_threshold}).")
    if mean_sharp < 4.0:
        failure_reasons.append(f"Severe blur / low stroke definition detected (sharpness={mean_sharp:.1f} < 4.0).")
    if mean_contrast < 8.0:
        failure_reasons.append(f"Low contrast output detected (contrast={mean_contrast:.1f} < 8.0).")
    if mean_dyn_range < 25.0:
        failure_reasons.append(f"Extremely compressed dynamic range detected ({mean_dyn_range:.1f} < 25.0).")
    if mean_coverage < 0.5:
        failure_reasons.append(f"Blank or nearly blank output detected (foreground coverage={mean_coverage:.1f}% < 0.5%).")
    elif mean_coverage > 65.0:
        failure_reasons.append(f"Excessively saturated / noisy output detected (foreground coverage={mean_coverage:.1f}% > 65.0%).")

    is_failure = len(failure_reasons) > 0
    status = "Quality Verified" if not is_failure else f"Uncertain / Generation Limitations: {len(failure_reasons)} issue(s)"

    return {
        "is_failure": is_failure,
        "failure_reasons": failure_reasons,
        "diversity": diversity,
        "mean_sharpness": round(mean_sharp, 2),
        "mean_contrast": round(mean_contrast, 2),
        "mean_dynamic_range": round(mean_dyn_range, 2),
        "mean_foreground_coverage": round(mean_coverage, 2),
        "status": status,
    }
