"""
Prescription Page Processing, Preprocessing, and Text Line/Word Segmentation.
Prepares full-page prescription scans and images for word-level TrOCR recognition.
Includes orientation correction, denoising, contrast enhancement (CLAHE), adaptive thresholding,
contour-based word/line segmentation, reading-order sorting, and visual bounding box overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass
class SegmentedCrop:
    crop_id: int
    line_index: int
    box: tuple[int, int, int, int]  # (x, y, w, h)
    image: Image.Image
    area: int


@dataclass
class SegmentationResult:
    is_single_crop: bool
    original_image: Image.Image
    preprocessed_grayscale: Image.Image
    enhanced_contrast: Image.Image
    threshold_preview: Image.Image
    annotated_page: Image.Image
    crops: list[SegmentedCrop]
    line_count: int
    crop_count: int
    notes: str


def preprocess_image_pipeline(
    pil_image: Image.Image,
    apply_clahe: bool = True,
    apply_denoise: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes standard computer vision preprocessing on an input PIL image:
    1. EXIF orientation correction
    2. RGB to Grayscale
    3. Bilateral / Gaussian Denoising
    4. Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    5. Adaptive Gaussian Thresholding / Otsu Binarization
    """
    oriented = ImageOps.exif_transpose(pil_image).convert("RGB")
    rgb = np.asarray(oriented)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 1. Denoise
    if apply_denoise:
        denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    else:
        denoised = gray

    # 2. Contrast Enhancement (CLAHE)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
    else:
        enhanced = denoised

    # 3. Binarization
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=10,
    )

    return gray, enhanced, binary


def segment_prescription_page(
    pil_image: Image.Image,
    min_area: int = 400,
    max_area_ratio: float = 0.85,
    pad_px: int = 8,
) -> SegmentationResult:
    """
    Analyzes an uploaded prescription page. If the input is already a single cropped word
    (e.g., RxHandBD-ML dimensions ~512x512 with high aspect ratio or single stroke cluster),
    it wraps it cleanly. If it is a full prescription page, it segments text lines and words
    using morphological dilation and connected component analysis.
    """
    oriented = ImageOps.exif_transpose(pil_image).convert("RGB")
    width, height = oriented.size

    gray, enhanced, binary = preprocess_image_pipeline(oriented)

    # For both document scans and crops, perform horizontal line dilation to detect text regions
    # Morphological kernel tuned to bridge letter gaps within words and lines
    kernel_w = max(10, width // 45)
    kernel_h = max(3, height // 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    page_area = width * height
    min_box_area = max(100, int(page_area * 0.0006))

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Filter out tiny noise specks or page border bounding boxes
        if area < min_box_area:
            continue
        if area > page_area * max_area_ratio:
            continue
        # Text line aspect ratio filter: avoid huge empty rectangles
        if h < 8 or w < 10:
            continue

        boxes.append((x, y, w, h))

    # If no contours detected (e.g. faint ink), fall back to whole image
    if not boxes:
        fallback_crop = SegmentedCrop(
            crop_id=1,
            line_index=1,
            box=(0, 0, width, height),
            image=oriented.copy(),
            area=width * height,
        )
        return SegmentationResult(
            is_single_crop=True,
            original_image=oriented,
            preprocessed_grayscale=Image.fromarray(gray),
            enhanced_contrast=Image.fromarray(enhanced),
            threshold_preview=Image.fromarray(255 - binary),
            annotated_page=oriented.copy(),
            crops=[fallback_crop],
            line_count=1,
            crop_count=1,
            notes="No distinct text contours segmented; falling back to full-image processing.",
        )

    if len(boxes) == 1:
        bx, by, bw, bh = boxes[0]
        px1 = max(0, bx - pad_px)
        py1 = max(0, by - pad_px)
        px2 = min(width, bx + bw + pad_px)
        py2 = min(height, by + bh + pad_px)
        single_crop = SegmentedCrop(
            crop_id=1,
            line_index=1,
            box=boxes[0],
            image=oriented.crop((px1, py1, px2, py2)),
            area=bw * bh,
        )
        return SegmentationResult(
            is_single_crop=True,
            original_image=oriented,
            preprocessed_grayscale=Image.fromarray(gray),
            enhanced_contrast=Image.fromarray(enhanced),
            threshold_preview=Image.fromarray(255 - binary),
            annotated_page=oriented.copy(),
            crops=[single_crop],
            line_count=1,
            crop_count=1,
            notes="Single handwriting word crop detected.",
        )

    # Sort boxes into lines top-to-bottom, then left-to-right within line
    # Sort primarily by y
    boxes = sorted(boxes, key=lambda b: b[1])

    # Group into lines if vertical distance is within tolerance
    lines: list[list[tuple[int, int, int, int]]] = []
    current_line: list[tuple[int, int, int, int]] = []
    current_y = boxes[0][1]
    line_height_threshold = max(20, int(np.mean([b[3] for b in boxes]) * 0.75))

    for box in boxes:
        if abs(box[1] - current_y) < line_height_threshold:
            current_line.append(box)
        else:
            # Sort previous line left-to-right
            current_line.sort(key=lambda b: b[0])
            lines.append(current_line)
            current_line = [box]
            current_y = box[1]

    if current_line:
        current_line.sort(key=lambda b: b[0])
        lines.append(current_line)

    # Build SegmentedCrop list and annotate page
    crops: list[SegmentedCrop] = []
    annotated = oriented.copy()
    draw = ImageDraw.Draw(annotated)

    crop_counter = 1
    for line_idx, line_boxes in enumerate(lines, start=1):
        for box in line_boxes:
            bx, by, bw, bh = box
            # Apply padding
            px1 = max(0, bx - pad_px)
            py1 = max(0, by - pad_px)
            px2 = min(width, bx + bw + pad_px)
            py2 = min(height, by + bh + pad_px)

            cropped_pil = oriented.crop((px1, py1, px2, py2))

            crop = SegmentedCrop(
                crop_id=crop_counter,
                line_index=line_idx,
                box=box,
                image=cropped_pil,
                area=bw * bh,
            )
            crops.append(crop)

            # Draw visual bounding box and ID tag
            draw.rectangle([(bx, by), (bx + bw, by + bh)], outline="#167D78", width=3)
            draw.rectangle([(bx, max(0, by - 16)), (bx + 26, by)], fill="#167D78")
            draw.text((bx + 3, max(0, by - 16)), str(crop_counter), fill="white")

            crop_counter += 1

    return SegmentationResult(
        is_single_crop=False,
        original_image=oriented,
        preprocessed_grayscale=Image.fromarray(gray),
        enhanced_contrast=Image.fromarray(enhanced),
        threshold_preview=Image.fromarray(255 - binary),
        annotated_page=annotated,
        crops=crops,
        line_count=len(lines),
        crop_count=len(crops),
        notes=(
            f"Successfully segmented page into {len(lines)} line(s) and {len(crops)} text crop(s). "
            "Crops are ordered top-to-bottom and left-to-right to preserve prescription reading order."
        ),
    )
