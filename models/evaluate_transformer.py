from pathlib import Path
import time

import pandas as pd
import torch
from PIL import Image
from jiwer import cer, wer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "RxHandBD-ML"
    / "RxHandBD-ML"
)

TEST_DIR = DATASET_DIR / "Test_Set"
LABEL_FILE = DATASET_DIR / "Test_Label.csv"

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "transformer"
    / "best_trocr_model"
)

OUTPUT_CSV = PROJECT_ROOT / "models" / "transformer_evaluation.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_BEAMS = 2
MAX_NEW_TOKENS = 24


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print("Loading TrOCR checkpoint...")

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

    print("Model loaded.")
    print("Device:", DEVICE)

    return processor, model


# ============================================================
# EVALUATION
# ============================================================

def main():
    if not LABEL_FILE.is_file():
        raise FileNotFoundError(f"Labels not found: {LABEL_FILE}")

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Test images not found: {TEST_DIR}")

    labels = pd.read_csv(LABEL_FILE)

    required_columns = {"Images", "Text"}
    if not required_columns.issubset(labels.columns):
        raise ValueError(
            f"Expected columns {required_columns}; "
            f"found {set(labels.columns)}"
        )

    labels = labels.dropna(subset=["Images", "Text"]).copy()
    labels["Images"] = labels["Images"].astype(str)
    labels["Text"] = labels["Text"].astype(str)

    processor, model = load_model()

    records = []
    start_all = time.perf_counter()
    skipped = []

    for start in range(0, len(labels), BATCH_SIZE):
        batch = labels.iloc[start:start + BATCH_SIZE]

        images = []
        valid_rows = []

        for _, row in batch.iterrows():
            # Use only a filename, preventing paths from escaping Test_Set.
            filename = Path(row["Images"]).name
            image_path = TEST_DIR / filename

            if not image_path.is_file():
                skipped.append({
                    "image": filename,
                    "reason": "Image file not found",
                })
                continue

            try:
                with Image.open(image_path) as source:
                    images.append(source.convert("RGB"))
                valid_rows.append(row)
            except Exception as exc:
                skipped.append({
                    "image": filename,
                    "reason": f"Image read error: {exc}",
                })

        if not images:
            continue

        pixel_values = processor(
            images=images,
            return_tensors="pt",
        ).pixel_values.to(DEVICE)

        batch_start = time.perf_counter()

        with torch.inference_mode():
            generated_ids = model.generate(
                pixel_values,
                num_beams=NUM_BEAMS,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        predictions = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        elapsed = time.perf_counter() - batch_start
        seconds_per_image = elapsed / len(predictions)

        for row, prediction in zip(valid_rows, predictions):
            reference = row["Text"].strip()
            prediction = prediction.strip()

            records.append({
                "image": Path(row["Images"]).name,
                "reference": reference,
                "prediction": prediction,
                "exact_match": reference.casefold()
                == prediction.casefold(),
                "cer": cer(reference, prediction),
                "wer": wer(reference, prediction),
                "inference_seconds": seconds_per_image,
            })

        completed = len(records)
        print(
            f"Processed {completed}/{len(labels)} "
            f"labelled rows"
        )

    if not records:
        raise RuntimeError(
            "No images were evaluated. Check dataset paths and labels."
        )

    results = pd.DataFrame(records)
    results.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # Corpus-level metrics aggregate errors across all examples.
    references = results["reference"].tolist()
    predictions = results["prediction"].tolist()

    corpus_cer = cer(references, predictions)
    corpus_wer = wer(references, predictions)
    exact_match_rate = results["exact_match"].mean()
    total_seconds = time.perf_counter() - start_all

    print("\n" + "=" * 60)
    print("TrOCR EVALUATION RESULTS")
    print("=" * 60)
    print("Images evaluated:", len(results))
    print("Images skipped:", len(skipped))
    print(f"Corpus CER: {corpus_cer:.4f} ({corpus_cer * 100:.2f}%)")
    print(f"Corpus WER: {corpus_wer:.4f} ({corpus_wer * 100:.2f}%)")
    print(
        f"Case-insensitive exact match: "
        f"{exact_match_rate * 100:.2f}%"
    )
    print(f"Total elapsed time: {total_seconds:.2f} seconds")
    print(f"Average recorded inference time: "
          f"{results['inference_seconds'].mean():.4f} seconds/image")
    print(f"\nDetailed results saved to: {OUTPUT_CSV}")

    if skipped:
        skipped_file = (
            PROJECT_ROOT / "models" / "transformer_skipped_images.csv"
        )
        pd.DataFrame(skipped).to_csv(skipped_file, index=False)
        print(f"Skipped-image report: {skipped_file}")

    print("\nFirst 10 predictions:")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()