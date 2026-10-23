import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import tempfile
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from jiwer import cer, wer
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "transformer" / "best_trocr_model"
DATASET_DIR = PROJECT_ROOT / "dataset" / "RxHandBD-ML" / "RxHandBD-ML"
EVALUATION_CSV = MODELS_DIR / "transformer_evaluation.csv"
VALIDATION_CSV = CHECKPOINT_DIR / "val.csv"
OUTPUT_ROOT = MODELS_DIR / "transformer_error_analysis"
BATCH_SIZE = 4


def _display_character(character: str) -> str:
    return "<space>" if character.isspace() else character


def _character_edits(reference: str, prediction: str) -> list[dict[str, str]]:
    reference = reference.casefold()
    prediction = prediction.casefold()
    rows = len(reference) + 1
    columns = len(prediction) + 1
    distances = [[0] * columns for _ in range(rows)]

    for i in range(rows):
        distances[i][0] = i
    for j in range(columns):
        distances[0][j] = j

    for i in range(1, rows):
        for j in range(1, columns):
            substitution_cost = reference[i - 1] != prediction[j - 1]
            distances[i][j] = min(
                distances[i - 1][j] + 1,
                distances[i][j - 1] + 1,
                distances[i - 1][j - 1] + substitution_cost,
            )

    edits = []
    i = len(reference)
    j = len(prediction)
    while i or j:
        if (
            i
            and j
            and reference[i - 1] == prediction[j - 1]
            and distances[i][j] == distances[i - 1][j - 1]
        ):
            i -= 1
            j -= 1
        elif (
            i
            and j
            and reference[i - 1] != prediction[j - 1]
            and distances[i][j] == distances[i - 1][j - 1] + 1
        ):
            edits.append({
                "type": "substitution",
                "reference_character": reference[i - 1],
                "prediction_character": prediction[j - 1],
            })
            i -= 1
            j -= 1
        elif i and distances[i][j] == distances[i - 1][j] + 1:
            edits.append({
                "type": "missing",
                "reference_character": reference[i - 1],
                "prediction_character": "",
            })
            i -= 1
        else:
            edits.append({
                "type": "extra",
                "reference_character": "",
                "prediction_character": prediction[j - 1],
            })
            j -= 1

    edits.reverse()
    return edits


def _error_records(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    records = []
    totals = Counter()
    substitutions = Counter()
    missing_characters = Counter()
    extra_characters = Counter()

    for row in data.itertuples(index=False):
        filename = str(row.image)
        reference = str(row.reference).strip()
        prediction = str(row.prediction).strip()
        casefold_exact = reference.casefold() == prediction.casefold()
        raw_text_differs = reference != prediction
        if not raw_text_differs:
            continue

        edits = _character_edits(reference, prediction)
        groups = sorted({edit["type"] for edit in edits})
        if not groups:
            groups = ["case_only"]
        for edit in edits:
            if edit["type"] == "substitution":
                key = (
                    f"{_display_character(edit['reference_character'])}"
                    f" -> "
                    f"{_display_character(edit['prediction_character'])}"
                )
                substitutions[key] += 1
            elif edit["type"] == "missing":
                missing_characters[
                    _display_character(edit["reference_character"])
                ] += 1
            elif edit["type"] == "extra":
                extra_characters[
                    _display_character(edit["prediction_character"])
                ] += 1

        if casefold_exact:
            error_group = "case_only"
            totals["case_only_differences"] += 1
        else:
            error_group = (
                groups[0] if len(groups) == 1 else "mixed"
            )
            totals["non_case_exact_mismatches"] += 1
            totals[f"rows_with_{error_group}"] += 1
            totals["substitutions"] += sum(
                edit["type"] == "substitution" for edit in edits
            )
            totals["missing_characters"] += sum(
                edit["type"] == "missing" for edit in edits
            )
            totals["extra_characters"] += sum(
                edit["type"] == "extra" for edit in edits
            )

            reference_words = set(reference.casefold().split())
            prediction_words = set(prediction.casefold().split())
            if reference_words and not reference_words.intersection(prediction_words):
                totals["no_exact_word_token_shared"] += 1

        records.append({
            "image": filename,
            "reference": reference,
            "prediction": prediction,
            "casefold_exact": casefold_exact,
            "error_group": error_group,
            "reference_characters": len(reference),
            "prediction_characters": len(prediction),
            "reference_words": len(reference.split()),
            "prediction_words": len(prediction.split()),
            "character_substitutions": sum(
                edit["type"] == "substitution" for edit in edits
            ),
            "missing_characters": sum(
                edit["type"] == "missing" for edit in edits
            ),
            "extra_characters": sum(
                edit["type"] == "extra" for edit in edits
            ),
            "character_edits": json.dumps(edits, ensure_ascii=False),
            "cer": float(row.cer),
            "wer": float(row.wer),
            "inference_seconds": float(row.inference_seconds),
        })

    return pd.DataFrame(records), {
        "character_substitutions": substitutions,
        "missing_characters": missing_characters,
        "extra_characters": extra_characters,
        "totals": totals,
    }


def _plot_saved_errors(data: pd.DataFrame, operations: dict, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    charts = [
        ("character_substitutions", "Most common substitutions"),
        ("missing_characters", "Most common missing characters"),
        ("extra_characters", "Most common extra characters"),
    ]
    for axis, (key, title) in zip(axes, charts):
        counter = operations[key]
        values = counter.most_common(12)
        if values:
            labels, counts = zip(*values)
            axis.barh(list(reversed(labels)), list(reversed(counts)), color="#187d79")
            axis.set_xlabel("Aligned edit count")
        else:
            axis.text(0.5, 0.5, "No edits", ha="center", va="center")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("TrOCR character-level errors (case-insensitive alignment)")
    figure.tight_layout()
    figure.savefig(output_dir / "character_errors.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(data["cer"].astype(float), bins=30, color="#187d79")
    axes[0].set_title("Per-record character error rate")
    axes[0].set_xlabel("CER")
    axes[0].set_ylabel("Images")
    axes[1].hist(data["wer"].astype(float), bins=30, color="#d19043")
    axes[1].set_title("Per-record word error rate")
    axes[1].set_xlabel("WER")
    axes[1].set_ylabel("Images")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "error_rate_distributions.png", dpi=180)
    plt.close(figure)

    plot_data = data.copy()
    plot_data["reference_length_group"] = pd.cut(
        plot_data["reference"].astype(str).str.len(),
        bins=[0, 5, 8, 12, 20, float("inf")],
        labels=["1-5", "6-8", "9-12", "13-20", "21+"],
    )
    grouped = plot_data.groupby(
        "reference_length_group",
        observed=False,
    ).agg(
        exact_match_rate=("exact_match", "mean"),
        images=("exact_match", "size"),
    )
    figure, axis = plt.subplots(figsize=(8, 4))
    if not grouped.empty:
        bars = axis.bar(
            grouped.index.astype(str),
            grouped["exact_match_rate"] * 100,
            color="#187d79",
        )
        for bar, count in zip(bars, grouped["images"]):
            axis.annotate(
                f"n={count}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axis.set_title("Case-insensitive exact match by reference length")
    axis.set_xlabel("Reference length (characters)")
    axis.set_ylabel("Exact match (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "accuracy_by_reference_length.png", dpi=180)
    plt.close(figure)


def _new_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{timestamp}_", dir=OUTPUT_ROOT))


def _validate_evaluation_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Saved evaluation results not found: {path}")
    data = pd.read_csv(path)
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
        raise ValueError(f"Evaluation CSV has no records: {path}")
    return data.fillna({"reference": "", "prediction": ""})


def analyze_saved_evaluation(output_dir: Path) -> None:
    data = _validate_evaluation_csv(EVALUATION_CSV)
    error_examples, operations = _error_records(data)
    error_examples.to_csv(
        output_dir / "error_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _plot_saved_errors(data, operations, output_dir)

    references = data["reference"].fillna("").astype(str).str.strip().tolist()
    predictions = data["prediction"].fillna("").astype(str).str.strip().tolist()
    summary = {
        "source": str(EVALUATION_CSV),
        "split": "saved test evaluation; descriptive analysis only",
        "evaluated_images": len(data),
        "case_insensitive_exact_match_percent": float(
            data["exact_match"].astype(float).mean() * 100
        ),
        "mean_per_record_cer": float(data["cer"].astype(float).mean()),
        "mean_per_record_wer": float(data["wer"].astype(float).mean()),
        "corpus_cer": float(cer(references, predictions)),
        "corpus_wer": float(wer(references, predictions)),
        "mean_recorded_generation_seconds_per_image": float(
            data["inference_seconds"].astype(float).mean()
        ),
        "raw_text_differences_in_error_csv": len(error_examples),
        "edit_alignment_counts": dict(operations["totals"]),
        "top_substitutions": operations[
            "character_substitutions"
        ].most_common(20),
        "top_missing_characters": operations[
            "missing_characters"
        ].most_common(20),
        "top_extra_characters": operations[
            "extra_characters"
        ].most_common(20),
        "note": (
            "This report describes the existing test predictions. Do not use "
            "it to choose preprocessing or decoding settings."
        ),
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Read-only analysis of saved evaluation: {EVALUATION_CSV}")
    print(f"Images evaluated: {len(data)}")
    print(
        "Case-insensitive exact match: "
        f"{summary['case_insensitive_exact_match_percent']:.2f}%"
    )
    print(
        "Mean CER/WER: "
        f"{summary['mean_per_record_cer']:.4f} / "
        f"{summary['mean_per_record_wer']:.4f}"
    )
    print(f"New report files saved under: {output_dir}")


def _validation_records(limit: int | None) -> list[tuple[str, str, Path]]:
    if not VALIDATION_CSV.is_file():
        raise FileNotFoundError(f"Saved validation split not found: {VALIDATION_CSV}")
    validation = pd.read_csv(VALIDATION_CSV)
    required = {"Images", "Text"}
    missing = required.difference(validation.columns)
    if missing:
        raise ValueError(
            f"Validation CSV is missing required columns: {sorted(missing)}"
        )
    if validation.empty:
        raise ValueError(f"Validation split has no rows: {VALIDATION_CSV}")

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        validation = validation.head(limit)

    records = []
    for row in validation.itertuples(index=False):
        filename = Path(str(row.Images)).name
        image_path = DATASET_DIR / "Train_Set" / filename
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Validation image listed in {VALIDATION_CSV} is missing: "
                f"{image_path}"
            )
        records.append((filename, str(row.Text).strip(), image_path))
    return records


def _generate_batch(
    images: list[Image.Image],
    processor,
    model,
    device: torch.device,
    num_beams: int,
    max_new_tokens: int,
) -> tuple[list[str], list[int], list[bool], float]:
    pixel_values = processor(
        images=images,
        return_tensors="pt",
    ).pixel_values.to(device)
    started_at = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            pixel_values,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
    elapsed = time.perf_counter() - started_at
    sequences = generated.tolist()
    predictions = processor.batch_decode(
        generated,
        skip_special_tokens=True,
    )

    eos_token_id = model.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    elif isinstance(eos_token_id, (list, tuple)):
        eos_ids = set(eos_token_id)
    else:
        eos_ids = set()

    generated_lengths = []
    hit_token_limit = []
    for sequence in sequences:
        decoder_tokens = sequence[1:]
        eos_index = next(
            (
                index
                for index, token_id in enumerate(decoder_tokens)
                if token_id in eos_ids
            ),
            None,
        )
        token_count = (
            eos_index + 1 if eos_index is not None else len(decoder_tokens)
        )
        generated_lengths.append(token_count)
        hit_token_limit.append(
            eos_index is None and token_count >= max_new_tokens
        )

    return (
        [prediction.strip() for prediction in predictions],
        generated_lengths,
        hit_token_limit,
        elapsed / len(predictions),
    )


def _validation_metrics(records: list[dict]) -> dict[str, float | int]:
    references = [row["reference"] for row in records]
    predictions = [row["prediction"] for row in records]
    return {
        "images": len(records),
        "exact_match_percent": sum(
            reference.casefold() == prediction.casefold()
            for reference, prediction in zip(references, predictions)
        )
        / len(records)
        * 100,
        "mean_per_record_cer": sum(row["cer"] for row in records) / len(records),
        "mean_per_record_wer": sum(row["wer"] for row in records) / len(records),
        "corpus_cer": float(cer(references, predictions)),
        "corpus_wer": float(wer(references, predictions)),
        "mean_generation_seconds_per_image": sum(
            row["inference_seconds"] for row in records
        )
        / len(records),
        "outputs_at_max_new_tokens": sum(
            row["hit_max_new_tokens"] for row in records
        ),
    }


def run_validation_sweep(
    output_dir: Path,
    beams: list[int],
    max_new_tokens_values: list[int],
    limit: int | None,
) -> None:
    if not beams or any(value <= 0 for value in beams):
        raise ValueError("--beams values must all be positive integers.")
    if not max_new_tokens_values or any(
        value <= 0 for value in max_new_tokens_values
    ):
        raise ValueError("--max-new-tokens values must all be positive integers.")

    from transformer_inference import DEVICE, load_model

    validation_records = _validation_records(limit)
    processor, model = load_model()
    summary_rows = []

    for num_beams in beams:
        for max_new_tokens in max_new_tokens_values:
            setting_records = []
            for start in range(0, len(validation_records), BATCH_SIZE):
                batch = validation_records[start:start + BATCH_SIZE]
                images = []
                for filename, _, image_path in batch:
                    try:
                        with Image.open(image_path) as source:
                            images.append(source.convert("RGB"))
                    except (UnidentifiedImageError, OSError, ValueError) as exc:
                        raise ValueError(
                            f"Could not open validation image {filename}: {exc}"
                        ) from exc

                predictions, token_lengths, hit_limit, seconds_per_image = (
                    _generate_batch(
                        images,
                        processor,
                        model,
                        DEVICE,
                        num_beams,
                        max_new_tokens,
                    )
                )
                for (
                    (filename, reference, _),
                    prediction,
                    token_length,
                    reached_limit,
                ) in zip(batch, predictions, token_lengths, hit_limit):
                    setting_records.append({
                        "image": filename,
                        "reference": reference,
                        "prediction": prediction,
                        "exact_match": reference.casefold()
                        == prediction.casefold(),
                        "cer": float(cer(reference, prediction)),
                        "wer": float(wer(reference, prediction)),
                        "generated_tokens": token_length,
                        "hit_max_new_tokens": reached_limit,
                        "inference_seconds": seconds_per_image,
                    })

                processed = min(start + len(batch), len(validation_records))
                print(
                    f"Validation beam={num_beams}, "
                    f"max_new_tokens={max_new_tokens}: "
                    f"{processed}/{len(validation_records)}"
                )

            metrics = _validation_metrics(setting_records)
            summary_rows.append({
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
                **metrics,
            })
            predictions_path = (
                output_dir
                / f"validation_predictions_beams_{num_beams}_max_{max_new_tokens}.csv"
            )
            pd.DataFrame(setting_records).to_csv(
                predictions_path,
                index=False,
                encoding="utf-8-sig",
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / "validation_experiment_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    for max_new_tokens in max_new_tokens_values:
        subset = summary[summary["max_new_tokens"] == max_new_tokens]
        axis.plot(
            subset["num_beams"],
            subset["exact_match_percent"],
            marker="o",
            label=f"max_new_tokens={max_new_tokens}",
        )
    axis.set_title("Validation exact match by decoding settings")
    axis.set_xlabel("Beam count")
    axis.set_ylabel("Case-insensitive exact match (%)")
    axis.set_xticks(sorted(set(beams)))
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "validation_exact_match.png", dpi=180)
    plt.close(figure)

    print(f"Validation-only comparison saved under: {output_dir}")
    print(summary.to_string(index=False))
    print(
        "The saved test evaluation was not loaded or used for this sweep."
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze saved TrOCR evaluation errors or compare decoding "
            "settings on the saved validation split. Outputs use a new "
            "timestamped directory and never replace the original evaluation CSV."
        )
    )
    parser.add_argument(
        "--validation-sweep",
        action="store_true",
        help=(
            "Run checkpoint inference only on the saved validation split. "
            "Without this flag, analyze the existing evaluation CSV descriptively."
        ),
    )
    parser.add_argument(
        "--beams",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Positive beam counts for validation-only sweeps (default: 1 2 4).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        nargs="+",
        default=[24],
        help=(
            "Positive generation limits for validation-only sweeps "
            "(default: 24)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Optional deterministic prefix of the validation split for a "
            "quick smoke test; omit to use every saved validation row."
        ),
    )
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    output_dir = _new_output_dir()
    if arguments.validation_sweep:
        run_validation_sweep(
            output_dir,
            arguments.beams,
            arguments.max_new_tokens,
            arguments.limit,
        )
    else:
        analyze_saved_evaluation(output_dir)


if __name__ == "__main__":
    main()
