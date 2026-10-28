from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from models.prescription_upload import (
    MAX_UPLOAD_SIZE_BYTES,
    UploadValidationError,
    load_prescription_pages,
)
from models.privacy_controls import (
    clear_analysis_results,
    clear_uploaded_data,
)
from models.synthetic_record_gan import (
    CHECKPOINT_FORMAT_VERSION,
    GAN_ARCHITECTURE,
    IMAGE_SIZE,
    LATENT_DIMENSION,
    Discriminator,
    FictionalRecord,
    Generator,
    SYNTHETIC_IMAGE_BANNER,
    SYNTHETIC_OUTPUT_LABEL,
    SyntheticDocumentDataset,
    _upscale_generated_image,
    _tensor_to_grayscale_image,
    generate_records,
    records_as_dicts,
    train_gan,
)


class UploadValidationTests(unittest.TestCase):
    def _png_bytes(self) -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (2, 2), color="white").save(buffer, format="PNG")
        return buffer.getvalue()

    def test_accepts_matching_small_image(self) -> None:
        pages = load_prescription_pages("sample.png", self._png_bytes())

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].image.size, (2, 2))

    def test_rejects_mismatched_image_extension_without_details(self) -> None:
        with self.assertRaises(UploadValidationError) as context:
            load_prescription_pages("private.jpg", self._png_bytes())

        self.assertEqual(
            str(context.exception),
            "The file contents do not match the selected image format.",
        )
        self.assertNotIn("private.jpg", str(context.exception))

    def test_rejects_oversized_upload_before_decoding(self) -> None:
        content = b"x" * (MAX_UPLOAD_SIZE_BYTES + 1)

        with self.assertRaisesRegex(UploadValidationError, "10 MB limit"):
            load_prescription_pages("private.png", content)


class PrivacyControlTests(unittest.TestCase):
    def test_replacing_upload_clears_previous_extracted_text(self) -> None:
        state = {
            "active_upload_id": "old upload",
            "latest_analysis": {"recognized_text": "private text"},
            "analysis_old_run_page_1": "private text",
            "prescription_upload_0": b"new upload",
        }

        clear_analysis_results(state)

        self.assertEqual(
            state,
            {"prescription_upload_0": b"new upload"},
        )

    def test_clear_removes_upload_and_result_state(self) -> None:
        state = {
            "prescription_upload_generation": 2,
            "prescription_upload_2": b"private image bytes",
            "latest_analysis": {"recognized_text": "private text"},
            "active_upload_id": "private fingerprint",
            "analysis_run_page_1": "private text",
            "verified_run_page_1": "private text",
            "human_verified_run": True,
            "processing_statistics": {"runs": 1},
            "unrelated_setting": "keep",
        }

        clear_uploaded_data(state)

        self.assertEqual(
            state,
            {
                "prescription_upload_generation": 3,
                "processing_statistics": {"runs": 1},
                "unrelated_setting": "keep",
            },
        )


class SyntheticOutputTests(unittest.TestCase):
    def test_training_data_is_form_shaped_normalized_and_varied(self) -> None:
        dataset = SyntheticDocumentDataset(count=2, seed=1)
        first, second = dataset[0], dataset[1]

        self.assertEqual(
            tuple(first.shape),
            (1, IMAGE_SIZE, IMAGE_SIZE),
        )
        self.assertGreaterEqual(first.min().item(), -1.0)
        self.assertLessEqual(first.max().item(), 1.0)
        self.assertFalse((first == second).all().item())

    def test_generator_and_discriminator_have_matching_image_shapes(self) -> None:
        generator = Generator()
        discriminator = Discriminator()
        generated = generator(torch.randn(2, LATENT_DIMENSION, 1, 1))

        self.assertEqual(tuple(generated.shape), (2, 1, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(tuple(discriminator(generated).shape), (2,))

    def test_training_saves_compatible_checkpoint_and_generates_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "gan.pth"
            history = train_gan(
                epochs=1,
                record_count=2,
                batch_size=2,
                seed=7,
                checkpoint_path=checkpoint,
            )
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            with patch("models.synthetic_record_gan.CHECKPOINT_PATH", checkpoint):
                outputs = generate_records(count=2, seed=11)

        self.assertEqual(saved["format_version"], CHECKPOINT_FORMAT_VERSION)
        self.assertEqual(saved["architecture"], GAN_ARCHITECTURE)
        self.assertEqual(saved["image_size"], IMAGE_SIZE)
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(history[0]["generator_loss"]))
        self.assertTrue(np.isfinite(history[0]["diversity_loss"]))
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0][1].size, (512, 512))
        self.assertEqual(outputs[0][1].getpixel((0, 460)), (145, 28, 38))
        self.assertNotEqual(
            np.asarray(outputs[0][1].crop((0, 0, 512, 450))).tobytes(),
            np.asarray(outputs[1][1].crop((0, 0, 512, 450))).tobytes(),
        )
        self.assertEqual(
            records_as_dicts([outputs[0][0]])[0]["content"],
            "GAN-generated text-free form",
        )

    def test_legacy_card_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pth"
            torch.save({"generator_state_dict": {}}, checkpoint)
            with patch("models.synthetic_record_gan.CHECKPOINT_PATH", checkpoint):
                with self.assertRaisesRegex(ValueError, "legacy card dataset"):
                    generate_records(count=1)

    def test_output_is_prominently_labeled_synthetic(self) -> None:
        self.assertIn("SYNTHETIC", SYNTHETIC_OUTPUT_LABEL)
        self.assertIn("NOT A REAL PRESCRIPTION", SYNTHETIC_OUTPUT_LABEL)
        self.assertIn("NOT FOR MEDICAL USE", SYNTHETIC_OUTPUT_LABEL)
        self.assertIn("AI-GENERATED", SYNTHETIC_IMAGE_BANNER)
        self.assertIn("NOT A PRESCRIPTION", SYNTHETIC_IMAGE_BANNER)

    def test_inverse_normalization_maps_tanh_range_to_grayscale(self) -> None:
        tensor = torch.tensor([[-1.0, 0.0, 1.0]])

        image = _tensor_to_grayscale_image(tensor)

        self.assertEqual(image.mode, "L")
        self.assertEqual(list(image.getdata()), [0, 128, 255])

    def test_upscaling_preserves_dimensions_without_filtering(self) -> None:
        source = Image.fromarray(
            np.tile(np.array([0, 0, 255, 255], dtype=np.uint8), (64, 16)),
            mode="L",
        )
        smooth = source.resize((512, 512), Image.Resampling.LANCZOS)
        upscaled = _upscale_generated_image(source)

        self.assertEqual(upscaled.size, (512, 512))
        self.assertEqual(
            np.asarray(upscaled).tobytes(),
            np.asarray(smooth.convert("RGB")).tobytes(),
        )


if __name__ == "__main__":
    unittest.main()
