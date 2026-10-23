from io import BytesIO
import unittest

from PIL import Image

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
    FictionalRecord,
    SYNTHETIC_IMAGE_BANNER,
    SYNTHETIC_OUTPUT_LABEL,
    records_as_dicts,
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
    def test_sample_metadata_has_no_identity_or_clinical_fields(self) -> None:
        record = FictionalRecord(
            record_id="SYNTHETIC-000001",
            content="NO CLINICAL CONTENT",
            status="NOT A PRESCRIPTION",
        )

        self.assertEqual(
            records_as_dicts([record]),
            [{
                "record_id": "SYNTHETIC-000001",
                "content": "NO CLINICAL CONTENT",
                "status": "NOT A PRESCRIPTION",
            }],
        )
        self.assertIn("SYNTHETIC", SYNTHETIC_OUTPUT_LABEL)
        self.assertIn("NOT A REAL PRESCRIPTION", SYNTHETIC_OUTPUT_LABEL)
        self.assertIn("NOT FOR MEDICAL USE", SYNTHETIC_OUTPUT_LABEL)
        self.assertIn("AI-GENERATED", SYNTHETIC_IMAGE_BANNER)
        self.assertIn("NOT A PRESCRIPTION", SYNTHETIC_IMAGE_BANNER)


if __name__ == "__main__":
    unittest.main()
