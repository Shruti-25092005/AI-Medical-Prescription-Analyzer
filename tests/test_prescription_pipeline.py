import unittest
import numpy as np
import torch
from PIL import Image

from models.prescription_parser import (
    normalize_medicine_name,
    extract_prescription_fields,
    MEDICINE_REFERENCE_DB,
)
from models.medicine_knowledge import explain_medicine
from models.drug_interactions import check_drug_interactions
from models.page_segmenter import segment_prescription_page, preprocess_image_pipeline
from models.metrics_evaluator import (
    compute_reconstruction_metrics,
    compute_ocr_metrics,
    compute_generative_diversity,
    compute_nearest_neighbor_distance,
)
from models.handwriting_gan import smoke_test_gan
from models.handwriting_diffusion import smoke_test_diffusion
from models.vae_inference import compute_vae_loss_components, SharperVAE


class PrescriptionPipelineTests(unittest.TestCase):
    def test_medicine_normalization_exact_match(self):
        result = normalize_medicine_name("Napa")
        self.assertEqual(result["status"], "Verified Exact")
        self.assertEqual(result["generic_name"], "Paracetamol")
        self.assertGreaterEqual(result["similarity_score"], 0.95)

    def test_medicine_normalization_fuzzy_match(self):
        # Slightly misspelled or noisy OCR output: "Nexlav" or "Zimaxx"
        result = normalize_medicine_name("Zimaxx")
        self.assertIn(result["status"], ["Verified Exact", "Likely Match"])
        self.assertEqual(result["generic_name"], "Azithromycin")
        self.assertTrue(len(result["candidate_matches"]) > 0)

    def test_medicine_normalization_uncertain_handling(self):
        # Completely unknown or corrupted text
        result = normalize_medicine_name("xyz123qwer")
        self.assertEqual(result["status"], "Uncertain / Review Needed")
        # Critical safety rule: must NEVER silently replace an unclear medicine name with a guessed name
        self.assertIsNone(result["matched_name"])

    def test_prescription_fields_extraction_complete(self):
        sample_prescription_text = (
            "Dr. M. Rahman MBBS FCPS\n"
            "Reg No: 12345-BD\n"
            "Date: 24/10/2026\n"
            "Tab. Napa 500mg 1-0-1 5 days oral after food\n"
            "Cap. Zimax 500mg 1-0-0 3 days oral before food\n"
        )
        metadata, items = extract_prescription_fields(sample_prescription_text)

        self.assertIn("M. Rahman", str(metadata.prescriber_name))
        self.assertEqual(metadata.prescription_date, "24/10/2026")
        self.assertEqual(len(items), 2)

        napa_item = items[0]
        self.assertEqual(napa_item.dosage_form, "Tablet")
        self.assertEqual(napa_item.strength, "500mg")
        self.assertEqual(napa_item.frequency, "1-0-1")
        self.assertEqual(napa_item.duration, "5 days")

    def test_prescription_fields_non_inference_of_missing(self):
        # Safety rule: Do NOT infer missing dosages, frequencies, or durations!
        sample_sparse_text = "Napa\n"
        _, items = extract_prescription_fields(sample_sparse_text)
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.dosage_form, "Not specified")
        self.assertEqual(item.strength, "Not specified")
        self.assertEqual(item.frequency, "Not specified")
        self.assertEqual(item.duration, "Not specified")
        self.assertEqual(item.route, "Not specified")
        self.assertEqual(item.instructions, "Not specified")

    def test_medicine_explanation_retrieval(self):
        mono = explain_medicine("Paracetamol")
        self.assertTrue(mono["found"])
        self.assertIn("indications", mono)
        self.assertIn("mechanism", mono)
        self.assertIn("side_effects", mono)
        self.assertIn("warnings", mono)
        self.assertIn("sources", mono)
        self.assertIn("Academic Research Prototype", mono["disclaimer"])

    def test_medicine_explanation_uncatalogued(self):
        mono = explain_medicine("UnknownDrugXYZ")
        self.assertFalse(mono["found"])
        self.assertIn("not available in the local offline reference database", mono["indications"][0])

    def test_drug_interactions_checking(self):
        # Propranolol + Clonazepam is a documented moderate interaction
        result = check_drug_interactions(["Propranolol", "Clonazepam"])
        self.assertEqual(result["interaction_count"], 1)
        self.assertEqual(result["interactions"][0]["severity"], "Moderate")
        self.assertIn("CRITICAL SAFETY NOTE", result["safety_disclaimer"])

    def test_drug_interactions_major_severity(self):
        # Aspirin + Dexibuprofen is a major severity interaction
        result = check_drug_interactions(["Aspirin", "Dexibuprofen"])
        self.assertEqual(result["interaction_count"], 1)
        self.assertTrue(result["has_major_alert"])
        self.assertEqual(result["interactions"][0]["severity"], "Major / Severe")

    def test_drug_interactions_no_record_safety_claim(self):
        # Rule: Application must never claim a prescription is medically safe solely because no interaction was found
        result = check_drug_interactions(["Paracetamol", "Cetirizine"])
        self.assertIn("DOES NOT prove that this combination is medically safe", result["safety_disclaimer"])

    def test_page_segmenter_pipeline(self):
        img = Image.new("RGB", (300, 300), color="white")
        seg_res = segment_prescription_page(img)
        self.assertTrue(seg_res.is_single_crop)
        self.assertEqual(len(seg_res.crops), 1)

    def test_metrics_evaluator(self):
        img1 = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img2 = Image.new("RGB", (64, 64), color=(105, 105, 105))
        recon_metrics = compute_reconstruction_metrics(img1, img2)
        self.assertIn("mse", recon_metrics)
        self.assertIn("psnr_db", recon_metrics)
        self.assertIn("ssim", recon_metrics)

        ocr_metrics = compute_ocr_metrics(["napa", "zimax"], ["napa", "zimax"])
        self.assertEqual(ocr_metrics["exact_match_acc"], 100.0)
        self.assertEqual(ocr_metrics["mean_cer"], 0.0)

    def test_vae_loss_components_computation(self):
        device = torch.device("cpu")
        model = SharperVAE(latent_channels=8, image_size=32).to(device)
        sample_img = Image.new("L", (32, 32), color=128)
        recon_mse, kl_div = compute_vae_loss_components(sample_img, model, image_size=32, device=device)
        self.assertGreaterEqual(recon_mse, 0.0)
        self.assertGreaterEqual(kl_div, 0.0)

    def test_gan_and_diffusion_smoke_tests(self):
        self.assertTrue(smoke_test_gan())
        self.assertTrue(smoke_test_diffusion())


if __name__ == "__main__":
    unittest.main()
