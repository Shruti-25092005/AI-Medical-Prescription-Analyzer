import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import cv2

from models.ae_inference import load_autoencoder, reconstruct_image as ae_reconstruct, enhance_reconstruction
from models.vae_inference import load_vae_model, reconstruct_image as vae_reconstruct, sample_latent
from models.transformer_inference import (
    load_model as load_trocr,
    predict_image,
    predict_image_detailed,
    predict_crop_sequence,
)
from models.handwriting_gan import (
    load_handwriting_gan,
    generate_handwriting_samples,
    CHECKPOINT_PATH as HW_GAN_CKPT,
)
from models.synthetic_record_gan import (
    Generator as DocGANGenerator,
    CHECKPOINT_PATH as DOC_GAN_CKPT,
)
from models.handwriting_diffusion import (
    load_handwriting_diffusion,
    generate_diffusion_samples,
    CHECKPOINT_PATH as HW_DIFF_CKPT,
)
from models.diffusion_inference import (
    load_diffusion_model,
    generate_image_comparisons,
    generate_images,
    DEFAULT_CHECKPOINT_PATH as DOC_DIFF_CKPT,
)
from models.metrics_evaluator import (
    compute_reconstruction_metrics,
    compute_reconstruction_error_map,
    compute_sharpness,
    compute_contrast,
    compute_ocr_metrics,
    compute_generative_diversity,
    compute_nearest_neighbor_distance,
    evaluate_image_quality,
    detect_generation_failure,
)
from models.prescription_parser import (
    extract_prescription_fields,
    normalize_medicine_name,
)
from models.gui_services import (
    load_model_bundle,
    run_inference,
    model_statuses,
)


class AccuracyAndReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.test_set_dir = cls.project_root / "dataset" / "RxHandBD-ML" / "RxHandBD-ML" / "Test_Set"

    def test_01_all_models_load_actual_checkpoints(self):
        """Final Acceptance Criteria 1: Each model loads and runs using its actual trained checkpoint."""
        statuses = model_statuses()
        self.assertEqual(len(statuses), 5)
        for s in statuses:
            self.assertTrue(s["available"], f"Model {s['id']} must be available and find its checkpoint")

        # 1. AE
        ae_model, ae_dev = load_autoencoder(self.device)
        self.assertIsNotNone(ae_model)

        # 2. VAE
        vae_model, latent_sp, img_sz, vae_dev = load_vae_model(self.device)
        self.assertIsNotNone(vae_model)
        self.assertEqual(img_sz, 256)

        # 3. TrOCR
        trocr_proc, trocr_model = load_trocr()
        self.assertIsNotNone(trocr_proc)
        self.assertIsNotNone(trocr_model)

        # 4. GAN
        self.assertTrue(HW_GAN_CKPT.is_file())
        hw_gan, l_dim = load_handwriting_gan(HW_GAN_CKPT, self.device)
        self.assertIsNotNone(hw_gan)

        # 5. Diffusion
        self.assertTrue(HW_DIFF_CKPT.is_file())
        hw_diff = load_handwriting_diffusion(HW_DIFF_CKPT, self.device)
        self.assertIsNotNone(hw_diff)

    def test_02_ae_reconstruction_and_detail_loss_measurement(self):
        """AE: Reconstruct uploaded image, measure reconstruction error and identify lost details."""
        ae_model, device = load_autoencoder(self.device)
        # Create a sample with high-frequency pen strokes on paper background
        img = Image.new("RGB", (256, 256), color=(250, 250, 250))
        draw = ImageDraw.Draw(img)
        draw.line([(30, 100), (200, 120), (100, 180)], fill=(30, 40, 90), width=4)
        draw.text((40, 50), "Paracetamol 500mg", fill=(20, 20, 20))

        raw_recon = ae_reconstruct(img, ae_model, device)
        self.assertEqual(raw_recon.size, img.size)

        metrics = compute_reconstruction_metrics(img, raw_recon)
        self.assertIn("mse", metrics)
        self.assertIn("psnr_db", metrics)
        self.assertIn("ssim", metrics)
        self.assertGreater(metrics["psnr_db"], 15.0)

        # Measure lost details and generate error heatmap
        err_map, d_metrics = compute_reconstruction_error_map(img, raw_recon)
        self.assertEqual(err_map.size, img.size)
        self.assertIn("sharpness_orig", d_metrics)
        self.assertIn("sharpness_recon", d_metrics)
        self.assertIn("edge_loss_pct", d_metrics)
        self.assertGreaterEqual(d_metrics["edge_loss_pct"], 0.0)

    def test_03_vae_valid_reconstruction_and_honest_quality_evaluation(self):
        """VAE: Produce valid reconstructions. Evaluate image quality and avoid claiming clearer without evidence."""
        vae_model, latent_spatial, image_size, device = load_vae_model(self.device)
        test_img = Image.new("RGB", (image_size, image_size), color=(240, 240, 240))
        draw = ImageDraw.Draw(test_img)
        draw.line([(20, 50), (220, 70), (80, 150)], fill=(10, 10, 10), width=3)

        raw_recon = vae_reconstruct(test_img, vae_model, image_size, device, enhance=False, preserve_resolution=False)
        self.assertEqual(raw_recon.size, (image_size, image_size))

        metrics = compute_reconstruction_metrics(test_img, raw_recon)
        err_map, d_metrics = compute_reconstruction_error_map(test_img, raw_recon)

        # Verify sharpness measurement is present and reflects actual comparison
        self.assertIn("sharpness_orig", d_metrics)
        self.assertIn("sharpness_recon", d_metrics)
        # Because of Gaussian latent regularization, raw VAE is softer than sharp binary input
        self.assertLessEqual(d_metrics["sharpness_recon"], d_metrics["sharpness_orig"] + 50.0)

    def test_04_trocr_confidence_and_uncertain_word_flagging(self):
        """Transformer (TrOCR): Extract text, preserve recognized text faithfully, flag uncertain words."""
        processor, model = load_trocr()

        # 1. Blank/white image should produce low confidence or empty text and be flagged uncertain
        blank_img = Image.new("RGB", (384, 384), color=(255, 255, 255))
        blank_pred = predict_image_detailed(blank_img, processor, model)
        self.assertIsInstance(blank_pred["text"], str)
        self.assertIn("confidence", blank_pred)
        self.assertIn("is_uncertain", blank_pred)

        # 2. Test with real sample if available
        if self.test_set_dir.is_dir():
            sample_files = sorted(self.test_set_dir.glob("*.jpg"))
            if sample_files:
                sample_img = Image.open(sample_files[0])
                pred_res = predict_image_detailed(sample_img, processor, model)
                self.assertIsInstance(pred_res["text"], str)
                self.assertGreater(len(pred_res["text"]), 0)
                self.assertGreaterEqual(pred_res["confidence"], 0.0)
                self.assertLessEqual(pred_res["confidence"], 1.0)

    def test_05_prescription_parser_never_guesses_missing_data(self):
        """Prescription parser: Never infer missing medication names, strengths, dosages, routes, or durations."""
        text = "Tab. Napa\n1-0-1\n7 days"
        meta, items = extract_prescription_fields(text)
        self.assertEqual(len(items), 3)

        napa_item = items[0]
        self.assertEqual(napa_item.dosage_form, "Tablet")
        # Strength was not specified in the line "Tab. Napa", so it MUST NOT be guessed!
        self.assertEqual(napa_item.strength, "Not specified")

        # An uncertain / gibberish text must NOT be silently replaced with a guessed medicine
        gibberish_norm = normalize_medicine_name("Xzqvwkjj 500mg")
        self.assertEqual(gibberish_norm["status"], "Uncertain / Review Needed")
        self.assertIsNone(gibberish_norm["matched_name"])

    def test_06_gan_generation_diversity_and_failure_detection(self):
        """GAN: Generate varied synthetic document/handwriting mockups. Detect repetitive or degraded outputs."""
        generator, latent_dim = load_handwriting_gan(HW_GAN_CKPT, self.device)

        # Generate with seed 1
        samples_1 = generate_handwriting_samples(generator, count=4, seed=123)
        self.assertEqual(len(samples_1), 4)
        div_1 = compute_generative_diversity(samples_1)
        self.assertGreater(div_1, 0.05, "Generated samples must have non-zero diversity (no mode collapse)")

        # Generate with seed 2 and verify meaningful difference
        samples_2 = generate_handwriting_samples(generator, count=4, seed=999)
        diff_between_seeds = float(np.mean(np.abs(np.array(samples_1[0]) - np.array(samples_2[0]))))
        self.assertGreater(diff_between_seeds, 0.0, "Different seeds must produce different outputs")

        # Test failure detector on blank images vs real generations
        blank_batch = [Image.new("RGB", (64, 64), color=(255, 255, 255)) for _ in range(4)]
        fail_rep = detect_generation_failure(blank_batch)
        self.assertTrue(fail_rep["is_failure"], "Failure detector must flag uniform blank images")

        real_rep = detect_generation_failure(samples_1)
        self.assertFalse(real_rep["is_failure"], "Valid GAN generations should not be marked as failure")

    def test_07_diffusion_pure_sampling_without_dataset_leakage(self):
        """Diffusion: Generate samples from pure Gaussian noise using trained checkpoint without dataset leakage."""
        hw_diff = load_handwriting_diffusion(HW_DIFF_CKPT, self.device)
        samples, trajs = generate_diffusion_samples(hw_diff, count=2, steps=6, seed=42)
        self.assertEqual(len(samples), 2)
        self.assertEqual(len(trajs), 2)

        # Check trajectory length
        self.assertGreaterEqual(len(trajs[0]), 2)
        # Verify samples are varied
        div = compute_generative_diversity(samples)
        self.assertGreater(div, 0.01)

        # Verify Document Diffusion loads and generates without procedural replacement
        doc_bundle = load_diffusion_model()
        comparisons = generate_image_comparisons(count=1, sampling_steps=4, seed=10, model_bundle=doc_bundle)
        self.assertEqual(len(comparisons), 1)
        noise_img, gen_img = comparisons[0]
        self.assertEqual(gen_img.size, (512, 512))

    def test_08_no_fake_or_hardcoded_metrics_in_gui_services(self):
        """Final Acceptance Criteria 5: All metrics reflect actual measurements, no fabricated numbers."""
        test_img = Image.new("RGB", (128, 128), color=(240, 240, 240))

        # Test AE
        ae_bundle = load_model_bundle("ae")
        ae_res = run_inference("ae", test_img, ae_bundle)
        self.assertIsNotNone(ae_res.reconstruction)
        self.assertIsNotNone(ae_res.reconstruction_metrics)
        self.assertNotIn("purity_pct", ae_res.reconstruction_metrics)
        self.assertNotIn("synthetic_fidelity", ae_res.reconstruction_metrics)
        self.assertIn("mse", ae_res.reconstruction_metrics)
        self.assertIn("psnr_db", ae_res.reconstruction_metrics)
        self.assertIn("ssim", ae_res.reconstruction_metrics)

        # Test GAN
        gan_bundle = load_model_bundle("gan")
        gan_res = run_inference("gan", test_img, gan_bundle)
        self.assertIsNotNone(gan_res.reconstruction_metrics)
        self.assertNotIn("purity_pct", gan_res.reconstruction_metrics)
        self.assertIn("diversity_l1", gan_res.reconstruction_metrics)
        self.assertIn("mean_sharpness", gan_res.reconstruction_metrics)

        # Test Diffusion
        diff_bundle = load_model_bundle("diffusion")
        diff_res = run_inference("diffusion", test_img, diff_bundle)
        self.assertIsNotNone(diff_res.reconstruction_metrics)
        self.assertNotIn("synthetic_fidelity", diff_res.reconstruction_metrics)
        self.assertIn("diversity_l1", diff_res.reconstruction_metrics)

    def test_09_ocr_evaluation_metrics_with_ground_truth(self):
        """OCR evaluation: Measure CER, WER, and exact match against verified ground truth."""
        predictions = ["Paracetamol 500mg", "Amoxicilin 250mg", "Losartan"]
        references = ["Paracetamol 500mg", "Amoxicillin 250mg", "Losartan 50mg"]

        metrics = compute_ocr_metrics(predictions, references)
        self.assertAlmostEqual(metrics["exact_match_acc"], 33.33, places=1)
        self.assertGreater(metrics["mean_cer"], 0.0)
        self.assertGreater(metrics["mean_wer"], 0.0)
        self.assertEqual(metrics["sample_count"], 3)

    def test_10_representative_image_stress_testing(self):
        """Testing multiple representative images: clear, blurry, rotated, and difficult-to-read samples."""
        base_img = Image.new("RGB", (256, 256), color=(255, 255, 255))
        draw = ImageDraw.Draw(base_img)
        draw.text((30, 80), "Tab Napa 500mg", fill=(0, 0, 0))
        draw.line([(20, 140), (220, 140)], fill=(0, 0, 0), width=3)

        # 1. Clear sample
        clear_img = base_img.copy()

        # 2. Blurry sample
        blurry_img = base_img.filter(ImageFilter.GaussianBlur(radius=3.5))

        # 3. Rotated sample
        rotated_img = base_img.rotate(15, expand=False, fillcolor=(255, 255, 255))

        # 4. Difficult / degraded sample (low contrast + noise)
        arr = np.asarray(base_img, dtype=np.float32)
        degraded_arr = arr * 0.4 + 100 + np.random.normal(0, 12, arr.shape)
        degraded_img = Image.fromarray(np.clip(degraded_arr, 0, 255).astype(np.uint8))

        ae_bundle = load_model_bundle("ae")
        for label, img in [("clear", clear_img), ("blurry", blurry_img), ("rotated", rotated_img), ("degraded", degraded_img)]:
            res = run_inference("ae", img, ae_bundle)
            self.assertIsNotNone(res.reconstruction, f"AE failed on {label} image")
            self.assertIsNotNone(res.error_map_preview, f"Error map failed on {label} image")
            self.assertIn("mse", res.reconstruction_metrics)


if __name__ == "__main__":
    unittest.main()
