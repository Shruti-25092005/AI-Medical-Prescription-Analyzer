import unittest
from pathlib import Path
import torch
from PIL import Image

from models.ae_inference import load_autoencoder, reconstruct_image as ae_reconstruct, enhance_reconstruction
from models.vae_inference import load_vae_model, reconstruct_image as vae_reconstruct, sample_latent, interpolate_latent, compute_vae_loss_components
from models.handwriting_gan import load_handwriting_gan, generate_handwriting_samples, CHECKPOINT_PATH as GAN_CKPT_PATH
from models.handwriting_diffusion import load_handwriting_diffusion, generate_diffusion_samples, CHECKPOINT_PATH as DIFF_CKPT_PATH
from models.transformer_inference import load_model as load_trocr, predict_image
from models.metrics_evaluator import compute_reconstruction_metrics, compute_generative_diversity


class AllModelsIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")

    def test_01_autoencoder_inference(self):
        model, device = load_autoencoder(self.device)
        self.assertIsNotNone(model)
        dummy_img = Image.new("RGB", (128, 128), color=(200, 200, 200))
        recon = ae_reconstruct(dummy_img, model, device)
        enhanced = enhance_reconstruction(recon)
        self.assertEqual(enhanced.size, dummy_img.size)
        metrics = compute_reconstruction_metrics(dummy_img, enhanced)
        self.assertIn("mse", metrics)
        self.assertIn("psnr_db", metrics)

    def test_02_vae_inference_and_generation(self):
        model, latent_spatial, image_size, device = load_vae_model(self.device)
        self.assertIsNotNone(model)
        dummy_img = Image.new("RGB", (image_size, image_size), color=(220, 220, 220))

        # 1. Reconstruction
        recon = vae_reconstruct(dummy_img, model, image_size, device)
        self.assertEqual(recon.size, (image_size, image_size))

        # 2. Loss & KL divergence components
        recon_mse, kl_div = compute_vae_loss_components(dummy_img, model, image_size, device)
        self.assertGreaterEqual(recon_mse, 0.0)
        self.assertGreaterEqual(kl_div, 0.0)

        # 3. Latent prior sampling
        samples = sample_latent(model=model, count=2, latent_spatial=latent_spatial, seed=42, device=device)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].size, (image_size, image_size))

        # 4. Latent interpolation
        dummy_img2 = Image.new("RGB", (image_size, image_size), color=(180, 180, 180))
        interp = interpolate_latent(model=model, image1=dummy_img, image2=dummy_img2, steps=3, image_size=image_size, device=device)
        self.assertEqual(len(interp), 3)

    def test_03_gan_generation(self):
        self.assertTrue(GAN_CKPT_PATH.is_file(), "GAN checkpoint must exist")
        generator, latent_dim = load_handwriting_gan(GAN_CKPT_PATH, self.device)
        self.assertIsNotNone(generator)
        samples = generate_handwriting_samples(generator, count=4, seed=42, device=self.device)
        self.assertEqual(len(samples), 4)
        for s in samples:
            self.assertEqual(s.size, (256, 256))
        diversity = compute_generative_diversity(samples)
        self.assertGreater(diversity, 0.0)

    def test_04_diffusion_generation(self):
        self.assertTrue(DIFF_CKPT_PATH.is_file(), "Diffusion checkpoint must exist")
        diffusion = load_handwriting_diffusion(DIFF_CKPT_PATH, self.device)
        self.assertIsNotNone(diffusion)
        # Fast 5-step test
        samples, trajs = generate_diffusion_samples(diffusion, count=2, steps=5, seed=42, image_size=64)
        self.assertEqual(len(samples), 2)
        self.assertEqual(len(trajs), 2)
        self.assertGreaterEqual(len(trajs[0]), 2)

    def test_05_trocr_inference(self):
        processor, model = load_trocr()
        self.assertIsNotNone(processor)
        self.assertIsNotNone(model)
        dummy_word = Image.new("RGB", (384, 384), color=(255, 255, 255))
        pred = predict_image(dummy_word, processor, model)
        self.assertIsInstance(pred, str)


if __name__ == "__main__":
    unittest.main()
