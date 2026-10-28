from pathlib import Path
import errno
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from models.diffusion_inference import (
    CHECKPOINT_FORMAT_VERSION,
    IMAGE_NORMALIZATION,
    MODEL_ARCHITECTURE,
    NOISE_SCHEDULE,
    PREDICTION_TYPE,
    DiffusionDenoiser,
    generate_image_comparisons,
    generate_images,
    cosine_noise_schedule,
    load_diffusion_model,
    render_synthetic_form,
    velocity_to_clean_and_noise,
)
from models.train_diffusion import SyntheticFormDataset, _write_history, train


class DiffusionModelTests(unittest.TestCase):
    def test_denoiser_preserves_spatial_dimensions(self) -> None:
        model = DiffusionDenoiser(
            image_channels=1,
            base_channels=4,
            timesteps=4,
        )

        output = model(torch.randn(2, 1, 16, 16), torch.tensor([0, 3]))

        self.assertEqual(output.shape, (2, 1, 16, 16))

    def test_training_data_is_procedural_and_text_free(self) -> None:
        dataset = SyntheticFormDataset(count=2, image_size=32, seed=10)

        first = dataset[0]
        second = dataset[1]

        self.assertEqual(tuple(first.shape), (1, 32, 32))
        self.assertGreaterEqual(first.min().item(), -1.0)
        self.assertLessEqual(first.max().item(), 1.0)
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(render_synthetic_form(0, 32, 10).mode, "L")

    def test_tiny_training_checkpoint_generates_watermarked_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "diffusion_model.pth"
            train(
                epochs=1,
                sample_count=2,
                batch_size=2,
                image_size=16,
                timesteps=4,
                base_channels=4,
                seed=9,
                checkpoint_path=checkpoint_path,
                sample_interval=1,
                sample_directory=Path(directory) / "progress",
                samples_per_checkpoint=1,
            )

            images = generate_images(
                count=1,
                sampling_steps=2,
                seed=11,
                checkpoint_path=checkpoint_path,
            )
            grid_path = (
                Path(directory)
                / "progress"
                / "training_comparison_grid.png"
            )
            self.assertTrue(grid_path.is_file())

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].size, (512, 512))
        self.assertEqual(images[0].getpixel((0, 0)), (150, 25, 35))
        self.assertEqual(images[0].getpixel((0, 459)), (150, 25, 35))

    def test_training_checkpoint_records_inference_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "diffusion_model.pth"
            train(
                epochs=1,
                sample_count=2,
                batch_size=2,
                image_size=16,
                timesteps=4,
                base_channels=4,
                seed=5,
                checkpoint_path=checkpoint_path,
                sample_interval=0,
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            loaded_bundle = load_diffusion_model(checkpoint_path)

        self.assertEqual(checkpoint["format_version"], CHECKPOINT_FORMAT_VERSION)
        self.assertEqual(checkpoint["architecture"], MODEL_ARCHITECTURE)
        self.assertEqual(checkpoint["normalization"], IMAGE_NORMALIZATION)
        self.assertEqual(checkpoint["noise_schedule"], NOISE_SCHEDULE)
        self.assertEqual(checkpoint["prediction_type"], PREDICTION_TYPE)
        self.assertGreater(checkpoint["validation_velocity_mse"], 0)
        self.assertEqual(checkpoint["image_size"], 16)
        self.assertEqual(checkpoint["training_epochs"], 1)
        self.assertEqual(loaded_bundle["training_epochs"], 1)
        self.assertEqual(loaded_bundle["training_sample_count"], 2)

    def test_velocity_prediction_recovers_clean_image_and_noise(self) -> None:
        clean = torch.randn(2, 1, 4, 4)
        noise = torch.randn_like(clean)
        alpha = torch.tensor([0.8, 0.35]).view(2, 1, 1, 1)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        velocity = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean

        predicted_clean, predicted_noise = velocity_to_clean_and_noise(
            noisy,
            velocity,
            alpha.reshape(2),
        )

        self.assertTrue(torch.allclose(predicted_clean, clean, atol=1e-6))
        self.assertTrue(torch.allclose(predicted_noise, noise, atol=1e-6))

    def test_training_resumes_and_warm_starts_at_new_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.pth"
            target_path = Path(directory) / "target.pth"
            train(
                epochs=1,
                sample_count=2,
                batch_size=2,
                image_size=16,
                timesteps=4,
                base_channels=4,
                seed=13,
                checkpoint_path=source_path,
                sample_interval=0,
            )
            train(
                epochs=2,
                sample_count=2,
                batch_size=2,
                image_size=16,
                timesteps=4,
                base_channels=4,
                seed=13,
                checkpoint_path=source_path,
                resume=True,
                sample_interval=0,
            )
            source_checkpoint = torch.load(
                source_path,
                map_location="cpu",
                weights_only=True,
            )
            train(
                epochs=1,
                sample_count=2,
                batch_size=2,
                image_size=32,
                timesteps=4,
                base_channels=4,
                seed=17,
                checkpoint_path=target_path,
                resume_from=source_path,
                sample_interval=0,
            )
            target_checkpoint = torch.load(
                target_path,
                map_location="cpu",
                weights_only=True,
            )
            history_path = target_path.with_name(
                "target_training_history.csv"
            )
            history = history_path.read_text(encoding="utf-8")

        self.assertEqual(source_checkpoint["training_epochs"], 2)
        self.assertEqual(target_checkpoint["training_epochs"], 1)
        self.assertEqual(target_checkpoint["warm_start_epochs"], 2)
        self.assertEqual(target_checkpoint["image_size"], 32)
        self.assertIn("validation_velocity_mse", history)

    def test_cosine_schedule_is_monotonic_and_ends_near_pure_noise(self) -> None:
        betas, alphas, alpha_cumulative = cosine_noise_schedule(100, "cpu")

        self.assertEqual(betas.shape, (100,))
        self.assertTrue(torch.all(betas[1:] > betas[:-1]))
        self.assertTrue(torch.all(alphas[1:] < alphas[:-1]))
        self.assertTrue(torch.all(alpha_cumulative[1:] < alpha_cumulative[:-1]))
        self.assertLess(alpha_cumulative[-1].item(), 1e-4)

    def test_generation_rejects_invalid_sample_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 8"):
            generate_images(count=9)

    def test_generation_comparison_shows_start_noise_and_model_output(self) -> None:
        model = DiffusionDenoiser(
            image_channels=1,
            base_channels=4,
            timesteps=4,
        ).eval()
        bundle = {
            "model": model,
            "image_size": 16,
            "timesteps": 4,
            "device": torch.device("cpu"),
        }

        comparisons = generate_image_comparisons(
            count=1,
            sampling_steps=4,
            seed=11,
            model_bundle=bundle,
        )

        self.assertEqual(len(comparisons), 1)
        noise_image, generated_image = comparisons[0]
        self.assertEqual(noise_image.size, (512, 512))
        self.assertEqual(generated_image.size, (512, 512))
        self.assertNotEqual(noise_image.tobytes(), generated_image.tobytes())

    def test_training_history_retries_transient_file_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "history.csv"
            replace = os.replace
            replace_calls = 0

            def replace_after_lock(source: Path, destination: Path) -> None:
                nonlocal replace_calls
                replace_calls += 1
                if replace_calls == 1:
                    raise PermissionError(
                        errno.EACCES,
                        "File is temporarily locked.",
                    )
                replace(source, destination)

            with (
                patch(
                    "models.train_diffusion.os.replace",
                    side_effect=replace_after_lock,
                ),
                patch("models.train_diffusion.time.sleep"),
            ):
                _write_history(
                    history_path,
                    [
                        {
                            "epoch": 3,
                            "train_velocity_mse": 0.06,
                            "validation_velocity_mse": 0.05,
                            "best_validation_velocity_mse": 0.05,
                        }
                    ],
                )

            history = history_path.read_text(encoding="utf-8")

        self.assertEqual(replace_calls, 2)
        self.assertIn("0.05", history)

    def test_loader_rejects_outdated_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "old_checkpoint.pth"
            torch.save({"format_version": 2}, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "outdated or incompatible"):
                load_diffusion_model(checkpoint_path)


if __name__ == "__main__":
    unittest.main()
