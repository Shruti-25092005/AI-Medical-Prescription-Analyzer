# Deployment notes

The Streamlit configuration in `.streamlit/config.toml` sets a 10 MB upload
limit, hides detailed application errors, enables XSRF/CORS protections, and
disables browser usage-stat collection. These are baseline settings, not a
security or compliance certification.

Before exposing the app:

- Put it behind HTTPS and an authenticated access gateway; neither is provided
  by this project.
- Supply any required credentials through the deployment platform's secret
  manager. Do not commit `.env` files or `.streamlit/secrets.toml`.
- For multiple app instances, configure a strong shared Streamlit cookie secret
  through the deployment environment/secret manager. Never put its value in
  this repository.
- Review host, proxy, backup, and monitoring retention policies for uploaded
  files and generated/transcribed content. The app's clear action removes
  session-held app data; it cannot clear data retained by infrastructure.
- Recheck the upload limits, resource usage, and model checkpoints for the
  target environment before launch.

No legal compliance, medical accuracy, or deployment security claim is made.

## Diffusion synthetic image model

The diffusion trainer intentionally does **not** use
`dataset/RxHandBD-ML/RxHandBD-ML`: its train/test CSVs map prescription images
to handwritten transcription text, which could teach a generator to reproduce
real medication or patient content. Instead, it creates procedural,
text-free form layouts in memory. Generated previews receive visible
synthetic/not-a-prescription watermarks. This reduces exposure to real
identities and clinical text; it is not a guarantee that arbitrary model output
can never resemble meaningful writing.

Train from the project root with the existing PyTorch installation. The
default run uses 100 epochs, 512 procedural forms, batch size 16, 128-pixel
training images, and 100 diffusion timesteps. CPU training is supported but
can take a long time; use Colab or another CUDA-enabled environment for the
full run. Checkpoints are saved every five epochs, and loss history plus
watermarked samples/comparison grids are saved for monitoring.

```powershell
.\.venv\Scripts\python.exe -m models.train_diffusion --epochs 100 --samples 512 --batch-size 8 --image-size 128 --learning-rate 2e-4 --seed 42 --checkpoint models/diffusion_model_128.pth --resume-from models/diffusion_model.pth --checkpoint-interval 5 --sample-interval 10
```

The existing 64-pixel checkpoint may be used as a weight warm start at the
new 128-pixel resolution; the trainer intentionally starts a fresh optimizer
and validation-loss history because this run uses a changed resolution and
expanded procedural layouts. It writes to a separate candidate checkpoint so
the currently configured model remains untouched. After inspecting samples,
point `DIFFUSION_MODEL_PATH` at the candidate to use it in Streamlit. To
continue a compatible 128-pixel run, reuse its checkpoint and set the target
epoch count higher than the recorded epoch, for example:

```powershell
.\.venv\Scripts\python.exe -m models.train_diffusion --epochs 150 --samples 512 --batch-size 8 --image-size 128 --learning-rate 2e-4 --seed 42 --checkpoint models/diffusion_model_128.pth --resume --checkpoint-interval 5 --sample-interval 10
```

Override settings with `--epochs`, `--samples`, `--batch-size`, `--image-size`,
`--timesteps`, `--base-channels`, `--learning-rate`, and `--seed`.
`--resume-from` warm-starts from a compatible checkpoint, including a
different image size; `--resume` restores optimizer/EMA/loss state for a
matching-resolution checkpoint. Checkpoints (`*.pth`) are ignored by Git and
should not be committed.

Training and validation velocity-MSE are written to
`<checkpoint-stem>_training_history.csv`. Every ten epochs (and at the final
epoch), the trainer saves two watermarked samples and updates an
epoch-labeled `training_comparison_grid.png` under
`generated_images/diffusion/<checkpoint-stem>_training_progress/`. Review the
actual samples as well as the curves; validation loss is not a measure of
perceptual image quality.

For an interactive Colab or local Jupyter workflow, open
[`notebooks/Diffusion_Model_Training.ipynb`](./notebooks/Diffusion_Model_Training.ipynb).
It inspects only the existing dataset's folder counts and CSV headers, shows
procedural training examples, checks preprocessing and diffusion equations,
runs a train/save/reload/sample smoke test, trains/resumes, plots loss, and
displays the stage comparison grid and inference samples. In Colab, mount
Drive for persistent checkpoints and outputs. The notebook defaults to a
100-epoch 128-pixel run and uses the existing 64-pixel checkpoint only as an
optional weight warm start when that local/Drive file is available.

The checkpoint stores the U-Net architecture, sinusoidal timestep embedding,
grayscale `[-1, 1]` normalization, cosine alpha-bar schedule, velocity
prediction objective, EMA and optimizer state, training epoch/sample counts,
and validation velocity-MSE. Inference validates the architecture and
preprocessing contract and samples in reverse timestep order. Generated
images are visibly watermarked; this labeling does not guarantee that every
possible output is free of ambiguous visual marks.

Generate images from a trained local checkpoint:

```powershell
.\.venv\Scripts\python.exe -m models.diffusion_inference --count 4 --steps 100 --seed 42
```

Use all 100 reverse steps for the configured checkpoint. Fewer steps are
supported for experimentation but can reduce sample quality. Images are saved
under `generated_images/diffusion/`. Start the app with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Use **Diffusion Synthetic Lab** in the sidebar. The page reports the loaded
checkpoint's resolution, epoch count, procedural sample count, and validation
velocity-MSE; it gives a specific non-sensitive message if the checkpoint is
missing or incompatible.

For deployment, provide the checkpoint without committing it:

- Set `DIFFUSION_MODEL_PATH` to a mounted private checkpoint file; or
- Put the checkpoint in a private Hugging Face model repository and configure
  `DIFFUSION_MODEL_ID`, optional `DIFFUSION_MODEL_FILENAME` (defaults to
  `diffusion_model.pth`), and `HF_TOKEN` in the deployment secret manager or
  Streamlit secrets. The runtime downloads only that checkpoint file.

The application requires an existing checkpoint locally or one of these
deployment configurations before enabling diffusion generation. The local
repository checkpoint is a 64-pixel model; the new 128-pixel candidate must
be trained and visually inspected before configuring it for deployment.
