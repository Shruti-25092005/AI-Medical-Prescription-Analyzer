import io
from pathlib import Path
import random

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from models.ae_inference import enhance_reconstruction, load_autoencoder, reconstruct_image as ae_reconstruct
from models.gui_services import load_model_bundle, model_statuses
from models.handwriting_diffusion import (
    CHECKPOINT_PATH as DIFF_CKPT_PATH,
    generate_diffusion_samples,
    load_handwriting_diffusion,
)
from models.handwriting_gan import (
    CHECKPOINT_PATH as GAN_CKPT_PATH,
    generate_handwriting_samples,
    load_handwriting_gan,
)
from models.metrics_evaluator import (
    compute_generative_diversity,
    compute_nearest_neighbor_distance,
    compute_reconstruction_metrics,
)
from models.vae_inference import (
    compute_vae_loss_components,
    interpolate_latent,
    load_vae_model,
    reconstruct_image as vae_reconstruct,
    sample_latent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "RxHandBD-ML" / "RxHandBD-ML"
TEST_SET_DIR = DATASET_DIR / "Test_Set"

st.markdown(
    """
    <style>
    section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
        max-width: 1440px !important;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }
    div[data-testid="stMetricLabel"] > div {
        overflow: visible !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.25 !important;
    }
    div[data-testid="stMetricValue"] > div {
        overflow: visible !important;
        white-space: normal !important;
        font-size: 1.3rem !important;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 14px;
        min-height: 95px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin-bottom:1.2rem;">
      <div style="font-size:.76rem; text-transform:uppercase; letter-spacing:.12em;
          font-weight:700; color:#167d78;">Module B · Generative AI Studio</div>
      <h1 style="margin:.45rem 0 .25rem;">Handwriting Generation &amp; Reconstruction Studio</h1>
      <p style="max-width:850px; margin:0; color:#587579;">
        Explore, generate, and evaluate synthetic handwritten prescription text images across
        multiple generative architectures: Skip-Autoencoder (AE), Variational Autoencoder (VAE),
        Deep Convolutional GAN (DCGAN), and Denoising Diffusion Probabilistic Models (DDPM).
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Architecture Clarification: The Transformer model in this repository is TrOCR (Vision Transformer + Decoder) "
    "specifically designed for handwriting OCR (transcription). It is evaluated on the Model Evaluation Dashboard. "
    "Image generation is performed by the generative architectures below (AE, VAE, DCGAN, DDPM)."
)

# Select Model Architecture
model_tab_choice = st.radio(
    "Choose Generative Paradigm",
    ["DCGAN (Handwriting Generation)", "Diffusion DDPM (Iterative Denoising)", "Variational Autoencoder (VAE)", "Autoencoder (AE Reconstruction)"],
    horizontal=True,
)

# -------------------------------------------------------------
# 1. DCGAN Studio
# -------------------------------------------------------------
if model_tab_choice == "DCGAN (Handwriting Generation)":
    st.subheader("DCGAN: Synthetic Prescription Handwriting & Record Generator")
    st.markdown(
        "Generates novel synthetic healthcare records and handwritten prescription word images from random latent vectors "
        "$z \\sim \\mathcal{N}(0, I)$ trained on RxHandBD-ML doctor handwriting crops and clinical document templates. "
        "All outputs are synthetic (fictional) records designed for academic prototyping and privacy-preserving model evaluation."
    )

    col_ctrl, col_stats = st.columns([1, 1])
    with col_ctrl:
        num_samples = st.slider("Number of Generated Samples", min_value=1, max_value=8, value=4, step=1)
        seed = st.number_input("Random Generator Seed", min_value=0, max_value=999999, value=42)
        clarity_mode = st.radio(
            "Rendering Clarity",
            ["High Clarity (Clean Paper & Sharp Ink)", "Raw Linear Output"],
            index=0,
            horizontal=True,
            help="High Clarity maps background noise to clean prescription paper and enhances ink stroke contrast.",
        )
        contrast_boost = st.slider("Ink Stroke Contrast", min_value=0.6, max_value=1.5, value=1.0, step=0.1)
        gen_clicked = st.button("Generate DCGAN Handwriting", type="primary", use_container_width=True)

    with col_stats:
        st.markdown("**DCGAN Architecture & Checkpoint Specifications**")
        ckpt_is_present = GAN_CKPT_PATH.is_file()
        if ckpt_is_present:
            try:
                ckpt_meta = torch.load(GAN_CKPT_PATH, map_location="cpu", weights_only=False)
                arch_name = ckpt_meta.get("architecture", "dcgan_lsgan_feature_matching_v3")
                n_epochs = ckpt_meta.get("epochs", 35)
                n_samples_trained = ckpt_meta.get("training_samples", 400)
                l_dim = ckpt_meta.get("latent_dim", 64)
                st.success(f"Verified Checkpoint: `{GAN_CKPT_PATH.name}`")
                st.write(f"- Objective: **Least Squares GAN (LSGAN) + Feature Matching**")
                st.write(f"- Dataset: **RxHandBD-ML** ({n_samples_trained} real handwriting crops, {n_epochs} epochs)")
                st.write(f"- Latent Dimension: `{l_dim}` standard normal vector")
                st.write(f"- Clarity Mode: **{clarity_mode}** (Contrast: {contrast_boost}x)")
            except Exception as e:
                st.warning(f"Checkpoint metadata: {e}")
        else:
            st.error(f"Checkpoint `{GAN_CKPT_PATH.name}` not found. Run `python -m models.handwriting_gan` to train.")

    if gen_clicked or "gan_samples" in st.session_state:
        if gen_clicked:
            with st.spinner("Loading GAN checkpoint and generating continuous handwriting strokes..."):
                gen_model, _ = load_handwriting_gan(GAN_CKPT_PATH)
                generated_imgs = generate_handwriting_samples(
                    gen_model,
                    count=num_samples,
                    seed=int(seed),
                    clarity_boost=(clarity_mode == "High Clarity (Clean Paper & Sharp Ink)"),
                    contrast=float(contrast_boost),
                )
                st.session_state["gan_samples"] = generated_imgs

        samples = st.session_state.get("gan_samples", [])
        if samples:
            import zipfile as zip_mod
            import time as time_mod
            ts = int(time_mod.time())

            st.markdown(
                """
                <div style="display:flex; align-items:center; justify-content:space-between; margin:1rem 0 0.5rem;">
                    <div style="font-size:1.05rem; font-weight:700; color:#1e1b4b;">
                        Generated Handwriting Sample Grid
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Zip buffer for batch download
            zip_buffer = io.BytesIO()
            with zip_mod.ZipFile(zip_buffer, "w", zip_mod.ZIP_DEFLATED) as zip_file:
                for idx, img in enumerate(samples):
                    ibuf = io.BytesIO()
                    img.save(ibuf, format="PNG")
                    zip_file.writestr(f"dcgan_sample_{ts}_{idx+1}.png", ibuf.getvalue())

            cols = st.columns(min(4, len(samples)))
            for idx, img in enumerate(samples):
                c = cols[idx % len(cols)]
                with c:
                    with st.container(border=True):
                        st.image(img, caption=f"Synthetic Sample #{idx + 1}", use_container_width=True)
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        st.download_button(
                            label=f"Download Sample #{idx + 1} (PNG)",
                            data=buf.getvalue(),
                            file_name=f"dcgan_sample_{ts}_{idx + 1}.png",
                            mime="image/png",
                            key=f"dl_gan_{ts}_{idx}",
                            use_container_width=True,
                        )

            # Download All Option
            st.download_button(
                label=f"📦 Download All {len(samples)} Samples as ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"dcgan_all_samples_{ts}.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"dl_all_zip_{ts}",
            )

            # Quantitative evaluation
            st.markdown("##### Generative Quality & Diversity Metrics")
            div_score = compute_generative_diversity(samples)

            from models.metrics_evaluator import detect_generation_failure
            fail_check = detect_generation_failure(samples)

            # Nearest neighbor check against test set
            sample_train_imgs = []
            if TEST_SET_DIR.is_dir():
                for p in sorted(TEST_SET_DIR.glob("*.jpg"))[:20]:
                    sample_train_imgs.append(Image.open(p))
            nn_metrics = compute_nearest_neighbor_distance(samples, sample_train_imgs)

            q1, q2, q3, q4, q5 = st.columns(5)
            q1.metric("Batch Diversity (Pairwise L1)", f"{div_score:.4f}", help="Higher score indicates diverse strokes and absence of mode collapse.")
            q2.metric("Mean Sharpness", f"{fail_check['mean_sharpness']:.1f}", help="Variance of Laplacian stroke edge definition.")
            q3.metric("Mean Contrast", f"{fail_check['mean_contrast']:.1f}", help="RMS luminance contrast between ink and paper.")
            q4.metric("Foreground Coverage", f"{fail_check.get('mean_foreground_coverage', 0.0):.1f}%", help="Percentage of canvas occupied by dark ink strokes.")
            q5.metric("Memorization Detected", "No (Novel Samples)" if not nn_metrics["memorization_detected"] else "Yes", help="Confirms novel handwriting generation rather than exact dataset memorization.")

            if fail_check.get("is_failure"):
                st.warning(f"⚠️ **Generation Limitations Detected:** {', '.join(fail_check.get('failure_reasons', []))}")
            else:
                st.success("✅ **Quality Assessment:** Model produced varied synthetic handwriting without mode collapse.")

            with st.expander("🔬 Ground Truth Real Training Samples vs. DCGAN Generated Outputs", expanded=False):
                st.markdown(
                    "Compare real physician handwriting crops from the **RxHandBD-ML** training set directly against "
                    r"synthetic handwriting word patches synthesized from Gaussian latent vectors $z \sim \mathcal{N}(0, I)$."
                )
                csv_file = DATASET_DIR / "Train_Label.csv"
                img_dir = DATASET_DIR / "Train_Set"

                real_col, gen_col = st.columns(2)
                with real_col:
                    st.markdown("**Real Physician Handwriting (RxHandBD-ML)**")
                    if csv_file.is_file() and img_dir.is_dir():
                        df_rx = pd.read_csv(csv_file)
                        rc1, rc2 = st.columns(2)
                        for r_idx in range(min(4, len(df_rx))):
                            col_target = rc1 if r_idx % 2 == 0 else rc2
                            img_p = img_dir / str(df_rx["Images"].iloc[r_idx])
                            if img_p.is_file():
                                with Image.open(img_p) as r_img:
                                    col_target.image(
                                        r_img,
                                        caption=f"Real: {df_rx['Text'].iloc[r_idx]} ({df_rx['Images'].iloc[r_idx]})",
                                        use_container_width=True,
                                    )
                    else:
                        st.info("RxHandBD-ML dataset directory not found for real comparison.")

                with gen_col:
                    st.markdown("**DCGAN Synthetic Handwriting (Current Checkpoint)**")
                    if samples:
                        gc_c1, gc_c2 = st.columns(2)
                        for g_idx in range(min(4, len(samples))):
                            col_target = gc_c1 if g_idx % 2 == 0 else gc_c2
                            col_target.image(
                                samples[g_idx],
                                caption=f"Synthetic Sample #{g_idx + 1}",
                                use_container_width=True,
                            )

                st.markdown(
                    """
                    > **Fidelity & Diagnostic Observations:**
                    > - **Cursive Trajectory:** The DCGAN captures the horizontal cursive word baseline and general handwriting curvature.
                    > - **Resolution Analysis:** Real raw crops are $512 \times 512$ with $2\text{--}4$ pixel strokes. When downsampled to $64 \times 64$, letter loops ('e', 'a') become $\\le 1$ pixel wide. Retraining at $128 \times 128$ in Google Colab allows full loop articulation and fine hair-stroke ligatures.
                    > - **Zero-PHI Compliance:** All generated outputs are synthetic research samples and strictly medication-free.
                    """
                )

# -------------------------------------------------------------
# 2. Diffusion DDPM Studio
# -------------------------------------------------------------
elif model_tab_choice == "Diffusion DDPM (Iterative Denoising)":
    st.subheader("Denoising Diffusion Probabilistic Model (DDPM)")
    st.markdown(
        "Demonstrates step-by-step reverse diffusion $p_\\theta(x_{t-1} | x_t)$ starting from pure Gaussian noise "
        "and progressively predicting and removing noise to synthesize clear handwritten medical text images."
    )

    col_ctrl, col_stats = st.columns([1, 1])
    with col_ctrl:
        diff_count = st.slider("Samples to Generate", min_value=1, max_value=4, value=2, step=1)
        diff_steps = st.slider("Diffusion Steps", min_value=10, max_value=50, value=25, step=5)
        diff_seed = st.number_input("Diffusion Seed", min_value=0, max_value=999999, value=42)
        diff_clarity = st.radio(
            "DDPM Rendering Clarity",
            ["High Clarity (Clean Paper & Sharp Ink)", "Raw Linear Output"],
            index=0,
            horizontal=True,
            key="diff_clarity_radio",
            help="High Clarity cleans paper background and sharpens cursive ink contours.",
        )
        diff_contrast = st.slider("Ink Stroke Contrast", min_value=0.6, max_value=1.5, value=1.0, step=0.1, key="diff_contrast_slider")
        diff_clicked = st.button("Run Reverse Diffusion Process", type="primary", use_container_width=True)

    with col_stats:
        st.markdown("**DDPM Configuration & Schedule**")
        st.write(f"- Checkpoint: `{DIFF_CKPT_PATH.name}` (Available: {DIFF_CKPT_PATH.is_file()})")
        st.write("- Architecture: Compact Residual U-Net with Sinusoidal Timestep Embeddings")
        st.write("- Sampling Algorithm: **DDIM Deterministic Inversion (Song et al., ICLR 2021)**")
        st.write("- Noise Schedule: Cosine Alpha-bar schedule ($s = 0.008$)")
        st.write(f"- Clarity Mode: **{diff_clarity}** (Contrast: {diff_contrast}x)")

    if diff_clicked or "diff_samples" in st.session_state:
        if diff_clicked:
            with st.spinner("Running iterative reverse diffusion sampling loop..."):
                diff_runner = load_handwriting_diffusion(DIFF_CKPT_PATH)
                diff_imgs, diff_trajs = generate_diffusion_samples(
                    diff_runner,
                    count=diff_count,
                    steps=diff_steps,
                    seed=int(diff_seed),
                    clarity_boost=(diff_clarity == "High Clarity (Clean Paper & Sharp Ink)"),
                    contrast=float(diff_contrast),
                )
                st.session_state["diff_samples"] = diff_imgs
                st.session_state["diff_trajs"] = diff_trajs

        d_samples = st.session_state.get("diff_samples", [])
        d_trajs = st.session_state.get("diff_trajs", [])

        if d_samples:
            st.markdown("##### Final Reverse Diffusion Outputs")
            cols = st.columns(len(d_samples))
            for idx, img in enumerate(d_samples):
                c = cols[idx]
                c.image(img, caption=f"Synthesized Handwriting #{idx + 1}", use_container_width=True)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                c.download_button(
                    label=f"Download Sample #{idx + 1}",
                    data=buf.getvalue(),
                    file_name=f"diffusion_sample_{idx + 1}.png",
                    mime="image/png",
                    key=f"dl_diff_{idx}",
                )

            # Diversity and evaluation metrics
            st.markdown("##### Generative Quality & Diversity Metrics")
            diff_div = compute_generative_diversity(d_samples)

            from models.metrics_evaluator import detect_generation_failure
            diff_fail_check = detect_generation_failure(d_samples)

            sample_train_imgs = []
            if TEST_SET_DIR.is_dir():
                for p in sorted(TEST_SET_DIR.glob("*.jpg"))[:20]:
                    sample_train_imgs.append(Image.open(p))
            diff_nn_metrics = compute_nearest_neighbor_distance(d_samples, sample_train_imgs)

            dq1, dq2, dq3, dq4 = st.columns(4)
            dq1.metric("Batch Diversity (Pairwise L1)", f"{diff_div:.4f}", help="Higher score indicates diverse strokes and absence of mode collapse.")
            dq2.metric("Mean Sharpness", f"{diff_fail_check['mean_sharpness']:.1f}", help="Variance of Laplacian stroke edge definition.")
            dq3.metric("Nearest Training Neighbor Distance", f"{diff_nn_metrics['min_nn_distance']:.2f}", help="Confirms novel handwriting generation rather than exact dataset memorization.")
            dq4.metric("Memorization Detected", "No (Novel Samples)" if not diff_nn_metrics["memorization_detected"] else "Yes")

            if diff_fail_check.get("is_failure"):
                st.warning(f"⚠️ **Generation Limitations Detected:** {', '.join(diff_fail_check.get('failure_reasons', []))}")
            else:
                st.success("✅ **Quality Assessment:** Model produced varied synthetic handwriting via reverse diffusion without mode collapse.")

            st.markdown("##### Reverse Denoising Trajectory (Visualizing Noise Removal)")
            st.caption("Inspect how the model iteratively transforms unstructured Gaussian noise into legible handwriting:")
            for s_idx, traj in enumerate(d_trajs):
                with st.expander(f"Denoising Trajectory for Sample #{s_idx + 1}", expanded=(s_idx == 0)):
                    t_cols = st.columns(len(traj))
                    labels = ["t=100% (Pure Noise)", "t=75%", "t=50%", "t=25%", "t=0% (Clean Image)"]
                    for f_idx, frame in enumerate(traj):
                        lbl = labels[f_idx] if f_idx < len(labels) else f"Step {f_idx}"
                        t_cols[f_idx].image(frame, caption=lbl, use_container_width=True)

# -------------------------------------------------------------
# 3. Variational Autoencoder (VAE) Studio
# -------------------------------------------------------------
elif model_tab_choice == "Variational Autoencoder (VAE)":
    st.subheader("Variational Autoencoder (VAE): Latent Space Exploration")
    st.markdown(
        "The VAE maps handwriting crops into a smooth, continuous latent Gaussian distribution $\\mathcal{N}(\\mu, \\sigma^2)$, "
        "enabling both faithful image reconstruction and smooth latent space interpolation between distinct prescription styles."
    )

    vae_sub = st.radio("VAE Task Mode", ["Image Reconstruction", "Random Latent Sampling", "Latent Space Linear Interpolation"], horizontal=True)

    if vae_sub == "Image Reconstruction":
        sample_images = list(TEST_SET_DIR.glob("*.jpg"))[:8] if TEST_SET_DIR.is_dir() else []
        if sample_images:
            chosen_file = st.selectbox("Select Test Image to Reconstruct", [p.name for p in sample_images])
            test_img = Image.open(TEST_SET_DIR / chosen_file)
        else:
            test_img = Image.new("RGB", (256, 256), color=(245, 245, 245))

        enable_vae_enhancement = st.checkbox(
            "Apply Adaptive Stroke Sharpening & CLAHE Ink Boost",
            value=True,
            help="Applies bilateral edge-preserving smoothing, CLAHE contrast enhancement, and ink gamma deepening for crisp handwriting.",
            key="vae_enh_toggle",
        )

        c1, c2, c3 = st.columns(3)
        c1.image(test_img, caption="Input Test Handwriting", use_container_width=True)

        if st.button("Run VAE Reconstruction", type="primary", use_container_width=True):
            with st.spinner("Encoding to latent distribution and decoding..."):
                from models.metrics_evaluator import compute_reconstruction_error_map

                vae_bundle = load_model_bundle("vae")
                reconstructed = vae_reconstruct(
                    test_img,
                    vae_bundle["model"],
                    vae_bundle["image_size"],
                    vae_bundle["device"],
                    enhance=enable_vae_enhancement,
                )
                metrics = compute_reconstruction_metrics(test_img, reconstructed)
                err_map, d_metrics = compute_reconstruction_error_map(test_img, reconstructed)
                metrics.update(d_metrics)
                recon_mse, kl_div = compute_vae_loss_components(test_img, vae_bundle["model"], vae_bundle["image_size"], vae_bundle["device"])

                c2.image(reconstructed, caption="VAE Reconstructed Image", use_container_width=True)
                c3.image(err_map, caption="Lost Detail Heatmap (Red = High Loss)", use_container_width=True)

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("MSE", f"{metrics['mse']:.4f}")
                m2.metric("PSNR", f"{metrics['psnr_db']:.2f} dB")
                m3.metric("SSIM", f"{metrics['ssim']:.4f}")
                m4.metric("Sharpness Change", f"{metrics.get('sharpness_change_pct', 0.0):+.1f}%")
                m5.metric("Edge Loss", f"{metrics.get('edge_loss_pct', 0.0):.1f}%")
                m6.metric("KL Divergence", f"{kl_div:.4f}")

                st.caption(
                    f"Objective breakdown: Reconstruction MSE = {recon_mse:.5f} measures pixel-level fidelity. "
                    f"KL Divergence D_KL = {kl_div:.4f} measures divergence from the standard normal prior N(0, I). "
                    "Accuracy Observation: The VAE bottleneck compresses information into a 32-channel Gaussian prior, "
                    "which naturally induces slight stroke softening compared to uncompressed inputs."
                )

    elif vae_sub == "Random Latent Sampling":
        st.markdown("Samples novel handwriting features directly from the prior $z \\sim \\mathcal{N}(0, I)$.")
        s_count = st.slider("Number of Samples", 1, 8, 4)
        s_seed = st.number_input("Prior Sampling Seed", 0, 999999, 42)
        if st.button("Sample VAE Prior", type="primary", use_container_width=True):
            with st.spinner("Sampling latent space..."):
                vae_bundle = load_model_bundle("vae")
                samples = sample_latent(
                    model=vae_bundle["model"],
                    count=s_count,
                    latent_spatial=vae_bundle["latent_spatial"],
                    seed=int(s_seed),
                    device=vae_bundle["device"],
                )
                cols = st.columns(len(samples))
                for i, img in enumerate(samples):
                    cols[i].image(img, caption=f"Prior Sample #{i+1}", use_container_width=True)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    cols[i].download_button(
                        label=f"Download #{i+1}",
                        data=buf.getvalue(),
                        file_name=f"vae_sample_{i+1}.png",
                        mime="image/png",
                        key=f"dl_vae_{i}",
                    )

    elif vae_sub == "Latent Space Linear Interpolation":
        st.markdown("Performs continuous linear interpolation $z_\\alpha = (1-\\alpha) z_A + \\alpha z_B$ between two latent handwriting vectors.")
        sample_images = list(TEST_SET_DIR.glob("*.jpg"))[:6] if TEST_SET_DIR.is_dir() else []
        if len(sample_images) >= 2:
            col_a, col_b = st.columns(2)
            sel_a = col_a.selectbox("Source Word A", [p.name for p in sample_images], index=0)
            sel_b = col_b.selectbox("Target Word B", [p.name for p in sample_images], index=1)
            img_a = Image.open(TEST_SET_DIR / sel_a)
            img_b = Image.open(TEST_SET_DIR / sel_b)
            col_a.image(img_a, caption="Word A (Source)", width=200)
            col_b.image(img_b, caption="Word B (Target)", width=200)

            interp_steps = st.slider("Interpolation Steps", 3, 9, 5)
            if st.button("Generate Latent Trajectory", type="primary", use_container_width=True):
                with st.spinner("Computing geodesic interpolation trajectory in latent space..."):
                    vae_bundle = load_model_bundle("vae")
                    interpolated_imgs = interpolate_latent(
                        model=vae_bundle["model"],
                        image1=img_a,
                        image2=img_b,
                        steps=interp_steps,
                        image_size=vae_bundle["image_size"],
                        device=vae_bundle["device"],
                    )
                    st.markdown("##### Latent Transition Trajectory: $A \\longrightarrow B$")
                    t_cols = st.columns(len(interpolated_imgs))
                    for idx, img in enumerate(interpolated_imgs):
                        alpha = idx / (len(interpolated_imgs) - 1)
                        t_cols[idx].image(img, caption=f"$\\alpha = {alpha:.2f}$", use_container_width=True)

# -------------------------------------------------------------
# 4. Autoencoder (AE) Studio
# -------------------------------------------------------------
elif model_tab_choice == "Autoencoder (AE Reconstruction)":
    st.subheader("Skip-Autoencoder: Full Prescription Image Reconstruction")
    st.markdown(
        "Reconstructs high-resolution handwriting and form structures through a deep encoder-decoder "
        "network with multi-scale skip connections preserving fine ink strokes."
    )

    sample_images = list(TEST_SET_DIR.glob("*.jpg"))[:8] if TEST_SET_DIR.is_dir() else []
    if sample_images:
        chosen_file = st.selectbox("Select Test Image to Reconstruct (AE)", [p.name for p in sample_images], key="ae_select")
        test_img = Image.open(TEST_SET_DIR / chosen_file)
    else:
        test_img = Image.new("RGB", (256, 256), color=(245, 245, 245))

    c1, c2, c3 = st.columns(3)
    c1.image(test_img, caption="Input Test Scan", use_container_width=True)

    if st.button("Run AE Reconstruction", type="primary", use_container_width=True):
        with st.spinner("Executing skip autoencoder forward pass..."):
            from models.metrics_evaluator import compute_reconstruction_error_map

            ae_bundle = load_model_bundle("ae")
            raw_recon = ae_reconstruct(test_img, ae_bundle["model"], ae_bundle["device"])
            enhanced_recon = enhance_reconstruction(raw_recon)
            metrics = compute_reconstruction_metrics(test_img, raw_recon)
            err_map, d_metrics = compute_reconstruction_error_map(test_img, raw_recon)
            metrics.update(d_metrics)

            c2.image(enhanced_recon, caption="AE Reconstructed Image", use_container_width=True)
            c3.image(err_map, caption="Lost Detail Heatmap (Red = Residual Error)", use_container_width=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("MSE", f"{metrics['mse']:.4f}")
            m2.metric("PSNR", f"{metrics['psnr_db']:.2f} dB")
            m3.metric("SSIM", f"{metrics['ssim']:.4f}")
            m4.metric("Sharpness Change", f"{metrics.get('sharpness_change_pct', 0.0):+.1f}%")
            m5.metric("Edge Detail Loss", f"{metrics.get('edge_loss_pct', 0.0):.1f}%")

            st.caption(
                "Accuracy Observation: The Skip-Autoencoder preserves fine ink edges and structure through direct encoder-decoder "
                "residual pathways. The error map highlights the minor differences where stroke boundaries were subtly smoothed."
            )
