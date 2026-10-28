import time

import streamlit as st

from models.diffusion_inference import (
    SYNTHETIC_OUTPUT_LABEL,
    configured_checkpoint_description,
    configured_checkpoint_path,
    generate_image_comparisons,
)
from models.gui_services import load_model_bundle, model_statuses


st.markdown(
    """
    <div style="margin-bottom:1.1rem;">
      <div style="font-size:.76rem; text-transform:uppercase; letter-spacing:.12em;
          font-weight:700; color:#167d78;">Synthetic research workspace</div>
      <h1 style="margin:.45rem 0 .25rem;">Diffusion image lab</h1>
      <p style="max-width:760px; margin:0;">
        Synthesize fictional medical prescription documents, doctor cursive handwriting, and document form geometry from Gaussian noise.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    f"⚠️ {SYNTHETIC_OUTPUT_LABEL}. All generated records and handwriting crops "
    "are purely synthetic AI research artifacts and must never be used for clinical decision-making."
)

diffusion_status = next(
    model for model in model_statuses() if model["id"] == "diffusion"
)
if not diffusion_status["available"]:
    st.warning(
        "No diffusion checkpoint is configured. Train one with "
        "`python -m models.train_diffusion` or configure private model storage."
    )
    st.stop()

st.success(
    f"Diffusion checkpoint active: {diffusion_status['checkpoint']}"
)

diff_target = st.radio(
    "Diffusion Synthesis Target",
    [
        "Doctor Handwritten Cursive Crops (RxHandBD-ML Style DDPM)",
        "Document Form Layout Structure (128×128 Spatial U-Net Benchmark)",
    ],
    index=0,
    horizontal=True,
    help="Select whether to synthesize doctor cursive words via reverse DDPM/DDIM diffusion or structural form layout geometry.",
)

if diff_target.startswith("Doctor Handwritten"):
    st.info(
        "✍️ **Doctor Handwritten Cursive Crops:** Synthesizes doctor cursive prescription words (RxHandBD-ML clinical style) "
        "via iterative DDIM reverse diffusion from Gaussian noise, yielding dark cursive ink strokes on paper."
    )
else:
    st.info(
        "📐 **Document Form Layout Structure:** Benchmarks the spatial U-Net denoiser's ability to reconstruct "
        "document layout geometry (borders, header blocks, and grid lines) from pure Gaussian noise without text."
    )

controls = st.columns(3)
count = controls[0].number_input(
    "Number of samples",
    min_value=1,
    max_value=4,
    value=2,
    step=1,
)
sampling_steps = controls[1].slider(
    "Sampling steps",
    min_value=5,
    max_value=100,
    value=25,
    step=5,
)
seed = controls[2].number_input(
    "Random seed",
    min_value=0,
    max_value=2_147_483_647,
    value=42,
    step=1,
)

if st.button(
    "Generate synthetic images",
    type="primary",
    icon=":material/auto_awesome:",
    use_container_width=True,
):
    st.session_state.pop("diffusion_image_comparisons", None)
    st.session_state.pop("diffusion_active_target", None)
    try:
        with st.spinner("Executing reverse diffusion denoising pipeline..."):
            sampling_started = time.perf_counter()

            if diff_target.startswith("Doctor Handwritten"):
                from models.handwriting_diffusion import load_handwriting_diffusion, generate_diffusion_samples, CHECKPOINT_PATH as HW_DIFF_PATH
                device = "cpu"
                hw_model = load_handwriting_diffusion(HW_DIFF_PATH, device)
                samples, trajs = generate_diffusion_samples(
                    hw_model,
                    count=int(count),
                    steps=int(min(sampling_steps, 35)),
                    seed=int(seed),
                    clarity_boost=True,
                    contrast=1.0,
                )
                comparisons = []
                for idx, (sample_img, traj) in enumerate(zip(samples, trajs)):
                    noise_img = traj[0] if traj else sample_img
                    comparisons.append((noise_img, sample_img))
                sampling_seconds = time.perf_counter() - sampling_started
                st.session_state["diffusion_sampling_steps"] = int(min(sampling_steps, 35))
                st.session_state["diffusion_image_size"] = 256
                st.session_state["diffusion_trajectories"] = trajs

            else:
                checkpoint_path = configured_checkpoint_path()
                if checkpoint_path.is_file():
                    checkpoint_stats = checkpoint_path.stat()
                    checkpoint_revision = (
                        f"{configured_checkpoint_description()}:"
                        f"{checkpoint_stats.st_mtime_ns}:"
                        f"{checkpoint_stats.st_size}"
                    )
                else:
                    checkpoint_revision = configured_checkpoint_description()
                model_bundle = load_model_bundle(
                    "diffusion",
                    checkpoint_revision=checkpoint_revision,
                )
                if int(sampling_steps) > int(model_bundle["timesteps"]):
                    raise ValueError(
                        "Sampling steps cannot exceed checkpoint timesteps."
                    )
                comparisons, trajs = generate_image_comparisons(
                    count=int(count),
                    sampling_steps=int(sampling_steps),
                    seed=int(seed),
                    model_bundle=model_bundle,
                    return_trajectories=True,
                )
                sampling_seconds = time.perf_counter() - sampling_started
                st.session_state["diffusion_sampling_steps"] = min(
                    int(sampling_steps),
                    int(model_bundle["timesteps"]),
                )
                st.session_state["diffusion_image_size"] = int(model_bundle["image_size"])
                st.session_state["diffusion_trajectories"] = trajs

    except ValueError as exc:
        st.error(f"Diffusion model configuration error: {exc}")
    except FileNotFoundError:
        st.error(
            "The configured diffusion checkpoint is missing or unavailable. "
            "Check the local path or private model-storage settings."
        )
    except ImportError:
        st.error(
            "A required diffusion dependency is unavailable. Install the "
            "project requirements and restart the app."
        )
    except (OSError, RuntimeError, TypeError) as exc:
        st.error(
            f"The diffusion model could not be loaded or sampling failed: {exc}. "
            "Check the checkpoint file and available memory, then retry."
        )
    except Exception as exc:
        st.error(f"Unexpected diffusion generation error: {exc}")
    else:
        st.session_state["diffusion_image_comparisons"] = comparisons
        st.session_state["diffusion_sampling_seconds"] = sampling_seconds
        st.session_state["diffusion_active_target"] = diff_target

comparisons = st.session_state.get("diffusion_image_comparisons", [])
active_target = st.session_state.get("diffusion_active_target", "Complete Fictional Prescription Records")
if comparisons:
    st.markdown("---")
    st.subheader("Denoising Synthesis Results")
    st.caption(
        f"Target: **{active_target}** · Completed {st.session_state.get('diffusion_sampling_steps', 25)} denoising "
        f"steps in {st.session_state.get('diffusion_sampling_seconds', 0.5):.2f} seconds."
    )
    
    from models.metrics_evaluator import (
        compute_generative_diversity,
        detect_generation_failure,
        evaluate_image_quality,
    )
    generated_imgs = [pair[1] for pair in comparisons]
    diversity_val = compute_generative_diversity(generated_imgs)
    qualities = [evaluate_image_quality(img) for img in generated_imgs]
    import numpy as np
    mean_sharp = float(np.mean([q["sharpness"] for q in qualities])) if qualities else 0.0
    mean_contrast = float(np.mean([q["contrast"] for q in qualities])) if qualities else 0.0
    fail_report = detect_generation_failure(generated_imgs)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Resolution", f"{comparisons[0][1].width} × {comparisons[0][1].height}")
    mc2.metric("Pairwise Diversity", f"{diversity_val:.4f} L1", help="Pairwise L1 pixel distance across generated samples.")
    mc3.metric("Mean Sharpness", f"{mean_sharp:.1f}", help="Variance of Laplacian stroke edge definition.")
    mc4.metric("Mean Contrast", f"{mean_contrast:.1f}", help="RMS luminance contrast.")

    if fail_report.get("is_failure"):
        st.warning(f"⚠️ **Generation Limitations Detected:** {', '.join(fail_report.get('failure_reasons', []))}")
    else:
        st.success("✅ **Quality Verification:** Model produced varied, legible outputs without mode collapse.")

    for index, (noise_image, generated_image) in enumerate(comparisons, start=1):
        with st.container(border=True):
            st.markdown(f"##### Sample #{index} · Latent Denoising Pair")
            noise_column, output_column = st.columns(2)
            noise_column.image(
                noise_image,
                caption="Initial Gaussian noise (before denoising)",
                use_container_width=True,
            )
            if active_target.startswith("Complete Fictional"):
                out_caption = f"Synthesized Complete HD Prescription Document — Sample #{index}"
            elif active_target.startswith("Doctor Handwritten"):
                out_caption = f"Synthesized Doctor Cursive Handwriting — Sample #{index}"
            else:
                out_caption = f"Synthesized Document Form Skeleton (128×128 Geometry Benchmark) — Sample #{index}"

            output_column.image(
                generated_image,
                caption=out_caption,
                use_container_width=True,
            )
            if st.session_state.get("diffusion_trajectories") and index - 1 < len(st.session_state["diffusion_trajectories"]):
                sample_traj = st.session_state["diffusion_trajectories"][index - 1]
                if len(sample_traj) > 1:
                    with st.expander(f"🔬 View Step-by-Step Denoising Trajectory (Sample #{index})", expanded=False):
                        st.caption("Iterative DDIM reverse diffusion stripping away Gaussian noise at each timestep:")
                        cols = st.columns(len(sample_traj))
                        for s_idx, f_img in enumerate(sample_traj):
                            cols[s_idx].image(f_img, caption=f"Step {s_idx + 1}/{len(sample_traj)}", use_container_width=True)

            try:
                import io as io_mod
                buf = io_mod.BytesIO()
                generated_image.save(buf, format="PNG")
                output_column.download_button(
                    label=f"Download Sample #{index} (PNG)",
                    data=buf.getvalue(),
                    file_name=f"diffusion_sample_{index}.png",
                    mime="image/png",
                    key=f"dl_diff_sample_{index}",
                    use_container_width=True,
                )
            except Exception:
                pass
    st.caption(
        "Generated output is informational demonstration content only. "
        "It contains no intentionally generated names, medicines, dosages, "
        "or clinical instructions and must never be used for care."
    )
