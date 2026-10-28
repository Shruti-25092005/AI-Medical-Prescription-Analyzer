import io
import streamlit as st
import numpy as np

from models.synthetic_record_gan import (
    CHECKPOINT_PATH,
    SYNTHETIC_OUTPUT_LABEL,
    generate_records,
)
from models.handwriting_gan import (
    load_handwriting_gan,
    generate_handwriting_samples,
    CHECKPOINT_PATH as HW_GAN_PATH,
)


st.markdown(
    """
    <div style="margin-bottom:1.1rem;">
      <div style="font-size:.76rem; text-transform:uppercase; letter-spacing:.12em;
          font-weight:700; color:#167d78;">Research workspace</div>
      <h1 style="margin:.45rem 0 .25rem;">Synthetic Prescription GAN Lab</h1>
      <p style="max-width:760px; margin:0;">
        Synthesize genuine generative outputs directly from the trained GAN checkpoints.
        All outputs are synthetic research artifacts designed for academic benchmarking and zero-PHI privacy preservation.
        No template text or fabricated patient details are used.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info(
    f"{SYNTHETIC_OUTPUT_LABEL}. Displays genuine neural network outputs from the trained GAN generators."
)

with st.container(border=True):
    st.subheader("Generate Synthetic Samples from Trained Checkpoints")
    target_mode = st.radio(
        "Generator Checkpoint",
        [
            "Continuous Handwriting Strokes (RxHandBD-ML DCGAN Checkpoint)",
            "Form Layout Structure (Synthetic Form DCGAN Checkpoint)",
        ],
        horizontal=True,
    )

    controls = st.columns([1, 1, 2])
    count = controls[0].number_input(
        "Number of samples",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
    )
    rec_seed = controls[1].number_input(
        "Synthesis Seed",
        min_value=0,
        max_value=999999,
        value=42,
    )
    generate_clicked = controls[2].button(
        "Generate GAN Samples",
        type="primary",
        icon=":material/auto_awesome:",
        use_container_width=True,
    )

    if generate_clicked or "synthetic_lab_samples" not in st.session_state:
        try:
            with st.spinner("Generating samples from trained GAN checkpoint..."):
                if target_mode.startswith("Continuous Handwriting"):
                    gen_model, _ = load_handwriting_gan(HW_GAN_PATH)
                    gen_imgs = generate_handwriting_samples(
                        gen_model,
                        count=int(count),
                        seed=int(rec_seed),
                        clarity_boost=True,
                    )
                    st.session_state["synthetic_lab_samples"] = [
                        (f"GAN-HW-{rec_seed}-{i+1}", img) for i, img in enumerate(gen_imgs)
                    ]
                else:
                    records = generate_records(
                        count=int(count),
                        seed=int(rec_seed),
                    )
                    st.session_state["synthetic_lab_samples"] = [
                        (rec.record_id, img) for rec, img in records
                    ]
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            st.error(f"GAN generation failed: {exc}")

generated_samples = st.session_state.get("synthetic_lab_samples", [])
if generated_samples:
    from models.metrics_evaluator import (
        compute_generative_diversity,
        detect_generation_failure,
        evaluate_image_quality,
    )

    rec_images = [img for _, img in generated_samples]
    diversity_val = compute_generative_diversity(rec_images)
    quals = [evaluate_image_quality(img) for img in rec_images]
    mean_sharp = float(np.mean([q["sharpness"] for q in quals])) if quals else 0.0
    mean_contrast = float(np.mean([q["contrast"] for q in quals])) if quals else 0.0
    mean_cov = float(np.mean([q.get("foreground_coverage_pct", 0.0) for q in quals])) if quals else 0.0
    fail_rep = detect_generation_failure(rec_images)

    st.subheader("Synthesized GAN Sample Gallery")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Generated Count", str(len(generated_samples)))
    mc2.metric("Pairwise Diversity", f"{diversity_val:.4f} L1", help="L1 diversity across generated batch.")
    mc3.metric("Mean Sharpness", f"{mean_sharp:.1f}", help="Laplacian variance.")
    mc4.metric("Mean Contrast", f"{mean_contrast:.1f}", help="RMS luminance contrast.")
    mc5.metric("Foreground Coverage", f"{mean_cov:.1f}%", help="Percentage of canvas occupied by dark strokes.")

    if fail_rep.get("is_failure"):
        st.warning(f"⚠️ **Generation Limitations Detected:** {', '.join(fail_rep.get('failure_reasons', []))}")
    else:
        st.success("✅ **Quality Assessment:** Meaningfully varied synthetic samples produced directly from trained generator checkpoint.")

    sample_columns = st.columns(min(4, len(generated_samples)))
    for index, (sample_id, image) in enumerate(generated_samples):
        col = sample_columns[index % len(sample_columns)]
        col.image(
            image,
            caption=f"{sample_id} (Synthetic Research Artifact)",
            use_container_width=True,
        )
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        col.download_button(
            label=f"Download {sample_id} (PNG)",
            data=buf.getvalue(),
            file_name=f"{sample_id}.png",
            mime="image/png",
            key=f"dl_lab_{index}",
            use_container_width=True,
        )
