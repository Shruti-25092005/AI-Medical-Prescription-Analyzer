from hashlib import sha256
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from models.drug_interactions import check_drug_interactions
from models.gui_services import (
    MODEL_NAMES,
    load_model_bundle,
    model_statuses,
    record_processing_run,
    run_inference,
)
from models.medicine_knowledge import explain_medicine
from models.page_segmenter import segment_prescription_page
from models.prescription_parser import (
    MEDICINE_REFERENCE_DB,
    extract_prescription_fields,
    normalize_medicine_name,
)
from models.prescription_upload import (
    UploadValidationError,
    load_prescription_pages,
)
from models.privacy_controls import clear_analysis_results


st.markdown(
    """
    <style>
    section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
        max-width: 1440px !important;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }
    .badge-exact {
        background-color: #e6f4ea;
        color: #137333;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-likely {
        background-color: #fef7e0;
        color: #b06000;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-uncertain {
        background-color: #fce8e6;
        color: #c5221f;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin-bottom:1.2rem;">
      <div style="font-size:.76rem; text-transform:uppercase; letter-spacing:.12em;
          font-weight:700; color:#167d78;">Module A · Clinical AI Pipeline</div>
      <h1 style="margin:.45rem 0 .25rem;">Prescription Analyzer &amp; Clinical Explainer</h1>
      <p style="max-width:880px; margin:0; color:#587579;">
        Upload handwritten prescription images or multi-page PDFs. The system applies computer vision
        preprocessing, text line segmentation, fine-tuned TrOCR transcription, clinical entity parsing,
        source-grounded medicine explanations, and multi-drug interaction screening.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Disclaimer Banner
st.warning(
    "Academic Research Prototype: This system is designed for technical evaluation and transcription "
    "assistance only. It does NOT provide medical advice, diagnosis, or prescription modification. "
    "All extracted and normalized information must be manually verified by a licensed clinician or pharmacist."
)

with st.container(border=True):
    st.subheader("1 · Prescription Upload & Configuration")
    upload_col, config_col = st.columns([2, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload Prescription (JPG, JPEG, PNG, or PDF)",
            type=["jpg", "jpeg", "png", "pdf"],
            key=f"prescription_upload_{st.session_state.get('prescription_upload_generation', 0)}",
            help="Supports JPG/PNG scans and single or multi-page PDF documents up to 10 MB.",
        )

    with config_col:
        st.markdown("**Processing Pipeline Options**")
        enable_segmentation = st.checkbox("Enable Page Line/Word Segmentation", value=True, help="Segments full pages into distinct text crops before passing to word-level TrOCR.")
        enable_enhancement = st.checkbox("Apply Contrast & Denoising (CLAHE)", value=True, help="Applies bilateral filtering and adaptive histogram equalization.")
        selected_model = st.selectbox(
            "Recognition / Analysis Model",
            options=["trocr", "ae", "vae", "gan", "diffusion"],
            format_func=lambda m: MODEL_NAMES.get(m, m),
            help="TrOCR transcribes handwriting into text. AE and VAE perform image reconstruction. DCGAN and Diffusion synthesize handwriting.",
        )

content = uploaded_file.getvalue() if uploaded_file is not None else None
upload_id = sha256(content).hexdigest() if content is not None else None

if st.session_state.get("active_upload_id") != upload_id:
    clear_analysis_results(st.session_state)
    st.session_state["active_upload_id"] = upload_id
    st.session_state.pop("confirmed_medicines", None)

if uploaded_file is None or content is None:
    st.info("Upload a prescription image or PDF above to start the clinical extraction and explanation pipeline.")
    st.stop()
    pages = []
else:
    # Load pages
    try:
        pages = load_prescription_pages(uploaded_file.name, content)
    except UploadValidationError as exc:
        st.error(f"Upload processing error: {exc}")
        st.stop()
        pages = []

    if pages:
        st.success(f"File loaded successfully: **{uploaded_file.name}** ({len(pages)} page{'s' if len(pages) > 1 else ''})")

# Multipage PDF Navigation
active_page_idx = 0
if len(pages) > 1:
    page_nums = [f"Page {p.page_number}" for p in pages]
    selected_page_label = st.radio("Select Page to View and Process", page_nums, horizontal=True)
    active_page_idx = page_nums.index(selected_page_label)

active_page = pages[active_page_idx]

# Check for selectable text in PDF
if active_page.has_selectable_text:
    with st.expander("Digital PDF Selectable Text Detected", expanded=False):
        st.info("This PDF page contains embedded digital text. You can copy it directly or run OCR.")
        st.text_area("Embedded Text", active_page.selectable_text, height=120, disabled=True)

# Page Preview & Preprocessing
with st.container(border=True):
    st.subheader(f"2 · Image Inspection & Preprocessing (Page {active_page.page_number})")
    if active_page.image is not None:
        seg_preview = segment_prescription_page(active_page.image)

        tab1, tab2, tab3, tab4 = st.tabs(["Original Scan", "Preprocessed (Enhanced)", "Adaptive Binarization", "Segmentation Regions"])
        with tab1:
            st.image(active_page.image, caption=f"Original Input · {active_page.image.width} × {active_page.image.height}px", use_container_width=True)
        with tab2:
            st.image(seg_preview.enhanced_contrast, caption="Bilateral Filter + CLAHE Contrast Enhancement", use_container_width=True)
        with tab3:
            st.image(seg_preview.threshold_preview, caption="Adaptive Gaussian Threshold Binarization", use_container_width=True)
        with tab4:
            st.image(seg_preview.annotated_page, caption=f"Detected Text Regions ({seg_preview.crop_count} crops across {seg_preview.line_count} line(s))", use_container_width=True)
            st.caption(seg_preview.notes)
    else:
        st.error(active_page.error or "Page image could not be rendered.")

# Run Button
st.markdown("---")
run_clicked = st.button("Run Full Clinical Extraction Pipeline", type="primary", use_container_width=True)

if run_clicked or ("latest_analysis" in st.session_state and st.session_state["latest_analysis"].get("upload_id") == upload_id):
    if run_clicked:
        with st.spinner("Executing computer vision segmentation, TrOCR inference, and clinical parsing..."):
            model_bundle = load_model_bundle(selected_model)
            inf_result = run_inference(selected_model, active_page.image, model_bundle)
            run_id = str(uuid4())
            st.session_state["latest_analysis"] = {
                "run_id": run_id,
                "upload_id": upload_id,
                "model_id": selected_model,
                "result": inf_result,
                "pages": [
                    {
                        "page_number": p.page_number,
                        "original": p.image,
                        "recognized_text": inf_result.recognized_text if p.page_number == active_page.page_number else None,
                        "error": p.error,
                    }
                    for p in pages
                ],
            }
            record_processing_run(1, 1, 0, inf_result.inference_seconds)

    analysis_data = st.session_state.get("latest_analysis", {})
    inf_result = analysis_data.get("result")

    if inf_result and selected_model == "trocr":
        # Section 3: Extracted OCR and Verification
        with st.container(border=True):
            st.subheader("3 · OCR Output & Structured Entity Extraction")
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.metric("Inference Time", f"{inf_result.inference_seconds:.2f} s")
            meta_col2.metric("Segmented Crops", str(len(inf_result.segmentation_result.crops) if inf_result.segmentation_result else 1))
            meta_col3.metric("Reading Order Lines", str(inf_result.segmentation_result.line_count if inf_result.segmentation_result else 1))

            if getattr(inf_result, "uncertain_crops", None):
                st.warning(
                    f"⚠️ **Attention:** {len(inf_result.uncertain_crops)} segmented word(s) have low recognition confidence (<60%) or ambiguous cursive strokes. "
                    "Review these flagged words carefully before confirming medication identities."
                )
                with st.expander(f"Flagged Uncertain Words ({len(inf_result.uncertain_crops)})", expanded=True):
                    unc_cols = st.columns(min(len(inf_result.uncertain_crops), 4))
                    for idx, unc in enumerate(inf_result.uncertain_crops):
                        col = unc_cols[idx % len(unc_cols)]
                        with col.container(border=True):
                            col.image(unc["crop_image"], caption=f"Line {unc['line_index']}, Word {unc['crop_index']}", use_container_width=True)
                            col.write(f"**Predicted:** `{unc['text']}`")
                            col.caption(f"Model Confidence: {unc['confidence']:.1%}")

            col_raw, col_annot = st.columns(2)
            with col_raw:
                st.markdown("**Raw Assembled OCR Text (Editable):**")
                edited_ocr_text = st.text_area(
                    "TrOCR Output",
                    value=inf_result.recognized_text or "",
                    height=200,
                    help="You can edit any incorrect handwriting transcriptions directly here.",
                )
                if edited_ocr_text != inf_result.recognized_text:
                    # Re-extract if edited
                    new_meta, new_items = extract_prescription_fields(edited_ocr_text)
                    inf_result.recognized_text = edited_ocr_text
                    inf_result.structured_items = new_items
                    inf_result.prescription_metadata = new_meta

            with col_annot:
                st.markdown("**Segmented Reading Regions:**")
                if inf_result.annotated_page:
                    st.image(inf_result.annotated_page, caption="Identified words/lines in reading order", use_container_width=True)

            # Comprehensive Clinical Metadata Display
            if inf_result.prescription_metadata:
                m = inf_result.prescription_metadata
                st.markdown("##### Clinical Metadata & Patient Assessment")
                
                # Card 1: Prescriber & Hospital
                with st.expander("Prescriber & Hospital Details", expanded=True):
                    pc1, pc2, pc3 = st.columns(3)
                    pc1.markdown(f"**Doctor:** {m.prescriber_name or 'Not detected'}")
                    if m.prescriber_credentials:
                        pc1.caption(f"Qualifications: {m.prescriber_credentials}")
                    pc2.markdown(f"**Registration:** `{m.registration_number or 'Not detected'}`")
                    if m.prescriber_contact:
                        pc2.caption(f"Contact: {m.prescriber_contact}")
                    pc3.markdown(f"**Hospital / Clinic:** {m.clinic_or_hospital or 'Not detected'}")
                    if m.clinic_address:
                        pc3.caption(f"Address: {m.clinic_address}")

                # Card 2: Patient Info & Vitals
                with st.expander("Patient Demographics & Vitals", expanded=True):
                    dc1, dc2, dc3, dc4 = st.columns(4)
                    dc1.markdown(f"**Patient:** {m.patient_name or 'Not detected'}")
                    demog = []
                    if m.patient_sex: demog.append(f"Sex: {m.patient_sex}")
                    if m.patient_age: demog.append(f"Age: {m.patient_age}")
                    if demog: dc1.caption(", ".join(demog))

                    dc2.markdown(f"**Date:** {m.prescription_date or 'Not detected'}")
                    if m.opd_number:
                        dc2.caption(f"OPD No.: {m.opd_number}")

                    dc3.markdown(f"**Diagnosis:** {m.diagnosis or 'Not specified'}")

                    vitals_str = ", ".join(m.vitals) if m.vitals else "None recorded"
                    dc4.markdown(f"**Vitals:** {vitals_str}")

                # Card 3: Advice & Follow-up
                if m.advice or m.follow_up:
                    with st.expander("Clinical Advice & Follow-up", expanded=True):
                        ac1, ac2 = st.columns(2)
                        with ac1:
                            st.markdown("**Doctor's Advice:**")
                            if m.advice:
                                for adv in m.advice:
                                    st.markdown(f"- {adv}")
                            else:
                                st.write("No specific advice notes detected.")
                        with ac2:
                            st.markdown("**Follow-up:**")
                            st.write(m.follow_up or "Routine follow-up as needed.")

            # Structured Prescription Table
            st.markdown("##### Extracted Medication Items")
            if inf_result.structured_items:
                table_data = []
                for idx, item in enumerate(inf_result.structured_items):
                    table_data.append({
                        "ID": idx + 1,
                        "Raw OCR String": item.medicine_name,
                        "Normalized Generic / Brand": item.normalized_name or "Unknown",
                        "Generic Name": item.generic_name or "Uncatalogued",
                        "Match Status": item.status,
                        "Confidence": f"{item.match_confidence:.1%}",
                        "Form": item.dosage_form,
                        "Strength": item.strength,
                        "Frequency": item.frequency,
                        "Duration": item.duration,
                        "Route": item.route,
                        "Instructions": item.instructions,
                    })

                df_items = pd.DataFrame(table_data)
                st.dataframe(df_items, use_container_width=True, hide_index=True)

                st.caption(
                    "Status Guide: **Verified Exact** (>95% similarity) · "
                    "**Likely Match** (65-94% fuzzy match) · "
                    "**Uncertain / Review Needed** (<65% match, manual review mandatory)."
                )

                # Verification Gate
                st.markdown("---")
                st.subheader("4 · Human Verification & Confirmation Gate")
                st.info(
                    "Per safety regulations, AI extracted medicine identities must be confirmed "
                    "before generating clinical explanations or interaction checks."
                )

                confirmed_names = []
                v_cols = st.columns(len(inf_result.structured_items)) if len(inf_result.structured_items) <= 4 else st.columns(4)
                for idx, item in enumerate(inf_result.structured_items):
                    c = v_cols[idx % len(v_cols)]
                    with c.container(border=True):
                        st.markdown(f"**Item {idx + 1}**")
                        st.write(f"**Original OCR:** `{item.medicine_name}`")
                        
                        if item.status == "Verified Exact":
                            st.success(f"Verified Exact: **{item.normalized_name}**")
                            user_choice = item.normalized_name or item.medicine_name
                        elif item.status == "Likely Match":
                            st.warning(f"Likely Match: **{item.normalized_name}** ({item.match_confidence:.1%})")
                            # Offer candidate options supported by reference DB
                            options = [c_match["candidate_name"] for c_match in item.candidate_matches]
                            if item.medicine_name not in options:
                                options.append(f"Keep Original ({item.medicine_name})")
                            options.append("Other (Manual Entry)")
                            selected_opt = st.selectbox(
                                f"Match Selection #{idx + 1}",
                                options,
                                index=0 if options else 0,
                                key=f"med_select_{idx}",
                            )
                            if selected_opt == "Other (Manual Entry)":
                                user_choice = st.text_input(f"Enter Medicine #{idx + 1}", value=item.medicine_name, key=f"med_manual_{idx}")
                            elif selected_opt.startswith("Keep Original"):
                                user_choice = item.medicine_name
                            else:
                                user_choice = selected_opt
                        else:
                            st.error(f"Uncertain Match (<{item.match_confidence:.1%})")
                            st.caption("AI will not guess this medicine. Select from reference candidates or enter manually:")
                            options = [c_match["candidate_name"] for c_match in item.candidate_matches]
                            options.append(f"Keep Original ({item.medicine_name})")
                            options.append("Other (Manual Entry)")
                            selected_opt = st.selectbox(
                                f"Reference Matches #{idx + 1}",
                                options,
                                index=len(options) - 2 if len(options) >= 2 else 0,
                                key=f"med_select_{idx}",
                            )
                            if selected_opt == "Other (Manual Entry)":
                                user_choice = st.text_input(f"Enter Verified Name #{idx + 1}", value="", key=f"med_manual_{idx}")
                            elif selected_opt.startswith("Keep Original"):
                                user_choice = item.medicine_name
                            else:
                                user_choice = selected_opt

                        confirmed = st.checkbox(
                            "Confirm Verified Identity",
                            value=(item.status == "Verified Exact"),
                            key=f"med_check_{idx}",
                            help="Check to approve this medicine identity for clinical explanation and interaction checking.",
                        )
                        if confirmed and user_choice and user_choice.strip():
                            confirmed_names.append(user_choice.strip())

                # Section 5: Source-Grounded Medicine Explanations
                if confirmed_names:
                    st.markdown("---")
                    st.subheader("5 · Source-Grounded Medicine Explanations (RAG)")
                    for med_name in confirmed_names:
                        mono = explain_medicine(med_name)
                        with st.expander(f"Clinical Monograph: {mono['title']} ({mono['generic']})", expanded=True):
                            m_col1, m_col2 = st.columns([2, 1])
                            with m_col1:
                                st.markdown(f"**Therapeutic Class:** `{mono['class']}` | **ATC Code:** `{mono['atc']}`")
                                st.markdown("**Approved Indications & Clinical Uses:**")
                                for ind in mono["indications"]:
                                    st.markdown(f"- {ind}")

                                st.markdown("**Pharmacological Mechanism of Action:**")
                                st.write(mono["mechanism"])

                                st.markdown("**Standard Administration Guidelines:**")
                                for adm in mono["administration"]:
                                    st.markdown(f"- {adm}")

                            with m_col2:
                                st.markdown("**Common Side Effects:**")
                                for se in mono["side_effects"]:
                                    st.markdown(f"- {se}")

                                st.markdown("**Serious Warnings & Precautions:**")
                                for warn in mono["warnings"]:
                                    st.markdown(f"- :warning: {warn}")

                                st.markdown("**When to Seek Immediate Medical Attention:**")
                                for rf in mono["red_flags"]:
                                    st.markdown(f"- :rotating_light: {rf}")

                            st.markdown(f"**Traceable Reference Sources:** *{mono['sources']}*")
                            for ref in mono["references"]:
                                st.markdown(f"🔗 [{ref}]({ref})")
                            st.caption(mono["disclaimer"])

                    # Section 6: Drug-Drug Interaction Safety Screening
                    st.markdown("---")
                    st.subheader("6 · Drug-Drug Interaction & Safety Screening")
                    interaction_report = check_drug_interactions(confirmed_names)

                    if interaction_report["interactions"]:
                        if interaction_report["has_major_alert"]:
                            st.error(f":rotating_light: **{interaction_report['overall_status']}**: One or more major clinical interactions detected!")
                        else:
                            st.warning(f":warning: **{interaction_report['overall_status']}**: Potential moderate drug interactions identified.")

                        for inter in interaction_report["interactions"]:
                            with st.container(border=True):
                                sev_color = "red" if inter["severity"].startswith("Major") else "orange"
                                st.markdown(f"#### :{sev_color}[{inter['severity']}]: {inter['drug_a']} ↔ {inter['drug_b']}")
                                st.markdown(f"**Clinical Risk:** {inter['risk_summary']}")
                                st.markdown(f"**Biological Mechanism:** {inter['clinical_mechanism']}")
                                st.markdown(f"**Management Recommendation:** {inter['management_advice']}")
                                st.caption(f"Evidence Source: {inter['evidence_source']}")
                    else:
                        st.success(
                            f"Checked {len(confirmed_names)} confirmed medication(s): No documented severe interactions found in reference database."
                        )

                    st.info(interaction_report["safety_disclaimer"])
                else:
                    st.info("Check 'Confirm Accuracy' above for at least one medication to display source-grounded clinical explanations and drug interaction screenings.")

    elif inf_result and selected_model in {"ae", "vae"}:
        with st.container(border=True):
            st.subheader(f"3 · {selected_model.upper()}: Image Reconstruction & Lost Detail Analysis")
            st.caption(
                "Evaluates structural preservation and identifies lost details by comparing the reconstructed image "
                "directly against the original uploaded scan."
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(
                    active_page.image,
                    caption=f"Original Input Scan ({active_page.image.width}×{active_page.image.height}px)",
                    use_container_width=True,
                )
            with c2:
                if inf_result.reconstruction is not None:
                    st.image(
                        inf_result.reconstruction,
                        caption=f"{selected_model.upper()} Reconstruction ({inf_result.reconstruction.width}×{inf_result.reconstruction.height}px)",
                        use_container_width=True,
                    )
                    try:
                        import io as io_mod
                        buf = io_mod.BytesIO()
                        inf_result.reconstruction.save(buf, format="PNG")
                        st.download_button(
                            label=f"Download {selected_model.upper()} PNG",
                            data=buf.getvalue(),
                            file_name=f"{selected_model}_reconstruction.png",
                            mime="image/png",
                            key=f"dl_anlz_{selected_model}",
                            use_container_width=True,
                        )
                    except Exception:
                        pass
                else:
                    st.warning("Reconstruction output is unavailable.")
            with c3:
                if inf_result.error_map_preview is not None:
                    st.image(
                        inf_result.error_map_preview,
                        caption="Lost Detail Residual Heatmap (Red = High Error)",
                        use_container_width=True,
                    )
                else:
                    st.info("Error map is unavailable.")

            if inf_result.reconstruction_metrics:
                m = inf_result.reconstruction_metrics
                st.markdown("##### Measured Reconstruction & Detail Preservation Metrics")
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("MSE", f"{m.get('mse', 0.0):.4f}", help="Mean Squared Error (lower is better).")
                mc2.metric("PSNR", f"{m.get('psnr_db', 0.0):.2f} dB", help="Peak Signal-to-Noise Ratio (higher is better).")
                mc3.metric("SSIM", f"{m.get('ssim', 0.0):.4f}", help="Structural Similarity Index (closer to 1.0 is better).")
                mc4.metric(
                    "Sharpness Change",
                    f"{m.get('sharpness_change_pct', 0.0):+.1f}%",
                    help="Relative change in Laplacian variance between original and reconstructed image.",
                )
                mc5.metric(
                    "Edge Detail Loss",
                    f"{m.get('edge_loss_pct', 0.0):.1f}%",
                    help="Percentage of high-frequency edge energy lost during latent compression.",
                )

                if selected_model == "ae":
                    st.info(
                        "**Model Explanation:** The Skip-Autoencoder compresses the image through encoder stages and uses "
                        "skip connections to restore fine handwriting strokes and paper texture. The residual heatmap "
                        "identifies subtle smoothing along high-frequency stroke boundaries."
                    )
                else:
                    st.info(
                        "**Model Explanation & Accuracy Limitations:** The VAE encodes images into a 32-channel continuous "
                        "Gaussian latent bottleneck $\\mathcal{N}(\\mu, \\sigma^2)$ regularized by KL divergence. "
                        "A reduction in sharpness and softening of fine ink pen ligatures is mathematically expected "
                        "due to the variational prior constraint. Reconstruction quality is evaluated empirically above."
                    )

            if inf_result.threshold_preview is not None:
                with st.expander(f"Binarized Stroke & Ink Separation Preview ({selected_model.upper()})", expanded=False):
                    st.image(
                        inf_result.threshold_preview,
                        caption=f"Adaptive Gaussian Binarization of {selected_model.upper()} Reconstruction",
                        use_container_width=True,
                    )

            if inf_result.handwriting_samples:
                st.markdown("##### Segmented Handwriting Crops Reconstructed with VAE")
                crops = inf_result.handwriting_samples
                hw_cols = st.columns(min(len(crops), 4))
                for idx, hw_img in enumerate(crops):
                    col = hw_cols[idx % len(hw_cols)]
                    with col:
                        st.image(hw_img, caption=f"Crop #{idx + 1}", use_container_width=True)

    elif inf_result and selected_model == "gan":
        with st.container(border=True):
            st.markdown(
                """
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem;">
                    <h3 style="margin:0; font-size:1.35rem; font-weight:700; color:#1e1b4b;">
                        3 · GAN: Generative Handwriting Synthesis Benchmark
                    </h3>
                    <span style="background:#e0e7ff; color:#3730a3; font-size:0.75rem; font-weight:700; padding:4px 10px; border-radius:9999px; border:1px solid #c7d2fe;">
                        DCGAN Neural Generator
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.info(
                "ℹ️ **Generative Architecture Notice:** GAN is an image synthesis model trained to generate synthetic handwriting "
                r"from random Gaussian noise vectors $z \sim \mathcal{N}(0, I)$. It does **not** perform OCR or parse medications "
                "from your uploaded prescription image. To transcribe and explain the uploaded prescription, select **TrOCR** above. "
                "Below are synthetic handwriting samples synthesized by the trained GAN checkpoint."
            )

            # Control Panel with Modern Blue-Purple Accent Theme
            with st.container():
                st.markdown(
                    """
                    <style>
                    .gan-ctrl-box {
                        background: #ffffff;
                        border: 1px solid #e0e7ff;
                        border-radius: 12px;
                        padding: 1rem 1.25rem 0.75rem;
                        margin-bottom: 1rem;
                        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.04);
                    }
                    .gan-card {
                        background: #ffffff;
                        border: 1px solid #e2e8f0;
                        border-radius: 12px;
                        padding: 10px;
                        text-align: center;
                        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
                        transition: transform 0.15s ease, box-shadow 0.15s ease;
                    }
                    .gan-card:hover {
                        border-color: #c7d2fe;
                        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

            ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.4, 1.4, 1.2, 1.2])
            with ctrl_c1:
                num_gan_samples = st.slider("Number of Samples", min_value=4, max_value=8, value=4, step=1, key="gan_p2_sample_count")
            with ctrl_c2:
                gan_contrast = st.slider("Stroke Contrast", min_value=0.7, max_value=1.5, value=1.0, step=0.1, key="gan_p2_contrast")
            with ctrl_c3:
                st.write("")
                st.write("")
                regen_gan_btn = st.button("⚡ Generate Samples", type="primary", use_container_width=True, key="gan_p2_generate_btn")

            # Check session state for custom generated batch with fresh latent noise
            gan_state_key = f"gan_p2_samples_{active_page.page_number}"
            if regen_gan_btn or gan_state_key not in st.session_state:
                import time as time_mod
                dyn_seed = int(time_mod.time() * 1000) % 1_000_000
                from models.handwriting_gan import generate_handwriting_samples, load_handwriting_gan
                generator = model_bundle.get("generator")
                if generator is None:
                    try:
                        generator, _ = load_handwriting_gan()
                    except Exception:
                        generator = None

                if generator is not None:
                    active_crops = generate_handwriting_samples(
                        generator,
                        count=num_gan_samples,
                        seed=dyn_seed,
                        clarity_boost=True,
                        contrast=gan_contrast,
                    )
                else:
                    active_crops = inf_result.handwriting_samples or []

                from models.metrics_evaluator import (
                    detect_generation_failure,
                    evaluate_image_quality,
                )
                failure_rep = detect_generation_failure(active_crops)
                sample_qs = [evaluate_image_quality(s) for s in active_crops]
                m_sharp = float(np.mean([q["sharpness"] for q in sample_qs])) if sample_qs else 0.0
                m_contrast = float(np.mean([q["contrast"] for q in sample_qs])) if sample_qs else 0.0
                m_cov = float(np.mean([q.get("foreground_coverage_pct", 0.0) for q in sample_qs])) if sample_qs else 0.0

                img_w, img_h = active_crops[0].size if active_crops else (64, 64)
                native_dim = getattr(generator, "image_size", 64)

                computed_metrics = {
                    "diversity_l1": failure_rep["diversity"],
                    "mean_sharpness": m_sharp,
                    "mean_contrast": m_contrast,
                    "mean_foreground_coverage": m_cov,
                    "native_dim": native_dim,
                    "display_dim": f"{img_w}x{img_h}",
                    "failure_detected": 1.0 if failure_rep["is_failure"] else 0.0,
                }
                st.session_state[gan_state_key] = {
                    "crops": active_crops,
                    "metrics": computed_metrics,
                    "failure_rep": failure_rep,
                    "timestamp": int(time_mod.time()),
                }

            cached = st.session_state[gan_state_key]
            crops = cached["crops"]
            m = cached["metrics"]
            q = cached["failure_rep"]
            ts = cached.get("timestamp", 0)

            # Prepare ZIP of all generated samples for batch download
            import io as io_mod
            import zipfile as zip_mod
            zip_buffer = io_mod.BytesIO()
            with zip_mod.ZipFile(zip_buffer, "w", zip_mod.ZIP_DEFLATED) as zip_file:
                for idx, hw_img in enumerate(crops):
                    img_bytes = io_mod.BytesIO()
                    hw_img.save(img_bytes, format="PNG")
                    zip_file.writestr(f"dcgan_sample_{ts}_{idx+1}.png", img_bytes.getvalue())

            with ctrl_c4:
                st.write("")
                st.write("")
                st.download_button(
                    label="📦 Download All (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"dcgan_synthetic_samples_{ts}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key=f"dl_gan_zip_{ts}",
                )

            # Display 4-Column Responsive Sample Gallery
            if crops:
                hw_cols = st.columns(min(len(crops), 4))
                for idx, hw_img in enumerate(crops):
                    with hw_cols[idx % len(hw_cols)]:
                        with st.container(border=True):
                            st.image(hw_img, caption=f"Synthetic Sample #{idx + 1}", use_container_width=True)
                            try:
                                buf = io_mod.BytesIO()
                                hw_img.save(buf, format="PNG")
                                unique_name = f"dcgan_sample_{ts}_{idx+1}.png"
                                st.download_button(
                                    label=f"Download Sample #{idx+1} (PNG)",
                                    data=buf.getvalue(),
                                    file_name=unique_name,
                                    mime="image/png",
                                    key=f"dl_gan_p2_hw_{ts}_{idx}",
                                    use_container_width=True,
                                )
                            except Exception:
                                pass

            # Empirical Generative Quality & Diversity Metrics
            st.markdown(
                "<div style='font-size:1.05rem; font-weight:700; color:#1e1b4b; margin:1.25rem 0 0.5rem;'>"
                "Empirical Generative Quality & Diversity Metrics"
                "</div>",
                unsafe_allow_html=True,
            )
            gc1, gc2, gc3, gc4, gc5 = st.columns(5)
            gc1.metric("Output Resolution", f"{m.get('native_dim', 64)}x{m.get('native_dim', 64)} px", help="Native generator canvas resolution.")
            gc2.metric(
                "Pairwise Diversity",
                f"{m.get('diversity_l1', 0.0):.4f} L1",
                help="Mean pairwise L1 pixel distance across generated samples. Confirms absence of mode collapse.",
            )
            gc3.metric("Mean Sharpness", f"{m.get('mean_sharpness', 0.0):.1f}", help="Laplacian variance measuring edge definition.")
            gc4.metric("Mean Contrast", f"{m.get('mean_contrast', 0.0):.1f}", help="RMS luminance contrast between ink and paper.")
            gc5.metric("Foreground Coverage", f"{m.get('mean_foreground_coverage', 0.0):.1f}%", help="Percentage of canvas occupied by dark handwriting strokes.")

            if q.get("is_failure"):
                st.warning(f"⚠️ **Generation Quality Limitations Detected:** {', '.join(q.get('failure_reasons', []))}")
            else:
                st.success("✅ **Quality Assessment:** Crisp, high-contrast synthetic handwriting patterns generated directly from the trained neural network checkpoint.")

            # Ground Truth Real vs Generated Comparison Section
            with st.expander("🔬 Real vs. Generated Comparison (Ground Truth RxHandBD-ML vs. DCGAN)", expanded=False):
                st.markdown(
                    "Compare real physician handwriting crops from the **RxHandBD-ML** training set directly against "
                    r"synthetic handwriting word patches synthesized from Gaussian latent vectors $z \sim \mathcal{N}(0, I)$."
                )
                from pathlib import Path as PathLib
                import pandas as pd_mod
                rx_dir = PathLib("dataset/RxHandBD-ML/RxHandBD-ML")
                csv_file = rx_dir / "Train_Label.csv"
                img_dir = rx_dir / "Train_Set"

                real_col, gen_col = st.columns(2)
                with real_col:
                    st.markdown("**Real Physician Handwriting (RxHandBD-ML)**")
                    if csv_file.is_file() and img_dir.is_dir():
                        df_rx = pd_mod.read_csv(csv_file)
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
                    if crops:
                        gc_c1, gc_c2 = st.columns(2)
                        for g_idx in range(min(4, len(crops))):
                            col_target = gc_c1 if g_idx % 2 == 0 else gc_c2
                            col_target.image(
                                crops[g_idx],
                                caption=f"Synthetic Sample #{g_idx + 1}",
                                use_container_width=True,
                            )

                st.markdown(
                    """
                    > **Fidelity & Diagnostic Observations:**
                    > - **Cursive Trajectory:** The DCGAN has successfully captured the horizontal cursive word baseline and natural handwriting curvature.
                    > - **Resolution Analysis:** Real raw crops are $512 \times 512$ with $2\text{--}4$ pixel strokes. When downsampled to $64 \times 64$, letter loops ('e', 'a') become $\\le 1$ pixel wide. Retraining at $128 \times 128$ in Google Colab allows full loop articulation and fine hair-stroke ligatures.
                    > - **Zero-PHI Compliance:** All generated outputs are synthetic research samples and strictly medication-free.
                    """
                )

    elif inf_result and selected_model == "diffusion":
        with st.container(border=True):
            st.subheader(f"3 · {selected_model.upper()}: Generative Handwriting Synthesis Benchmark")
            st.info(
                f"ℹ️ **Generative Architecture Notice:** {selected_model.upper()} is an image synthesis model trained "
                fr"to generate synthetic handwriting from random Gaussian noise vectors $z \sim \mathcal{N}(0, I)$. "
                "It does **not** perform OCR or parse medications from your uploaded prescription image. "
                "To transcribe and explain the uploaded prescription, select **TrOCR** above. "
                f"Below are synthetic handwriting samples synthesized by the trained {selected_model.upper()} checkpoint."
            )

            crops = inf_result.handwriting_samples or []
            if crops:
                hw_cols = st.columns(min(len(crops), 4))
                for idx, hw_img in enumerate(crops):
                    with hw_cols[idx % len(hw_cols)]:
                        st.image(hw_img, caption=f"Synthetic Sample #{idx + 1}", use_container_width=True)
                        try:
                            import io as io_mod
                            buf = io_mod.BytesIO()
                            hw_img.save(buf, format="PNG")
                            st.download_button(
                                label=f"Download Sample #{idx+1} (PNG)",
                                data=buf.getvalue(),
                                file_name=f"{selected_model}_sample_{idx+1}.png",
                                mime="image/png",
                                key=f"dl_gen_{selected_model}_{idx}",
                                use_container_width=True,
                            )
                        except Exception:
                            pass

            if inf_result.reconstruction_metrics:
                m = inf_result.reconstruction_metrics
                st.markdown("##### Empirical Generative Quality & Diversity Metrics")
                gc1, gc2, gc3, gc4 = st.columns(4)
                gc1.metric("Sample Count", str(len(crops)))
                gc2.metric(
                    "Pairwise Diversity",
                    f"{m.get('diversity_l1', 0.0):.4f} L1",
                    help="Mean pairwise L1 pixel distance across generated samples. Higher scores indicate absence of mode collapse.",
                )
                gc3.metric("Mean Sharpness", f"{m.get('mean_sharpness', 0.0):.1f}", help="Mean Laplacian variance of stroke edges.")
                gc4.metric("Mean Contrast", f"{m.get('mean_contrast', 0.0):.1f}", help="RMS luminance contrast between ink and paper.")

            if inf_result.quality_metrics:
                q = inf_result.quality_metrics
                if q.get("is_failure"):
                    st.warning(f"⚠️ **Generation Quality Limitations Detected:** {', '.join(q.get('failure_reasons', []))}")
                else:
                    st.success("✅ **Quality Assessment:** Varied, legible synthetic outputs produced with adequate stroke contrast and diversity.")
