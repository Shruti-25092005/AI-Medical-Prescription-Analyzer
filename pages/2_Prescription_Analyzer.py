from hashlib import sha256
from uuid import uuid4

import streamlit as st

from models.gui_services import (
    MODEL_NAMES,
    load_model_bundle,
    model_statuses,
    record_processing_run,
    run_inference,
)
from models.prescription_upload import (
    UploadValidationError,
    load_prescription_pages,
)
from models.privacy_controls import clear_analysis_results
from models.synthetic_record_gan import (
    SYNTHETIC_OUTPUT_LABEL,
    generate_records,
    records_as_dicts,
)


st.markdown(
    """
    <style>
    section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
        max-width: 1440px !important;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin-bottom:1.2rem;">
      <div style="font-size:.76rem; text-transform:uppercase; letter-spacing:.12em;
          font-weight:700; color:#167d78;">Document workflow</div>
      <h1 style="margin:.45rem 0 .25rem;">Prescription analyzer</h1>
      <p style="max-width:760px; margin:0;">
        Upload an image or PDF to transcribe handwritten text or preview an
        image reconstruction. All extracted text needs human review.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "This tool displays model output only. It does not identify or explain "
    "medicines, validate doses, diagnose conditions, or recommend treatment. "
    "Do not use it to make care decisions."
)

with st.container(border=True):
    st.subheader("1 · Upload and choose a model")
    uploaded_file = st.file_uploader(
        "Prescription image or PDF",
        type=["jpg", "jpeg", "png", "pdf"],
        key=(
            "prescription_upload_"
            f"{st.session_state.get('prescription_upload_generation', 0)}"
        ),
        help=(
            "Supported formats: JPG, JPEG, PNG, and PDF. Maximum file size: "
            "10 MB; PDFs are limited to 20 pages."
        ),
    )

content = uploaded_file.getvalue() if uploaded_file is not None else None
upload_id = (
    sha256(content).hexdigest()
    if content is not None
    else None
)
if st.session_state.get("active_upload_id") != upload_id:
    clear_analysis_results(st.session_state)
    st.session_state["active_upload_id"] = upload_id

available_models = {
    model["id"]: model
    for model in model_statuses()
    if model["available"]
}
model_labels = {
    model_id: MODEL_NAMES[model_id]
    for model_id in ("trocr", "ae", "vae", "gan")
    if model_id in available_models
}
if not model_labels:
    st.error("No supported model checkpoints are currently available.")
    st.stop()

with st.container(border=True):
    st.subheader("Choose task and run")
    selected_model = st.selectbox(
        "Model and task",
        options=list(model_labels),
        format_func=lambda model_id: model_labels[model_id],
        help=(
            "TrOCR recognizes text. AE and VAE reconstruct images. GAN "
            "generates fictional records and does not analyze uploads."
        ),
    )

    if selected_model == "gan":
        st.warning(
            f"{SYNTHETIC_OUTPUT_LABEL}. GAN generation does not use or analyze "
            "the uploaded file."
        )
        record_count = st.number_input(
            "Number of fictional records",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
        )
        generate_clicked = st.button(
            "Generate fictional records",
            type="primary",
            icon=":material/auto_awesome:",
            use_container_width=True,
        )
    else:
        generate_clicked = False
    if selected_model in {"ae", "vae"}:
        st.info(
            "This selection produces an image reconstruction, not recognized "
            "text. Choose TrOCR for handwriting recognition."
        )
    if selected_model == "ae":
        st.caption(
            "AE keeps the uploaded resolution up to 1024 px on the longest "
            "side; larger images are scaled proportionally."
        )

    run_analysis = (
        st.button(
            "Run analysis",
            type="primary",
            icon=":material/auto_awesome:",
            use_container_width=True,
        )
        if selected_model != "gan"
        else False
    )

if selected_model == "gan":
    if generate_clicked:
        try:
            with st.spinner("Generating fictional records..."):
                st.session_state["analyzer_synthetic_records"] = generate_records(
                    int(record_count)
                )
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            st.error(
                "Synthetic samples could not be generated. Check that the "
                "local GAN checkpoint is available."
            )
    generated_records = st.session_state.get("analyzer_synthetic_records", [])
    if generated_records:
        st.subheader("AI-generated fictional records")
        record_columns = st.columns(2)
        for index, (_, image) in enumerate(generated_records):
            record_columns[index % len(record_columns)].image(
                image,
                caption=(
                    f"{SYNTHETIC_OUTPUT_LABEL} — "
                    f"Sample {index + 1}"
                ),
                width="stretch",
            )
        st.dataframe(
            records_as_dicts([record for record, _ in generated_records]),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Generated samples contain only synthetic sample IDs and "
            "placeholder text. They contain no intended identity or clinical "
            "content and must never be used for care."
        )
    st.stop()

if uploaded_file is None or content is None or upload_id is None:
    st.info(
        "Upload an image or PDF to analyze it, or choose the GAN option to "
        "generate fictional records."
    )
    st.stop()

if run_analysis:
    try:
        pages = load_prescription_pages(uploaded_file.name, content)
    except UploadValidationError:
        st.error(
            "The upload could not be processed. Check the file format, "
            "10 MB size limit, PDF page limit, and image dimensions."
        )
        st.stop()

    model_bundle = None
    if any(page.image is not None for page in pages):
        try:
            with st.spinner(f"Loading {MODEL_NAMES[selected_model]} checkpoint..."):
                model_bundle = load_model_bundle(selected_model)
        except (
            FileNotFoundError,
            ImportError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            st.error(
                "The selected local model could not be loaded. Check that its "
                "checkpoint is available and compatible."
            )
            st.stop()
        loaded_models = set(st.session_state.get("loaded_model_ids", []))
        loaded_models.add(selected_model)
        st.session_state["loaded_model_ids"] = sorted(loaded_models)

    analysis_run_id = uuid4().hex
    analysis_pages = []
    successful_pages = 0
    failed_pages = 0
    inference_seconds = 0.0
    progress = st.progress(0)
    status = st.status("Preparing page analysis...", expanded=True)

    for index, page in enumerate(pages):
        status.write(f"Processing page {page.page_number} of {len(pages)}")
        record = {
            "page_number": page.page_number,
            "original": page.image,
            "recognized_text": None,
            "reconstruction": None,
            "threshold_preview": None,
            "inference_seconds": 0.0,
            "error": page.error,
        }

        if page.image is None or model_bundle is None:
            failed_pages += 1
        else:
            try:
                result = run_inference(selected_model, page.image, model_bundle)
                record["recognized_text"] = result.recognized_text
                record["reconstruction"] = result.reconstruction
                record["threshold_preview"] = result.threshold_preview
                record["inference_seconds"] = result.inference_seconds
                inference_seconds += result.inference_seconds
                successful_pages += 1
            except (
                ImportError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                record["error"] = "Model inference failed for this page."
                failed_pages += 1

        analysis_pages.append(record)
        progress.progress((index + 1) / len(pages))

    status.update(
        label=(
            f"Analysis complete: {successful_pages} successful, "
            f"{failed_pages} with errors."
        ),
        state="complete" if failed_pages == 0 else "error",
        expanded=False,
    )
    record_processing_run(
        len(pages),
        successful_pages,
        failed_pages,
        inference_seconds,
    )
    st.session_state["latest_analysis"] = {
        "run_id": analysis_run_id,
        "upload_id": upload_id,
        "model_id": selected_model,
        "pages": analysis_pages,
    }

analysis = st.session_state.get("latest_analysis")
if analysis and analysis["upload_id"] == upload_id:
    st.subheader("2 · Review extracted results")
    result_meta = st.columns(3)
    result_meta[0].markdown("**Input**  \nUploaded file")
    result_meta[1].markdown(
        f"**Model**  \n{MODEL_NAMES[analysis['model_id']]}"
    )
    result_meta[2].markdown(f"**Pages**  \n{len(analysis['pages'])}")
    for record in analysis["pages"]:
        with st.container(border=True):
            st.markdown(f"#### Page {record['page_number']}")
            if record["error"]:
                st.error(record["error"])
            if record["original"] is None:
                continue

            if (
                analysis["model_id"] == "ae"
                and record["reconstruction"] is not None
            ):
                image_width = max(
                    record["original"].width,
                    record["reconstruction"].width,
                )
                comparison_key = f"ae_comparison_page_{record['page_number']}"
                st.markdown(
                    f"""
                    <style>
                    .st-key-{comparison_key} [data-testid="stHorizontalBlock"] {{
                        flex-wrap: nowrap !important;
                        overflow-x: auto !important;
                    }}
                    .st-key-{comparison_key} [data-testid="stColumn"],
                    .st-key-{comparison_key} [data-testid="column"] {{
                        flex: 0 0 {image_width}px !important;
                        width: {image_width}px !important;
                        min-width: {image_width}px !important;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                with st.container(key=comparison_key):
                    original_column, output_column = st.columns(2, gap="small")
                    original_column.image(
                        record["original"],
                        caption=f"Original page · {record['original'].width} × "
                        f"{record['original'].height}px",
                        width=record["original"].width,
                    )
                    output_column.image(
                        record["reconstruction"],
                        caption=(
                            "AE reconstruction · light denoising, contrast and "
                            "edge enhancement "
                            f"· {record['reconstruction'].width} × "
                            f"{record['reconstruction'].height}px"
                        ),
                        width=record["reconstruction"].width,
                    )
                st.caption(
                    "Image enhancement is post-processing, not OCR. "
                    "Check every detail against the original; unclear "
                    "handwriting remains uncertain."
                )
                if record["threshold_preview"] is not None:
                    with st.expander(
                        "Optional adaptive-threshold view "
                        "(may remove faint strokes)"
                    ):
                        st.image(
                            record["threshold_preview"],
                            caption=(
                                "High-contrast reference derived from "
                                "the AE output; not recognized text."
                            ),
                            width="content",
                        )
            else:
                original_column, output_column = st.columns(2)
                original_column.image(
                    record["original"],
                    caption=f"Original page · {record['original'].width} × "
                    f"{record['original'].height}px",
                    width="content",
                )
            if analysis["model_id"] == "trocr":
                if record["recognized_text"] is not None:
                    output_column.text_area(
                        "Extracted text — verify and correct",
                        value=record["recognized_text"],
                        key=f"analysis_{analysis['run_id']}_page_{record['page_number']}",
                        height=220,
                    )
                    output_column.caption(
                        "OCR transcription only; medicine details are not "
                        "parsed or medically explained."
                    )
                elif not record["error"]:
                    output_column.info("No text was recognized on this page.")
            elif (
                analysis["model_id"] != "ae"
                and record["reconstruction"] is not None
            ):
                output_column.image(
                    record["reconstruction"],
                    caption="Image reconstruction (not OCR output)",
                    width="content",
                )
            if record["inference_seconds"]:
                st.caption(
                    f"Model inference: {record['inference_seconds']:.2f} seconds"
                )
