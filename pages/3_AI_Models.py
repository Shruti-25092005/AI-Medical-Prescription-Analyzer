from pathlib import Path
import streamlit as st
from models.gui_services import model_statuses, dataset_counts

st.markdown(
    """
    <style>
    section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
        max-width: 1440px !important;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin-bottom:1.1rem;">
      <div style="font-size:.76rem; text-transform:uppercase; letter-spacing:.12em;
          font-weight:700; color:#167d78;">Academic Documentation &amp; Architecture</div>
      <h1 style="margin:.45rem 0 .25rem;">About &amp; Research Information</h1>
      <p style="max-width:850px; margin:0; color:#587579;">
        Technical specifications, system architecture, dataset characteristics, ethical privacy controls,
        and research methodology for the AI Medical Prescription Explainer and Generative Handwriting Generator.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Section 1: Project Objectives
with st.container(border=True):
    st.subheader("1 · Project Objectives")
    st.markdown(
        """
        This project investigates two interconnected frontiers in clinical computer vision and Generative AI:
        1. **Module A — Clinical Prescription Understanding & Explanation:**
           Developing a robust document processing pipeline capable of handling scanned images and multipage PDFs,
           segmenting unconstrained handwritten medical text, performing optical character recognition with a fine-tuned
           Vision Transformer (TrOCR), extracting structured prescription entities, normalizing brand names against
           authoritative formularies, providing traceable clinical explanations (indications, mechanisms, side effects),
           and conducting multi-drug interaction safety screenings.
        2. **Module B — Synthetic Medical Handwriting Generation:**
           Implementing and empirically comparing multiple generative paradigms (Skip-AE, VAE, DCGAN, and DDPM)
           to generate synthetic handwritten medical text images. This provides a methodology for augmenting sparse clinical datasets
           without compromising real patient privacy or exposing protected health information (PHI).
        """
    )

# Section 2: End-to-End System Architecture
with st.container(border=True):
    st.subheader("2 · System Architecture")
    st.markdown(
        """
        ```mermaid
        flowchart TD
            subgraph Input ["Input Document"]
                A["Image (JPG/PNG) or Multi-page PDF"]
            end

            subgraph Preprocess ["Vision Preprocessing & Segmentation"]
                A --> B["Orientation Correction & Denoising"]
                B --> C["CLAHE Contrast Enhancement & Adaptive Thresholding"]
                C --> D["Line & Word Crop Segmentation (Contour / Morphological)"]
            end

            subgraph Recognition ["TrOCR Recognition & NLP Parsing"]
                D --> E["TrOCR Vision Transformer Inference"]
                E --> F["Assembled Prescription Transcription"]
                F --> G["Regex / Heuristic Entity Extraction (Dosage, Form, Frequency)"]
                G --> H["Medicine Normalization & Fuzzy Reference Match"]
            end

            subgraph Clinical ["Clinical Verification & Knowledge"]
                H --> I["Human Verification Gate"]
                I --> J["Source-Grounded Medicine Monograph Retrieval (RAG)"]
                I --> K["Pairwise Drug-Drug Interaction Checker"]
            end

            subgraph GenerativeStudio ["Module B: Generative AI Studio"]
                L["RxHandBD-ML Dataset (512x512)"] --> M["Skip-Autoencoder (AE)"]
                L --> N["Sharper VAE (EMA)"]
                L --> O["DCGAN Handwriting Generator"]
                L --> P["DDPM Compact U-Net Diffusion"]
            end
        ```
        """
    )

# Section 3: Checkpoint & Model Registry
with st.container(border=True):
    st.subheader("3 · Live Checkpoint Status & Model Registry")
    models = model_statuses()
    cols = st.columns(len(models))
    for col, m in zip(cols, models):
        with col.container(border=True):
            st.markdown(f"**{m['name'].split(' - ')[0]}**")
            st.caption(m["purpose"])
            if m["available"]:
                st.success("Checkpoint Active")
            else:
                st.warning("Not Configured")
            st.code(Path(m["checkpoint"]).name if Path(m["checkpoint"]).is_file() else m["checkpoint"], language="bash")

# Section 4: Ethical Considerations & Patient Privacy Safeguards
with st.container(border=True):
    st.subheader("4 · Ethical Safeguards & Privacy Policy")
    st.markdown(
        """
        1. **Ephemeral In-Memory Processing:**
           Uploaded patient prescription images and PDFs remain exclusively in memory during the active session.
           No uploaded images or transcribed patient identifiers are permanently stored or indexed.
        2. **Synthetic Data Policy:**
           Generative models (DCGAN and DDPM) generate synthetic handwriting from random Gaussian noise $z \\sim \\mathcal{N}(0, I)$
           and noise schedules, preventing the leakage of real patient records.
        3. **Mandatory Human-in-the-Loop Gate:**
           Medical prescriptions contain high-stakes directives. The system enforces a mandatory human review step
           before any clinical explanation or drug interaction check is computed.
        4. **Authoritative Source Grounding:**
           Medicine explanations are retrieved directly from traceable literature (WHO Model List of Essential Medicines,
           British National Formulary, OpenFDA). The model explicitly disclaims any authority to alter prescriptions.
        """
    )

# Section 5: Limitations & Future Research
with st.container(border=True):
    st.subheader("5 · Known Limitations & Future Work")
    st.markdown(
        """
        - **Handwriting Variability:** Unconstrained cursive handwriting by physicians varies drastically in slant,
          pen pressure, and ligature overlap, resulting in ~32.5% exact match accuracy at word level.
        - **Resolution Constraints:** Diffusion and GAN models operate at 64x64/128x128 resolution for real-time CPU evaluation.
          Scaling to multi-line diffusion models (e.g. Latent Diffusion Models at 512x512) represents an active area for GPU clusters.
        - **Formulary Coverage:** The local offline reference dictionary covers primary essential medications and the RxHandBD-ML
          lexicon. Expanding to exhaustive regional commercial databases remains ongoing work.
        """
    )
