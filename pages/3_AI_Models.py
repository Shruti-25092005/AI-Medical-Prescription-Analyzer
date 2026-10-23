import streamlit as st

from models.gui_services import model_statuses


st.markdown(
    """
    <div style="margin-bottom:1.1rem;">
      <div style="font-size:.76rem; text-transform:uppercase; letter-spacing:.12em;
          font-weight:700; color:#167d78;">Model center</div>
      <h1 style="margin:.45rem 0 .25rem;">AI models</h1>
      <p style="max-width:760px; margin:0;">
        Existing local checkpoints power the available recognition and
        reconstruction workflows. Models are not downloaded or retrained here.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

loaded_models = set(st.session_state.get("loaded_model_ids", []))
models = model_statuses()
for start in range(0, len(models), 3):
    columns = st.columns(3)
    for column, model in zip(columns, models[start : start + 3]):
        with column.container(border=True):
            st.subheader(model["name"])
            st.write(model["purpose"])
            st.caption(f"Checkpoint: `{model['checkpoint']}`")
            if model["available"]:
                st.success("Checkpoint file(s) found")
            else:
                st.error("Checkpoint missing or incomplete")
            if model["id"] in loaded_models:
                st.info("Loaded in this session")
            else:
                st.caption("Not loaded in this session")

st.subheader("Model capabilities")
st.markdown(
    """
    - **TrOCR:** the handwriting-recognition model. Its local transformer
      checkpoint is used to produce text from image pages.
    - **AE:** a skip autoencoder that encodes and reconstructs RGB images.
      Reconstruction is not text recognition.
    - **VAE:** a variational encoder-decoder that reconstructs grayscale images
      from the encoder mean. It does not perform text recognition.
    - **GAN:** generates clearly labeled fictional sample cards. Its output is
      synthetic, is not a prescription, and is not for medical use.
    """
)
st.info(
    "Only TrOCR currently has saved text-recognition evaluation metrics. "
    "AE/VAE OCR accuracy, CER, and WER are not measured and are not compared."
)
