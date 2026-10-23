from pathlib import Path

import streamlit as st

from models.gui_services import (
    evaluation_data,
    evaluation_metrics,
    model_evaluation_matrix,
)


st.markdown(
    """
    <div style="margin-bottom:1.1rem;">
      <div style="font-size:.76rem; text-transform:uppercase; letter-spacing:.12em;
          font-weight:700; color:#167d78;">Quality insights</div>
      <h1 style="margin:.45rem 0 .25rem;">Performance evaluation</h1>
      <p style="max-width:760px; margin:0;">
        Saved TrOCR test-set metrics, charts, and row-level evaluation results.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    data = evaluation_data()
    metrics = evaluation_metrics(data)
except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
    st.error(f"Could not load the existing evaluation results: {exc}")
    st.stop()

columns = st.columns(4)
columns[0].metric(
    "Case-insensitive exact match",
    f"{metrics['exact_match_percent']:.2f}%",
)
columns[1].metric("Mean per-record CER", f"{metrics['mean_cer']:.2%}")
columns[2].metric("Mean per-record WER", f"{metrics['mean_wer']:.2%}")
columns[3].metric(
    "Mean recorded inference time",
    f"{metrics['mean_inference_seconds']:.2f} s/image",
)

details = st.columns(3)
details[0].metric("Evaluated test images", f"{metrics['samples']:,}")
details[1].metric("Corpus CER", f"{metrics['corpus_cer']:.2%}")
details[2].metric("Corpus WER", f"{metrics['corpus_wer']:.2%}")

st.caption(
    "Per-record CER/WER are arithmetic means of the saved row metrics. "
    "Corpus CER/WER are recomputed over all saved references and predictions. "
    "Inference time is the saved generation time divided by batch size; it is "
    "not end-to-end upload or PDF processing time."
)

st.subheader("Evaluation charts")
models_dir = Path(__file__).resolve().parents[1] / "models"
chart_columns = st.columns(2)
for column, filename, caption in [
    (chart_columns[0], "exact_match_chart.png", "Case-insensitive exact match"),
    (chart_columns[1], "error_rates_chart.png", "Mean per-record CER and WER"),
]:
    chart_path = models_dir / filename
    if chart_path.is_file():
        column.image(str(chart_path), caption=caption, width="stretch")
    else:
        column.warning(f"Chart not found: {filename}")

with st.expander("Saved test-set records", expanded=False):
    st.dataframe(
        data[["image", "reference", "prediction", "exact_match", "cer", "wer", "inference_seconds"]],
        width="stretch",
        hide_index=True,
    )

st.info(
    "AE and VAE do not generate text, and no matching recognition metrics exist "
    "for those models. Their results are therefore not ranked against TrOCR."
)

st.subheader("Evaluation matrix - measured evidence")
st.dataframe(
    model_evaluation_matrix(metrics),
    width="stretch",
    hide_index=True,
)
