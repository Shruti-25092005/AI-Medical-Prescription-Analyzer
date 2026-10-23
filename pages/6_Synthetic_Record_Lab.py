import pandas as pd
import streamlit as st

from models.synthetic_record_gan import (
    CHECKPOINT_PATH,
    SYNTHETIC_OUTPUT_LABEL,
    generate_records,
    records_as_dicts,
)


st.markdown(
    """
    <div style="margin-bottom:1.1rem;">
      <div style="font-size:.76rem; text-transform:uppercase; letter-spacing:.12em;
          font-weight:700; color:#167d78;">Research workspace</div>
      <h1 style="margin:.45rem 0 .25rem;">Synthetic record lab</h1>
      <p style="max-width:760px; margin:0;">
        Generate fictional placeholder records using the local GAN checkpoint.
        This workflow is separate from prescription analysis.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.error(
    f"{SYNTHETIC_OUTPUT_LABEL}. Samples contain no identity or clinical fields."
)

with st.container(border=True):
    st.subheader("Generate samples")
    if not CHECKPOINT_PATH.is_file():
        st.warning(
            "The GAN checkpoint is not available. Open and run "
            "`notebooks/Synthetic_Record_GAN.ipynb` to train the model."
        )
        st.stop()

    st.success(f"GAN checkpoint ready: {CHECKPOINT_PATH.name}")
    controls = st.columns([1, 2])
    count = controls[0].number_input(
        "Number of fictional records",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
    )
    generate_clicked = controls[1].button(
        "Generate fictional records",
        type="primary",
        icon=":material/auto_awesome:",
        use_container_width=True,
    )
    if generate_clicked:
        try:
            with st.spinner("Generating fictional records..."):
                generated_records = generate_records(int(count))
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            st.error(
                "Synthetic samples could not be generated. Check that the "
                "local GAN checkpoint is available."
            )
        else:
            st.session_state["synthetic_records"] = generated_records

generated_records = st.session_state.get("synthetic_records", [])
if generated_records:
    st.subheader("AI-generated fictional records")
    record_columns = st.columns(2)
    for index, (_, image) in enumerate(generated_records):
        record_columns[index % len(record_columns)].image(
            image,
            caption=f"{SYNTHETIC_OUTPUT_LABEL} — Sample {index + 1}",
            width="stretch",
        )
    st.dataframe(
        pd.DataFrame(records_as_dicts([record for record, _ in generated_records])),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Output is synthetic demonstration content only. It is not a valid "
        "prescription, diagnosis, or treatment recommendation."
    )
