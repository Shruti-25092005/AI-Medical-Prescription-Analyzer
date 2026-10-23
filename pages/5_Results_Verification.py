import csv
from io import StringIO

import streamlit as st


st.markdown(
    """
    <div style="margin-bottom:1.1rem;">
      <div style="font-size:.76rem; text-transform:uppercase; letter-spacing:.12em;
          font-weight:700; color:#167d78;">Review workflow</div>
      <h1 style="margin:.45rem 0 .25rem;">Results &amp; verification</h1>
      <p style="max-width:760px; margin:0;">
        Compare the source page with recognized text, make corrections, and
        export only after confirming your review.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "OCR can be incorrect. Review every character against the source. "
    "This informational transcription does not provide diagnoses or "
    "treatment recommendations."
)

analysis = st.session_state.get("latest_analysis")
if analysis is None:
    st.info(
        "No page results are available yet. Use Prescription Analyzer and "
        "select TrOCR to recognize prescription text."
    )
    st.stop()

if analysis["model_id"] != "trocr":
    st.info(
        "The latest run used an image-reconstruction model, which does not "
        "produce text. Run TrOCR from Prescription Analyzer to review and "
        "export recognized text."
    )
    st.stop()

rows = []
for record in analysis["pages"]:
    with st.container(border=True):
        st.subheader(f"Page {record['page_number']}")
        if record["original"] is None:
            st.error(record["error"] or "This page could not be read.")
            continue
        if record["error"]:
            st.error(record["error"])

        image_column, text_column = st.columns(2)
        image_column.image(
            record["original"],
            caption=f"Uploaded page {record['page_number']}",
            width="stretch",
        )

        if record["recognized_text"] is None:
            text_column.info("No recognized text is available for this page.")
            continue

        reviewed_text = text_column.text_area(
            "Recognized text — review and correct",
            value=record["recognized_text"],
            key=f"verified_{analysis['run_id']}_page_{record['page_number']}",
            height=220,
        )
        rows.append({
            "page": record["page_number"],
            "recognized_text": record["recognized_text"],
            "reviewed_text": reviewed_text,
            "human_verified": False,
        })

if rows:
    st.subheader("Confirm and export")
    human_verified = st.checkbox(
        "I have compared and verified the reviewed text against every original page.",
        key=f"human_verified_{analysis['run_id']}",
    )
    if not human_verified:
        st.caption("Confirm the review above to enable export.")

    for row in rows:
        row["human_verified"] = human_verified

    csv_buffer = StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "page",
            "recognized_text",
            "reviewed_text",
            "human_verified",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

    txt_content = "\n\n".join(
        f"Page {row['page']}\n{row['reviewed_text']}" for row in rows
    )
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Download reviewed text (TXT)",
        data=txt_content,
        file_name="prescription_review.txt",
        mime="text/plain",
        disabled=not human_verified,
    )
    download_columns[1].download_button(
        "Download reviewed results (CSV)",
        data=csv_buffer.getvalue(),
        file_name="prescription_review.csv",
        mime="text/csv",
        disabled=not human_verified,
    )
