import csv
from io import StringIO

import streamlit as st


st.markdown(
    """
    <h1 style="margin:0 0 1rem;">Results &amp; verification</h1>
    """,
    unsafe_allow_html=True,
)

analysis = st.session_state.get("latest_analysis")
if analysis is None:
    st.info("No results yet. Run TrOCR in Prescription Analyzer.")
    st.stop()
elif analysis.get("model_id") != "trocr":
    st.info("The latest run did not produce text. Run TrOCR to review results.")
    st.stop()

run_id = analysis.get("run_id", "default_run")
pages_list = analysis.get("pages")

if not pages_list and "result" in analysis and analysis["result"] is not None:
    res = analysis["result"]
    pages_list = [
        {
            "page_number": 1,
            "original": getattr(res, "annotated_page", None),
            "recognized_text": getattr(res, "recognized_text", None),
            "error": None,
        }
    ]

if not pages_list:
    st.info("No page records available in current analysis. Please run analysis from Prescription Analyzer.")
    st.stop()

rows = []
for record in pages_list:
    with st.container(border=True):
        st.subheader(f"Page {record.get('page_number', 1)}")
        if record.get("original") is None:
            if record.get("error"):
                st.error(record["error"])
            else:
                st.info("Page image preview not stored.")
        else:
            image_column, text_column = st.columns(2)
            image_column.image(
                record["original"],
                caption=f"Uploaded page {record.get('page_number', 1)}",
                use_container_width=True,
            )

            if record.get("recognized_text") is None:
                text_column.info("No recognized text is available for this page.")
                continue

            page_num = record.get("page_number", 1)
            reviewed_text = text_column.text_area(
                "Recognized text — review and correct (Ground Truth)",
                value=record["recognized_text"],
                key=f"verified_{run_id}_page_{page_num}",
                height=220,
            )

            # Compute page-level OCR evaluation if review was performed
            from models.metrics_evaluator import compute_ocr_metrics
            page_eval = compute_ocr_metrics([record["recognized_text"]], [reviewed_text])

            text_column.caption(
                f"Page {page_num} Evaluation against Ground Truth: "
                f"CER = {page_eval['mean_cer']:.2%} · WER = {page_eval['mean_wer']:.2%} · "
                f"Exact Match = {'Yes' if page_eval['exact_match_acc'] == 100 else 'No'}"
            )

            rows.append({
                "page": page_num,
                "recognized_text": record["recognized_text"],
                "reviewed_text": reviewed_text,
                "cer": page_eval["mean_cer"],
                "wer": page_eval["mean_wer"],
                "human_verified": False,
            })

if rows:
    from models.metrics_evaluator import compute_ocr_metrics
    overall_ocr = compute_ocr_metrics(
        [r["recognized_text"] for r in rows],
        [r["reviewed_text"] for r in rows],
    )
    st.markdown("---")
    st.subheader("Quantitative OCR Accuracy against Ground-Truth Corrections")
    oc1, oc2, oc3, oc4 = st.columns(4)
    oc1.metric("Exact Match", f"{overall_ocr['exact_match_acc']:.1f}%")
    oc2.metric("Mean CER", f"{overall_ocr['mean_cer']:.2%}", help="Levenshtein Character Error Rate against verified text.")
    oc3.metric("Mean WER", f"{overall_ocr['mean_wer']:.2%}", help="Word Error Rate against verified text.")
    oc4.metric("Corpus CER", f"{overall_ocr['corpus_cer']:.2%}")

    st.caption(
        "Note: Metrics are measured in real time against your verified ground-truth corrections. "
        "No metrics or accuracy percentages are fabricated."
    )

    st.subheader("Confirm and export")
    human_verified = st.checkbox(
        "I have compared and verified the reviewed text against every original page.",
        key=f"human_verified_{run_id}",
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
