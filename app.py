import streamlit as st

from models.privacy_controls import clear_uploaded_data


st.set_page_config(
    page_title="Prescription Analyzer & Generative AI Studio",
    page_icon="Rx",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #19383c;
        --muted: #668084;
        --teal: #167d78;
        --teal-dark: #10645f;
        --blue: #4a83b5;
        --line: #e1eceb;
        --surface: #ffffff;
        --soft: #f3f8f8;
    }
    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system,
            BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--ink);
    }
    .stApp {background: #f7fafb;}
    [data-testid="stHeader"] {background: rgba(247, 250, 251, .9);}
    .block-container {
        max-width: 1440px;
        padding: 2.25rem clamp(1.1rem, 4vw, 3.5rem) 4rem;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -.035em;
        font-weight: 700;
    }
    h1 {font-size: clamp(2rem, 4vw, 2.75rem); margin-bottom: .4rem;}
    h2 {font-size: 1.55rem; margin-top: 1.65rem;}
    h3 {font-size: 1.15rem;}
    p, label, [data-testid="stCaptionContainer"] {color: var(--muted);}
    [data-testid="stSidebar"] {
        background: #fff;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child {padding-top: 1rem;}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.25rem;
        letter-spacing: -.02em;
        color: var(--teal-dark);
    }
    [data-testid="stSidebarNav"] {padding-top: .5rem;}
    [data-testid="stSidebarNav"] a {
        border-radius: 12px;
        margin: .15rem .4rem;
        min-height: 2.8rem;
        transition: background .15s ease, color .15s ease;
    }
    [data-testid="stSidebarNav"] a:hover {background: #eef7f6;}
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: #e7f4f3;
        color: var(--teal-dark);
        font-weight: 650;
    }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 5px 18px rgba(23, 61, 64, .035);
    }
    [data-testid="stMetricLabel"] {color: var(--muted);}
    [data-testid="stMetricValue"] {color: var(--ink); font-weight: 700;}
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border: 1px solid var(--line) !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 24px rgba(23, 61, 64, .045);
    }
    [data-testid="stFileUploader"] section {
        background: #f5faf9;
        border: 1px dashed #a8ccca;
        border-radius: 16px;
    }
    .stButton > button, .stDownloadButton > button,
    [data-testid="stPageLink"] a {
        border-radius: 12px;
        font-weight: 650;
        min-height: 2.75rem;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    [data-testid="stPageLink"] a,
    a[data-testid^="stPageLink"] {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: .65rem .9rem;
        border: 1px solid #d4e6e4;
        background: #fff;
        color: var(--teal-dark);
        box-shadow: 0 4px 14px rgba(23, 61, 64, .04);
    }
    [data-testid="stPageLink"] a:hover,
    a[data-testid^="stPageLink"]:hover {
        background: #eef7f6;
        border-color: #9fc9c6;
        color: var(--teal-dark);
    }
    .stButton > button[kind="primary"] {
        background: var(--teal);
        border-color: var(--teal);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(22, 125, 120, .15);
        border-color: var(--teal);
        color: var(--teal-dark);
    }
    [data-testid="stAlert"], [data-testid="stStatusWidget"] {
        border-radius: 14px;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
    }
    [data-testid="stExpander"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 14px;
    }
    [data-testid="stProgressBar"] > div > div {background: var(--teal);}
    @media (max-width: 768px) {
        .block-container {padding: 1.25rem 1rem 2.5rem;}
        [data-testid="stMetric"] {padding: .8rem;}
        [data-testid="stSidebar"] {min-width: 0;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("# Rx Workspace")
st.sidebar.caption("Prescription AI & Generative Handwriting")
st.sidebar.info(
    "Privacy: uploaded files and results remain strictly in this session memory for "
    "processing. The app does not save uploaded patient records to permanent files."
)
st.sidebar.warning(
    "AI limitations: transcription and generation can be incomplete or uncertain. "
    "Verify all results with qualified healthcare professionals."
)


def _clear_session_uploads() -> None:
    clear_uploaded_data(st.session_state)


st.sidebar.button(
    "Clear uploaded data and results",
    on_click=_clear_session_uploads,
    use_container_width=True,
)

navigation = st.navigation(
    {
        "Core Applications": [
            st.Page(
                "pages/2_Prescription_Analyzer.py",
                title="1 · Prescription Analyzer",
                icon=":material/document_scanner:",
                default=True,
            ),
            st.Page(
                "pages/6_Handwriting_Generation_Studio.py",
                title="2 · Handwriting Generation Studio",
                icon=":material/auto_awesome:",
            ),
            st.Page(
                "pages/4_Performance_Evaluation.py",
                title="3 · Model Evaluation Dashboard",
                icon=":material/analytics:",
            ),
            st.Page(
                "pages/3_AI_Models.py",
                title="4 · About & Research Info",
                icon=":material/info:",
            ),
        ],
        "Clinical Utilities & Labs": [
            st.Page(
                "pages/5_Results_Verification.py",
                title="Results & Verification",
                icon=":material/fact_check:",
            ),
            st.Page(
                "pages/1_Dashboard.py",
                title="Workspace Overview",
                icon=":material/dashboard:",
            ),
            st.Page(
                "pages/7_Diffusion_Synthetic_Lab.py",
                title="Procedural Diffusion Lab",
                icon=":material/blur_on:",
            ),
            st.Page(
                "pages/6_Synthetic_Record_Lab.py",
                title="Legacy Record Lab",
                icon=":material/history_edu:",
            ),
        ],
    },
    position="sidebar",
)
navigation.run()
