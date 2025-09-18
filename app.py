# app.py
import streamlit as st
import pandas as pd
import io
import json
import time
from analyzers import analyze_dataset
from groq_report import generate_report_streaming, GROQ_AVAILABLE  # we will rename internally

st.set_page_config(page_title="Leakage Sentinel v3", layout="wide")
st.title("🧹 Leakage Sentinel v3 — Data Leakage Analysis")

# --- File upload ---
uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded is None:
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.write(f"Dataset shape: {df.shape}")
st.dataframe(df.head())

# --- Target column and business context ---
target_col = st.selectbox("Select target column", df.columns.tolist())
business_scenario = st.text_area("Business scenario / context (optional)", "")

# --- Run analysis button ---
if st.button("Run Analysis") or "analysis" in st.session_state:
    if "analysis" not in st.session_state:
        with st.spinner("Running math-based feature analysis..."):
            start = time.time()
            st.session_state.analysis = analyze_dataset(df, target_col, sample_frac=1.0, n_jobs=4, cv=3)
            elapsed = time.time() - start
        st.success(f"Analysis completed in {elapsed:.1f}s.")

    analysis = st.session_state.analysis

    # --- Display metadata ---
    st.subheader("Dataset Metadata")
    st.json(analysis["metadata"])

    # --- Feature table ---
    st.subheader("Top Features by Composite Risk Score")
    results_df = pd.DataFrame(analysis["results"]).head(50)
    st.dataframe(results_df)

    # --- Risk chart ---
    st.subheader("Feature Risk Visualization")
    st.bar_chart(results_df.set_index("feature")["composite_score"])

    # --- Downloads ---
    st.download_button(
        "Download Analysis JSON",
        json.dumps(analysis, indent=2),
        "analysis.json",
        mime="application/json"
    )
    st.download_button(
        "Download Analysis CSV",
        results_df.to_csv(index=False),
        "analysis.csv",
        mime="text/csv"
    )

    # --- Automated report ---
    st.subheader("Automated Data Leakage Analysis Report")
    if not GROQ_AVAILABLE:
        st.warning("AI-powered report generation is not available. Please check the SDK or API key.")
    else:
        if "report_text" not in st.session_state:
            st.session_state.report_text = ""

        if st.button("Generate Detailed Report") or st.session_state.report_text:
            if not st.session_state.report_text:
                with st.spinner("Generating detailed report..."):
                    st.session_state.report_text = generate_report_streaming(
                        analysis, business_scenario=business_scenario
                    )
            st.markdown(st.session_state.report_text)