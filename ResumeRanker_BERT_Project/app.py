# Streamlit UI will be added here
import streamlit as st
import pandas as pd
import os
import shutil
from utils import read_resumes, score_resumes

st.set_page_config(page_title="AI Resume Ranker", layout="wide")

st.title("📄 AI-Powered Resume Ranker using BERT")
st.markdown("Upload resumes (.txt/.pdf) and paste a job description to see which candidate matches best using semantic similarity.")

# Create uploads folder
upload_folder = "resumes"
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Job description input
jd_text = st.text_area("📝 Paste Job Description Here", height=200)

# File uploader
uploaded_files = st.file_uploader("📤 Upload Resumes (.txt or .pdf)", type=["txt", "pdf"], accept_multiple_files=True)

# Save uploaded files
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(upload_folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

# Rank resumes
if st.button("🚀 Rank Resumes"):
    if not jd_text.strip():
        st.warning("Please paste the job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        with st.spinner("Scoring resumes using BERT..."):
            filenames, resume_texts = read_resumes(upload_folder)
            result_df = score_resumes(jd_text, resume_texts, filenames)
            result_df = result_df.sort_values(by="JD_Similarity", ascending=False).reset_index(drop=True)

            st.success("✅ Ranking completed!")
            st.subheader("📊 Resume Rankings:")
            st.dataframe(result_df)

            # Export to Excel
            result_df.to_excel("output.xlsx", index=False)
            with open("output.xlsx", "rb") as f:
                st.download_button("⬇️ Download Results as Excel", f, file_name="resume_rank_results.xlsx")
