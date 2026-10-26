
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from PDF
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract text from TXT
def extract_text_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# Read all resumes and extract text
def read_resumes(folder_path):
    resumes = []
    filenames = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            try:
                text = extract_text_from_pdf(path)
            except:
                text = ""
        elif file.endswith(".txt"):
            text = extract_text_from_txt(path)
        else:
            continue
        resumes.append(text)
        filenames.append(file)
    return filenames, resumes

# Score resumes using BERT
def score_resumes(jd_text, resumes, filenames):
    all_texts = [jd_text] + resumes
    embeddings = model.encode(all_texts, convert_to_tensor=True)
    jd_embedding = embeddings[0]
    resume_embeddings = embeddings[1:]

    scores = util.cos_sim(jd_embedding, resume_embeddings)[0].cpu().tolist()

    results = []
    for name, score in zip(filenames, scores):
        results.append({"Resume": name, "JD_Similarity": round(score, 2)})
    return pd.DataFrame(results)
