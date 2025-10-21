import streamlit as st
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import os

# -----------------------------
# Use local nltk_data folder in repo
# -----------------------------
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Add nltk_data folder to NLTK's search path
nltk.data.path.append(nltk_data_dir)

# -----------------------------
# Ensure punkt and stopwords are available locally
# -----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

# -----------------------------
# Text cleaning function
# -----------------------------
def clean(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in words if w not in stop_words]
    return ' '.join(filtered)

# -----------------------------
# Default CSV data
# -----------------------------
default_syllabus = pd.DataFrame({
    "Subject": ["Data Structures", "Web Development", "Machine Learning",
                "Database Management", "Operating Systems", "Cyber Security"],
    "Description": [
        "Covers arrays, linked lists, stacks, queues, trees, and hash tables",
        "Frontend and backend development using HTML, CSS, JavaScript, React",
        "Supervised and unsupervised learning, regression, classification, neural networks",
        "SQL, database design, normalization, transactions, indexing",
        "Processes, threads, memory management, scheduling, file systems",
        "Network security, encryption, firewalls, penetration testing"
    ]
})

default_jobs = pd.DataFrame({
    "Job Title": ["Data Analyst", "Frontend Developer", "ML Engineer",
                  "Database Administrator", "System Administrator", "Cybersecurity Analyst"],
    "Description": [
        "SQL, Python, data cleaning, visualization, business reporting",
        "HTML, CSS, JavaScript, React, responsive design",
        "Python, machine learning, TensorFlow, scikit-learn, model deployment",
        "Database administration, SQL, performance tuning, backups",
        "Operating systems, networking, Linux, troubleshooting",
        "Cybersecurity, penetration testing, firewalls, risk assessment"
    ]
})

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Dynamic Syllabus vs Industry Analyzer")
st.markdown("Upload your **Syllabus** and **Job CSV** to analyze alignment, or use the default example data.")

syllabus_file = st.file_uploader("Upload Syllabus CSV", type=["csv"])
job_file = st.file_uploader("Upload Job Descriptions CSV", type=["csv"])

syllabus_df = pd.read_csv(syllabus_file) if syllabus_file else default_syllabus
job_df = pd.read_csv(job_file) if job_file else default_jobs

# Validate necessary columns
if 'Description' not in syllabus_df.columns or 'Description' not in job_df.columns:
    st.error("One of the files is missing the 'Description' column.")
elif 'Subject' not in syllabus_df.columns or 'Job Title' not in job_df.columns:
    st.error("One of the files is missing 'Subject' or 'Job Title' column.")
else:
    # Clean text
    syllabus_df['cleaned'] = syllabus_df['Description'].apply(clean)
    job_df['cleaned'] = job_df['Description'].apply(clean)

    # TF-IDF vectorization
    all_text = list(syllabus_df['cleaned']) + list(job_df['cleaned'])
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)

    syllabus_vecs = tfidf_matrix[:len(syllabus_df)]
    job_vecs = tfidf_matrix[len(syllabus_df):]

    similarity_matrix = cosine_similarity(syllabus_vecs, job_vecs)

    # DataFrame version for display
    sim_df = pd.DataFrame(similarity_matrix,
                          index=syllabus_df['Subject'],
                          columns=job_df['Job Title'])

    st.subheader("Similarity Matrix")
    st.dataframe(sim_df.style.format("{:.2f}"))

    # Best match analysis
    st.subheader("Best Job Role for Each Subject")
    threshold = 0.3
    for i, subject in enumerate(syllabus_df['Subject']):
        scores = similarity_matrix[i]
        best_idx = scores.argmax()
        best_score = scores[best_idx]
        best_job = job_df['Job Title'].iloc[best_idx]
        st.markdown(f"**{subject}** ➡️ *{best_job}* — **{round(best_score * 100, 1)}%** match")
        if best_score < threshold:
            st.warning("Low match — consider updating or revising this subject.")

    # Heatmap visualization
    st.subheader("Heatmap of Similarity")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=job_df['Job Title'],
                yticklabels=syllabus_df['Subject'])
    st.pyplot(fig)
