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

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Cleaning function
def clean(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in words if w not in stop_words]
    return ' '.join(filtered)

# Streamlit UI
st.title("📚 Dynamic Syllabus vs Industry Analyzer")

st.markdown("Upload your **Syllabus** and **Job CSV** to analyze alignment.")

# File uploads
syllabus_file = st.file_uploader("Upload Syllabus CSV", type=["csv"])
job_file = st.file_uploader("Upload Job Descriptions CSV", type=["csv"])

if syllabus_file is not None and job_file is not None:
    syllabus_df = pd.read_csv(syllabus_file)
    job_df = pd.read_csv(job_file)

    st.subheader("📘 Syllabus Columns")
    st.write(syllabus_df.columns.tolist())

    st.subheader("💼 Job Columns")
    st.write(job_df.columns.tolist())

    # Validate columns
    if 'Description' not in syllabus_df.columns or 'Description' not in job_df.columns:
        st.error("❌ One of the files is missing the 'Description' column.")
    elif 'Subject' not in syllabus_df.columns or 'Job Title' not in job_df.columns:
        st.error("❌ One of the files is missing 'Subject' or 'Job Title' column.")
    else:
        # Clean text
        syllabus_df['cleaned'] = syllabus_df['Description'].apply(clean)
        job_df['cleaned'] = job_df['Description'].apply(clean)

        # TF-IDF
        all_text = list(syllabus_df['cleaned']) + list(job_df['cleaned'])
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_text)

        syllabus_vecs = tfidf_matrix[:len(syllabus_df)]
        job_vecs = tfidf_matrix[len(syllabus_df):]

        similarity_matrix = cosine_similarity(syllabus_vecs, job_vecs)

        # DataFrame version
        sim_df = pd.DataFrame(similarity_matrix,
                              index=syllabus_df['Subject'],
                              columns=job_df['Job Title'])

        st.subheader("📊 Similarity Matrix")
        st.dataframe(sim_df.style.format("{:.2f}"))

        # Best match analysis
        st.subheader("🎯 Best Job Role for Each Subject")
        threshold = 0.3
        for i, subject in enumerate(syllabus_df['Subject']):
            scores = similarity_matrix[i]
            best_idx = scores.argmax()
            best_score = scores[best_idx]
            best_job = job_df['Job Title'].iloc[best_idx]
            st.markdown(f"**{subject}** ➡️ *{best_job}* — **{round(best_score * 100, 1)}%** match")
            if best_score < threshold:
                st.warning("⚠️ Low match — consider updating or revising this subject.")

        # Heatmap
        st.subheader("🗺️ Heatmap of Similarity")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                    xticklabels=job_df['Job Title'],
                    yticklabels=syllabus_df['Subject'])
        st.pyplot(fig)
