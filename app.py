import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load CSV files
syllabus_df = pd.read_csv(r'C:\Users\ASUS\Downloads\syllabus.csv')
job_df = pd.read_csv(r'C:\Users\ASUS\Downloads\job_data.csv')


print("Syllabus CSV Columns:", syllabus_df.columns)
print("Job CSV Columns:", job_df.columns)


SYLLABUS_TEXT_COLUMN =  'Description'    
JOB_TEXT_COLUMN = 'Description'              


def clean(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

syllabus_df['cleaned'] = syllabus_df[SYLLABUS_TEXT_COLUMN].apply(clean)
job_df['cleaned'] = job_df[JOB_TEXT_COLUMN].apply(clean)


all_text = list(syllabus_df['cleaned']) + list(job_df['cleaned'])


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_text)

print(" TF-IDF matrix shape:", tfidf_matrix.shape)


feature_names = vectorizer.get_feature_names_out()
print(" Sample feature words:", feature_names[:10])


print("TF-IDF vector for first syllabus entry:\n", tfidf_matrix[0])


syllabus_vectors = tfidf_matrix[:len(syllabus_df)]
job_vectors = tfidf_matrix[len(syllabus_df):]


similarity_matrix = cosine_similarity(syllabus_vectors, job_vectors)


print("\n Cosine Similarity Matrix (Syllabus vs Jobs):")
print(np.round(similarity_matrix, 2)) 
print("\n Best Job Match for Each Syllabus Subject:")
for i, subject in enumerate(syllabus_df['Subject']):
    best_match_idx = similarity_matrix[i].argmax()
    best_match_score = similarity_matrix[i][best_match_idx]
    best_job_title = job_df['Job Title'].iloc[best_match_idx]
    print(f"{subject}  {best_job_title} (Score: {best_match_score:.2f})")

plt.figure(figsize=(12, 8))
sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=job_df['Job Title'],
            yticklabels=syllabus_df['Subject'])

plt.title("Syllabus vs Job Role Similarity")
plt.xlabel("Job Roles")
plt.ylabel("Syllabus Subjects")
plt.tight_layout()
plt.show()


print("\n Subject-to-Job Role Analysis:")
print("────────────────────────────────────────────")

low_match_threshold = 0.30  # Set threshold to flag outdated subjects

for i, subject in enumerate(syllabus_df['Subject']):
    scores = similarity_matrix[i]
    best_match_idx = scores.argmax()
    best_match_score = scores[best_match_idx]
    best_job = job_df['Job Title'].iloc[best_match_idx]
    
    match_percent = round(best_match_score * 100, 1)
    print(f" {subject} {best_job} — Similarity: {match_percent}%")
    
    if best_match_score < low_match_threshold:
        print(f" Low industry relevance. Consider updating this subject.")

