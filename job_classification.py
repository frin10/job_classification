# job_clustering_streamlit.py
import streamlit as st
import pandas as pd
import time
import requests
import joblib
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Job Clustering", layout="centered")
st.title("🔍 Job Clustering with Hierarchical Clustering")
st.write("Scrape jobs from Karkidi.com and cluster them based on required skills.")

# --------- Function Definitions ---------

def scrape_karkidi_jobs(keyword="data science", pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        st.info(f"Scraping page {page}...")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except Exception as e:
            st.warning(f"Failed to fetch page {page}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        job_blocks = soup.find_all("div", class_="ads-details")

        for job in job_blocks:
            try:
                title = job.find("h4").get_text(strip=True)
                company_tag = job.find("a", href=lambda x: x and "Employer-Profile" in x)
                company = company_tag.get_text(strip=True) if company_tag else "N/A"
                location = job.find("p").get_text(strip=True)
                experience_tag = job.find("p", class_="emp-exp")
                experience = experience_tag.get_text(strip=True) if experience_tag else "N/A"
                key_skills_tag = job.find("span", string="Key Skills")
                skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
                summary_tag = job.find("span", string="Summary")
                summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

                jobs_list.append({
                    "Title": title,
                    "Company": company,
                    "Location": location,
                    "Experience": experience,
                    "Summary": summary,
                    "Skills": skills
                })
            except Exception as e:
                continue
        time.sleep(1)
    return pd.DataFrame(jobs_list)

def preprocess_skills(df):
    df = df.copy()
    df['Skills'] = df['Skills'].fillna("").str.lower().str.strip()
    return df

def split_skills_tokenizer(x):
    return x.split(',')

def vectorize_skills(df):
    vectorizer = TfidfVectorizer(tokenizer=split_skills_tokenizer, lowercase=True)
    X = vectorizer.fit_transform(df['Skills'])
    return X, vectorizer

def cluster_skills(X, n_clusters=5):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X.toarray())
    return model, labels

# --------- Streamlit UI ---------

search_term = st.text_input("Enter job keyword (e.g., Data Science, Machine Learning):", "data science")
pages = st.slider("Number of pages to scrape from Karkidi:", 1, 5, 2)
clusters = st.slider("Number of skill-based clusters to form:", 2, 10, 5)

if st.button("🚀 Run Clustering"):
    with st.spinner("Processing..."):
        # Step 1: Scrape
        df = scrape_karkidi_jobs(search_term, pages)
        if df.empty:
            st.error("❌ No jobs found. Try another keyword.")
            st.stop()

        # Step 2: Preprocess and Vectorize
        df = preprocess_skills(df)
        X, vectorizer = vectorize_skills(df)

        # Step 3: Clustering
        model, labels = cluster_skills(X, n_clusters=clusters)
        df['Cluster'] = labels

        # Step 4: Display
        st.success(f"Clustered {len(df)} jobs into {clusters} clusters.")
        selected_cluster = st.selectbox("Select a cluster to view:", sorted(df['Cluster'].unique()))
        st.dataframe(df[df['Cluster'] == selected_cluster][['Title', 'Company', 'Skills']])

        # Step 5: Download option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Clustered CSV", data=csv, file_name="clustered_jobs.csv", mime="text/csv")
