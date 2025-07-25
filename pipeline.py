import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from utils import extract_text_from_file
from summarize import summarize_text_tfidf

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents(directory):
    paths = [os.path.join(directory, f) for f in os.listdir(directory)
             if f.lower().endswith((".pdf", ".docx", ".txt"))]
    docs = []
    for path in paths:
        text = extract_text_from_file(path)
        if text:
            docs.append({"file": path, "text": text})
    return docs

def vectorize_documents(docs):
    texts = [doc["text"] for doc in docs]
    embeddings = model.encode(texts)
    return embeddings

def cluster_documents(docs, embeddings, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    for i, doc in enumerate(docs):
        doc["cluster"] = int(kmeans.labels_[i])
        doc["summary"] = summarize_text_tfidf(doc["text"])
    return docs