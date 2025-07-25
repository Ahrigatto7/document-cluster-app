
import streamlit as st
from pipeline import load_documents, vectorize_documents, cluster_documents
from summarize import summarize_text_tfidf, summarize_text_with_openai
import pandas as pd
import os

st.title("📄 문서 클러스터링 및 요약 대시보드")

# 🔑 OpenAI API Key 입력 또는 secrets.toml 사용
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = st.text_input("🔑 OpenAI API Key", type="password")

# 🧠 요약 방식 선택
summary_method = st.selectbox("📝 요약 방식 선택", ["TF-IDF", "OpenAI GPT"])

if st.button("문서 분석 시작"):
    docs = load_documents("documents")
    embeddings = vectorize_documents(docs)
    clustered_docs = cluster_documents(docs, embeddings, n_clusters=4)

    summarized_docs = []
    for doc in clustered_docs:
        if summary_method == "TF-IDF":
            summary = summarize_text_tfidf(doc["text"])
        elif summary_method == "OpenAI GPT":
            summary = summarize_text_with_openai(doc["text"], api_key=api_key)
        else:
            summary = ""
        doc["summary"] = summary
        summarized_docs.append(doc)

    df = pd.DataFrame(summarized_docs)
    df.to_csv("clustered_documents.csv", index=False)

    for cluster_id in sorted(df["cluster"].unique()):
        st.header(f"🔹 클러스터 {cluster_id}")
        for _, row in df[df["cluster"] == cluster_id].iterrows():
            st.subheader(os.path.basename(row["file"]))
            st.write(row["summary"])
            with st.expander("전체 텍스트 보기"):
                st.write(row["text"])
