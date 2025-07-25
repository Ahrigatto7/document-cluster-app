import streamlit as st
from pipeline import load_documents, vectorize_documents, cluster_documents
from summarize import summarize_text_tfidf, summarize_text_with_openai
from rag_module import build_retriever, ask_question
import pandas as pd
import os

# 기본 설정
st.set_page_config(page_title="문서 분석 + GPT 질의응답", layout="wide")
st.title("📄 문서 클러스터링 및 요약 + 질의응답")

# OpenAI API Key 입력 처리
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = st.text_input("🔑 OpenAI API Key", type="password")

# 문서 업로드
uploaded_files = st.file_uploader("📂 문서 업로드", type=["txt", "pdf", "docx"], accept_multiple_files=True)

# 요약 방식 선택
summary_method = st.selectbox("📝 요약 방식 선택", ["TF-IDF", "OpenAI GPT"])

# 문서 분석 시작
if st.button("문서 분석 시작") and uploaded_files:
    os.makedirs("documents", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("documents", file.name), "wb") as f:
            f.write(file.getbuffer())

    docs = load_documents("documents")
    embeddings = vectorize_documents(docs)
    clustered_docs = cluster_documents(docs, embeddings, n_clusters=4)

    summarized_docs = []
    for doc in clustered_docs:
        try:
            if summary_method == "OpenAI GPT" and api_key:
                doc["summary"] = summarize_text_with_openai(doc["text"], api_key)
            else:
                doc["summary"] = summarize_text_tfidf(doc["text"])
        except Exception as e:
            doc["summary"] = f"❌ 요약 실패: {e}"
        summarized_docs.append(doc)

    st.session_state["docs"] = summarized_docs
    df = pd.DataFrame(summarized_docs)
    df.to_csv("clustered_documents.csv", index=False)

    for cluster_id in sorted(df["cluster"].unique()):
        st.header(f"🔹 클러스터 {cluster_id}")
        for _, row in df[df["cluster"] == cluster_id].iterrows():
            st.subheader(f"📄 {os.path.basename(row['file'])}")
            st.write(f"📝 요약: {row['summary']}")
            with st.expander("📚 전체 텍스트 보기"):
                st.write(row["text"])

# 질문 입력 영역
st.markdown("---")
st.subheader("💬 업로드된 문서 기반 GPT 질의응답")

question = st.text_input("❓ 질문을 입력하세요")

if st.button("질문 실행") and question and "docs" in st.session_state:
    try:
        retriever = build_retriever(st.session_state["docs"], api_key)
        response = ask_question(retriever, question, api_key)
        st.success("🧠 GPT 응답:")
        st.write(response)
    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
