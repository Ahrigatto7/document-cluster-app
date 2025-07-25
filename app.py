import streamlit as st
from pipeline import load_documents, vectorize_documents, cluster_documents
from summarize import summarize_text_with_openai
from rag_module import build_retriever, ask_question
import pandas as pd
import os

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = st.text_input("🔑 OpenAI API Key", type="password")

st.set_page_config(page_title="문서 분석 + 질의응답", layout="wide")
st.title("📄 문서 클러스터링 및 질의응답 대시보드")

uploaded_files = st.file_uploader("📂 문서 업로드", type=["txt", "pdf", "docx"], accept_multiple_files=True)
summary_method = st.selectbox("요약 방식 선택", ["기본 요약 (TF-IDF)", "OpenAI GPT 요약"])

if st.button("문서 분석 시작"):
    os.makedirs("documents", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("documents", file.name), "wb") as f:
            f.write(file.getbuffer())

    docs = load_documents("documents")
    embeddings = vectorize_documents(docs)
    clustered_docs = cluster_documents(docs, embeddings, n_clusters=4)

    for doc in clustered_docs:
        if summary_method == "OpenAI GPT 요약" and api_key:
            try:
                doc["summary"] = summarize_text_with_openai(doc["text"], api_key)
            except Exception as e:
                doc["summary"] = f"❌ 요약 실패: {e}"
        else:
            doc.setdefault("summary", doc.get("summary", "요약 없음"))

    st.session_state["docs"] = clustered_docs
    df = pd.DataFrame(clustered_docs)

    for cluster_id in sorted(df["cluster"].unique()):
        st.header(f"🔹 클러스터 {cluster_id}")
        for _, row in df[df["cluster"] == cluster_id].iterrows():
            st.subheader(f"📄 {os.path.basename(row['file'])}")
            st.write(f"📝 요약: {row['summary']}")
            with st.expander("📚 전체 텍스트 보기"):
                st.write(row["text"])

st.markdown("---")
st.subheader("💬 업로드된 문서에 질문해보세요")
question = st.text_input("❓ 질문 입력")

if st.button("질문 실행") and question:
    try:
        retriever = build_retriever(st.session_state["docs"], api_key)
        response = ask_question(retriever, question, api_key)
        st.success("🧠 GPT 응답:")
        st.write(response)
    except Exception as e:
        st.error(f"❌ 오류: {e}")
