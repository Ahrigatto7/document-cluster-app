import streamlit as st
from pipeline import load_documents, vectorize_documents, cluster_documents
from summarize import summarize_text_with_openai
import pandas as pd
import os

# 📌 안전하게 OpenAI API 키 로드
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = st.text_input("🔑 OpenAI API Key", type="password")

st.set_page_config(page_title="문서 클러스터링", layout="wide")
st.title("📄 문서 클러스터링 및 요약 대시보드")

# 📂 문서 업로드 UI
uploaded_files = st.file_uploader("📁 분석할 문서를 업로드하세요 (txt, pdf, docx)", type=["txt", "pdf", "docx"], accept_multiple_files=True)

# 🔘 요약 방식 선택
summary_method = st.selectbox("요약 방식 선택", ["기본 요약 (TF-IDF)", "OpenAI GPT 요약"])

# 🚀 실행 버튼
if st.button("문서 분석 시작"):
    with st.spinner("🔍 문서 분석 중..."):
        # ✅ 문서 저장
        os.makedirs("documents", exist_ok=True)
        for uploaded_file in uploaded_files:
            with open(os.path.join("documents", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

        # 📄 문서 불러오기 & 처리
        docs = load_documents("documents")
        embeddings = vectorize_documents(docs)
        clustered_docs = cluster_documents(docs, embeddings, n_clusters=4)

        # ✨ 요약 처리 (선택에 따라 OpenAI 사용)
        for doc in clustered_docs:
            if summary_method == "OpenAI GPT 요약" and api_key:
                try:
                    doc["summary"] = summarize_text_with_openai(doc["text"], api_key)
                except Exception as e:
                    doc["summary"] = f"❌ OpenAI 요약 오류: {e}"
            else:
                doc.setdefault("summary", doc.get("summary", "요약 없음"))

        # 📊 결과 출력
        df = pd.DataFrame(clustered_docs)
        df.to_csv("clustered_documents.csv", index=False)

        for cluster_id in sorted(df["cluster"].unique()):
            st.header(f"🔹 클러스터 {cluster_id}")
            for _, row in df[df["cluster"] == cluster_id].iterrows():
                st.subheader(f"📄 {os.path.basename(row['file'])}")
                st.write(f"📝 요약: {row['summary']}")
                with st.expander("📚 전체 텍스트 보기"):
                    st.write(row["text"])

        st.success("✅ 분석 완료! clustered_documents.csv 저장됨.")
