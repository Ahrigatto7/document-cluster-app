from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_retriever(docs, openai_api_key, chunk_size=500, chunk_overlap=50):
    texts = [doc.get("text", "") for doc in docs if doc.get("text", "")]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.create_documents(texts)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(split_docs, embeddings)
    return db.as_retriever()

def ask_question(retriever, question, openai_api_key):
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
        retriever=retriever
    )
    return qa_chain.run(question)
