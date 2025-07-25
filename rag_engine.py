from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_retriever(docs, openai_api_key):
    texts = [doc["text"] for doc in docs]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_texts = splitter.create_documents(texts)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(split_texts, embeddings)
    retriever = db.as_retriever()
    return retriever

def ask_question(retriever, question, openai_api_key):
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
        retriever=retriever
    )
    return qa.run(question)
