from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from openai import OpenAI

def summarize_text_tfidf(text, max_sentences=3):
    """TF-IDF 기반 간단 요약 함수"""
    sentences = [s.strip() for s in text.split("。") if s.strip()]
    if len(sentences) <= max_sentences:
        return text.strip()

    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    scores = np.sum(vectors, axis=1)
    top_indices = scores.argsort()[-max_sentences:][::-1]
    summary = [sentences[i] for i in sorted(top_indices)]
    return " ".join(summary)

def summarize_text_with_openai(text, api_key, model="gpt-3.5-turbo", max_tokens=300):
    """OpenAI GPT 기반 요약 함수"""
    client = OpenAI(api_key=api_key)

    prompt = f"다음 문서를 한국어로 간결하게 요약해줘:\n\n{text[:2000]}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()
