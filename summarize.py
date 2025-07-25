from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import openai

def summarize_text_tfidf(text, num_sentences=2):
    sentences = text.split(". ")
    if len(sentences) <= num_sentences:
        return text
    tfidf = TfidfVectorizer().fit_transform(sentences)
    scores = tfidf.sum(axis=1).A1
    top_idx = sorted(scores.argsort()[-num_sentences:])
    return ". ".join([sentences[i] for i in top_idx])

def summarize_text_with_openai(text, model="gpt-3.5-turbo", max_tokens=300):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        return "⚠️ OpenAI API 키가 설정되지 않았습니다."
    prompt = f"다음 문서를 한국어로 간결하게 요약해줘:\n\n{text[:2000]}"
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()