from openai import OpenAI

def summarize_text_with_openai(text, api_key, model="gpt-3.5-turbo", max_tokens=300):
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
