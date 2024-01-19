# Playground 

import openai

openai.api_key = "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"

def sentiment_analysis(text):
    messages = [
        {
            "role": "system",
            "content": """You are trained to analyze and detect the sentiment of given text.
            If you are unsure of an answer, you can say "not sure" and recommend users to review manually."""
        },
        {
            "role": "user",
            "content": f"""Analyze the following text and determine if the sentiment is positive or negative.
            Return the answer in a single word as either 'positive' or 'negative': {text}"""
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0.5
    )

    response_text = response.choices[0].message.content.strip().lower()
    return response_text

input_text = "I love Pakistan"
response = sentiment_analysis(input_text)
print(input_text, "the sentiment is:", response)
