import openai
import time

def call_openai(prompt, model='gpt-3.5-turbo', temperature=0.2):
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=temperature,
        max_tokens=256
    )
    latency = time.time() - start_time
    return response['choices'][0]['message']['content'], latency
