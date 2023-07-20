import json
import requests
from flask import jsonify
import openai
import constants

# GPT api setup 
GPT_MODEL = constants.GPT_MODEL
COMPLETION_MODEL = constants.COMPLETION_MODEL
OPENAI_API_KEY = constants.OPENAI_API_KEY
chatCompletionURL = constants.chatCompletionURL
completionURL = constants.completionURL
embeddingURL = constants.embeddingURL

# request header 
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

# Call Chat Completion API 
def callChatCompletionAPI(messages): 
    """ 
    Format of the message: 
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "How many citys are in the world?"}
    ]
    """
    payload = dict(
        model = GPT_MODEL,
        messages = messages, 
        max_tokens = 500
    )

    data = json.dumps(payload).encode('utf-8')

    return requests.post(
        url=chatCompletionURL,
        headers=headers,
        data=data
    )


# Call Completion API 
def callCompletionAPI(prompt): 
    payload = dict(
        model=COMPLETION_MODEL,
        prompt=prompt
    )

    data = json.dumps(payload).encode('utf-8')

    return requests.post(
        url=completionURL,
        headers=headers,
        data=data
    )


# Call Embedding API
def callEmbeddingsAPI(input):
    payload = dict(
        model=GPT_MODEL,
        input=input
    )

    data = json.dumps(payload).encode('utf-8')

    return requests.post(
        url=embeddingURL,
        headers=headers,
        data=data
    )
