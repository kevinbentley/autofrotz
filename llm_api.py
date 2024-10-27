import requests
import json
import re
import argparse
from transformers import LlamaTokenizerFast

def countTokens(text, model="LLaMA"):
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def sanitize(value):
    if isinstance(value, dict):
        value = {sanitize(k): sanitize(v) for k, v in value.items()}
    elif isinstance(value, list):
        value = [sanitize(v) for v in value]
    elif isinstance(value, str):
        value = re.sub(r"[ .]", "", value)
    return value

 
def getEmbeddings(endpoint, text):

    # Define the parameters for the API request
    params = {
        "model": "vicuna_13B",
        "input": text
    }
    api_endpoint = endpoint + "/embeddings"
    # Send the request to the API endpoint
    response = requests.post(api_endpoint, json=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        response_data = json.loads(response.text)

        # Access and print specific information from the response
        print("Object:", response_data["object"])
        print("Model:", response_data["model"])
        print("Usage - Prompt tokens:", response_data["usage"]["prompt_tokens"])
        print("Usage - Total tokens:", response_data["usage"]["total_tokens"])

        # Iterate over the embeddings in the data array
        for embedding in response_data["data"]:
            #print("Embedding object:", embedding["object"])
            #print("Embedding index:", embedding["index"])
            #print("Embedding values:", embedding["embedding"])
            return embedding["embedding"]
    else:
        print(f"Request failed with status code: {response.status_code}: " + response.text)


def getCompletion(endpoint, content, system_prompt, num_tokens=512, model="vicuna_13B"):
    completion = ""
    messages = []
    messages.append({"role":"system", "content":system_prompt})
    for msg in content:
        messages.append(msg)
    # Define the parameters for the API request
    params = {
        "model": model,
        "messages": messages,
        "max_tokens":num_tokens
    }
    api_endpoint = endpoint + "/chat/completions"
    # Send the request to the API endpoint
    response = requests.post(api_endpoint, json=params)
    #json_response = json
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        response_data = json.loads(response.text)

        # Access and print specific information from the response
        #print("Object:", response_data["object"])
        #print("Model:", response_data["model"])
        #print("Usage - Prompt tokens:", response_data["usage"]["prompt_tokens"])
        #print("Usage - Total tokens:", response_data["usage"]["total_tokens"])

        # Iterate over the embeddings in the data array
        for choice in response_data["choices"]:
            completion = completion + choice['message']['content']
            completion = completion + "\n"
        return completion
    else:
        print(f"Request failed with status code: {response.status_code}" + response. text)

