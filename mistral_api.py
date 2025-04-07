import os
import time
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Load API key from .env
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if MISTRAL_API_KEY:
    print(f"The Mistral API key is: {MISTRAL_API_KEY[:5]}...")
else:
    print("The MISTRAL_API_KEY environment variable is not set.")

# Initialize Mistral client
client = MistralClient(api_key=MISTRAL_API_KEY)

# --- Embedding ---
def embed_texts(texts):
    print("=== Making Embedding Request ===")
    time.sleep(3)  # Delay to avoid spamming API
    response = client.embeddings(model="mistral-embed", input=texts)
    return [item.embedding for item in response.data]

# --- Chat Completion ---
def generate_answer(context, question):
    print("=== Making Completion Request ===")
    time.sleep(3)  # Delay to avoid spamming API

    messages = [
        ChatMessage(role="system", content="""
            You are a helpful assistant. 
            Use the provided context to answer questions.
            If the user question is too short or ambiguous, ask for clarification instead.
            """),
        ChatMessage(role="user", content=f"Context:\n{context}\n\nQuestion: {question}")
    ]

    response = client.chat(model="mistral-large-latest", messages=messages)
    return response.choices[0].message.content
