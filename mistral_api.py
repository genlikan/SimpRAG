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
            Only use the provided context to answer questions.
            """),
        ChatMessage(role="user", content=f"Context:\n{context}\n\nQuestion: {question}")
    ]

    response = client.chat(model="mistral-large-latest", messages=messages)
    return response.choices[0].message.content

# --- Intent Detection ---
def detect_intent(query: str) -> str:
    print("=== Detecting Intent ===")
    time.sleep(3)  # Delay to avoid spamming API

    messages = [
        ChatMessage(
            role="system",
            content="""
            You are an intent detection assistant. 
            Your job is to classify user input into one of the following categories:

            - 'search_query': The input asks for information that might be found in a document or knowledge base.
            - 'small_talk': The input is too short or ambiguous, a greeting, polite phrase, or general conversation with no need for a knowledge search.

            Respond with only one of the labels: 'search_query' or 'small_talk'.
            """
        ),
        ChatMessage(role="user", content=query)
    ]

    response = client.chat(model="mistral-small-2503", temperature=0, messages=messages)
    intent = response.choices[0].message.content.strip().lower()
    print(f"=== Query Intent: {intent} ===")

    # Normalize the output just in case
    if "small" in intent:
        return "small_talk"
    return "search_query"
