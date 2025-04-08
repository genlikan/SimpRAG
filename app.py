import os
import fitz  # PyMuPDF
import base64
import json
from io import BytesIO
from uuid import uuid4
from PIL import Image
import pytesseract
import numpy as np
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from mistral_api import embed_texts, generate_answer, detect_intent
from pydantic import BaseModel

app = FastAPI()
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------- Helper Functions ----------------

def cosine_similarity_manual(query_vector, doc_vectors):
    # Normalize the query vector and compute cosine similarities
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return [0.0] * len(doc_vectors)
    query_vector = np.array(query_vector) / query_norm

    scores = []
    for vec in doc_vectors:
        vec = np.array(vec)
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            scores.append(0.0)
        else:
            vec = vec / vec_norm
            similarity = np.dot(query_vector, vec)
            scores.append(similarity)
    return scores

def cosine_similarity_vectors(vec1, vec2):
    # Compute cosine similarity between two vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def semantic_chunk_text(text: str, similarity_threshold: float = 0.75) -> list:
    """
    Splits text into semantically coherent chunks based on sentence similarity.
    The text is first split into sentences and then grouped into chunks
    where the cosine similarity (using embed_texts for sentence embeddings)
    between consecutive sentences is above a threshold.
    """
    # Split text into sentences (using a simple regex)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
    
    # Get embeddings for each sentence (assumed to be batched in embed_texts)
    sentence_embeddings = embed_texts(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Compare embedding of the current sentence with the previous sentence
        similarity = cosine_similarity_vectors(sentence_embeddings[i-1], sentence_embeddings[i])
        if similarity < similarity_threshold:
            # Start a new chunk if similarity is below threshold
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            # Otherwise, append sentence to the current chunk
            current_chunk.append(sentences[i])
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# ---------------- API Endpoints ----------------

# Ingest PDFs with semantic chunking
@app.post("/ingest")
async def ingest_pdfs(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        filename = file.filename
        file_bytes = await file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        chunks = []

        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text").replace("\n"," ")
            if page_text.strip():
                # Perform semantic chunking on the page text
                semantic_chunks = semantic_chunk_text(page_text)
                if not semantic_chunks:
                    semantic_chunks = [page_text.strip()]  # Fallback to full page if chunking fails
                for idx, chunk_text in enumerate(semantic_chunks):
                    chunks.append({
                        "chunk_id": f"{filename}_page{page_num}_chunk{idx}",
                        "text": chunk_text.strip(),
                        "type": "text"
                    })

        # Get embeddings for all chunk texts
        texts_for_embedding = [c["text"] for c in chunks]
        embeddings = embed_texts(texts_for_embedding)

        for i, emb in enumerate(embeddings):
            chunks[i]["embedding"] = emb
            chunks[i]["filename"] = filename

        # Save the chunks to a JSON file
        save_path = os.path.join(DATA_DIR, f"{filename}.json")
        with open(save_path, "w") as f:
            json.dump(chunks, f, indent=2)

        results.append({"filename": filename, "chunks": len(chunks)})

    return {"ingested": results}


# --- GET PDFs ---
@app.get("/ingested_files")
def list_ingested_files():
    files = [
        fname.replace(".json", "")
        for fname in os.listdir(DATA_DIR)
        if fname.endswith(".json")
    ]
    return {"filenames": files}

# ---------------- Data Model ----------------

class DeleteRequest(BaseModel):
    filenames: list[str]

# --- DELETE PDFs ---
@app.delete("/ingest")
def delete_pdfs(request: DeleteRequest = Body(...)):
    deleted = []
    not_found = []
    errors = []
    deleted_summary = {}

    for fname in request.filenames:
        # Validate PDF extension
        if not fname.lower().endswith(".pdf"):
            errors.append(f"'{fname}' is not a valid .pdf filename.")
            continue

        json_path = os.path.join(DATA_DIR, f"{fname}.json")
        print(f"Checking: {json_path}")

        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    chunks = json.load(f)
                    chunk_count = len(chunks)
            except Exception as e:
                chunk_count = 0
                errors.append(f"Error reading {fname}: {str(e)}")

            os.remove(json_path)
            deleted.append(fname)
            deleted_summary[fname] = chunk_count
            print(f"Deleted {fname} ({chunk_count} chunks)")
        else:
            not_found.append(fname)

    return {
        "deleted": deleted,
        "deleted_summary": deleted_summary,
        "not_found": not_found,
        "errors": errors
    }


# --- QUERY SYSTEM ---
@app.post("/query")
def query_system(question: str = Form(...)):
    if is_small_talk(question):
        return {"message": "This doesn't seem to be a knowledge-based question."}

    intent = detect_intent(question)
    if intent == "small_talk":
        return {"message": "This doesn't seem to be a knowledge-based question."}

    all_chunks = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(DATA_DIR, fname)) as f:
                all_chunks.extend(json.load(f))

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No documents ingested yet.")

    query_embedding = embed_texts([question])[0]

    chunk_texts = [chunk["text"] for chunk in all_chunks]
    chunk_embeddings = [chunk["embedding"] for chunk in all_chunks]
    semantic_scores = cosine_similarity_manual(query_embedding, chunk_embeddings)

    # Keyword score based on token overlap
    query_tokens = set(re.findall(r"\w+", question.lower()))
    keyword_scores = [
        len(query_tokens.intersection(set(re.findall(r"\w+", chunk["text"].lower()))))
        for chunk in all_chunks
    ]

    # Combine semantic and keyword scores in a hybrid manner
    final_scores = [
        0.7 * sem + 0.3 * (kw / max(keyword_scores) if max(keyword_scores) else 0)
        for sem, kw in zip(semantic_scores, keyword_scores)
    ]

    # Retrieve top-K chunks based on the final score
    top_k = 5
    top_indices = np.argsort(final_scores)[-top_k:][::-1]

    top_chunks = []
    for i in top_indices:
        chunk = all_chunks[i]
        score = final_scores[i]
        top_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "score": round(score, 4),
            "text": chunk["text"]
        })

    # Build context from the top chunks and generate an answer
    context = "\n\n".join(c["text"] for c in top_chunks)
    answer = generate_answer(context, question)

    return {
        "question": question,
        "answer": answer,
        "used_chunks": [
            {
                "chunk_id": c["chunk_id"],
                "filename": c["filename"],
                "text": c["text"],
                "score": c["score"]
            }
            for c in top_chunks
        ]
    }


# --- SMALL TALK DETECTION ---
def is_small_talk(q: str) -> bool:
    q_lower = q.strip().lower()

    question_words = ["what", "how", "why", "who", "where", "when", "can", "could", "is", "are", "do", "does", "should"]
    if q_lower.endswith("?") or any(q_lower.startswith(w) for w in question_words):
        return False

    small_talk_patterns = [
        r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bhow are you\b",
        r"\bgood morning\b", r"\bgood evening\b", r"\bthank you\b", r"\bthanks\b",
        r"\bwhat's up\b", r"\bgreetings\b"
    ]

    for pattern in small_talk_patterns:
        if re.search(pattern, q_lower):
            print(f"[Small talk matched] Pattern: {pattern}")
            return True

    return False
