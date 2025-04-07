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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from sklearn.metrics.pairwise import cosine_similarity
from mistral_api import embed_texts, generate_answer

app = FastAPI()
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# --- INGEST PDFs ---
@app.post("/ingest")
async def ingest_pdfs(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        filename = file.filename
        file_bytes = await file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        chunks = []

        for page_num, page in enumerate(doc):
            # Extract text
            text = page.get_text("text")
            if text.strip():
                chunks.append({
                    "chunk_id": f"{filename}_page{page_num}_text",
                    "text": text.strip(),
                    "type": "text"
                })

            # Extract images + OCR
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n < 5:  # Not CMYK
                    img_bytes = pix.tobytes("png")
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                    # OCR with pytesseract
                    pil_img = Image.open(BytesIO(img_bytes))
                    ocr_text = pytesseract.image_to_string(pil_img).strip()
                    final_text = ocr_text if ocr_text else "Image content (no readable text found)"

                    chunks.append({
                        "chunk_id": f"{filename}_page{page_num}_img{img_index}",
                        "text": final_text,
                        "image_base64": img_b64,
                        "type": "image"
                    })

        # Get embeddings
        texts_for_embedding = [c["text"] for c in chunks]
        embeddings = embed_texts(texts_for_embedding)

        for i, emb in enumerate(embeddings):
            chunks[i]["embedding"] = emb
            chunks[i]["filename"] = filename

        # Save chunks to file
        save_path = os.path.join(DATA_DIR, f"{filename}.json")
        with open(save_path, "w") as f:
            json.dump(chunks, f, indent=2)

        results.append({"filename": filename, "chunks": len(chunks)})

    return {"ingested": results}


# --- DELETE PDFs ---
@app.delete("/ingest")
def delete_pdfs(filenames: list[str] = Form(...)):
    deleted = []
    not_found = []

    for fname in filenames:
        path = os.path.join(DATA_DIR, f"{fname}.json")
        if os.path.exists(path):
            os.remove(path)
            deleted.append(fname)
        else:
            not_found.append(fname)

    return {"deleted": deleted, "not_found": not_found}


# --- QUERY SYSTEM ---
@app.post("/query")
def query_system(question: str = Form(...)):
    if is_small_talk(question):
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
    semantic_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]

    # Keyword score
    query_tokens = set(re.findall(r"\w+", question.lower()))
    keyword_scores = [
        len(query_tokens.intersection(set(re.findall(r"\w+", chunk["text"].lower()))))
        for chunk in all_chunks
    ]

    # Hybrid score (semantic + keyword)
    final_scores = [
        0.7 * sem + 0.3 * (kw / max(keyword_scores) if max(keyword_scores) else 0)
        for sem, kw in zip(semantic_scores, keyword_scores)
    ]

    # Top-K chunks
    top_k = 5
    top_indices = np.argsort(final_scores)[-top_k:][::-1]
    top_chunks = [all_chunks[i] for i in top_indices]
    context = "\n\n".join(chunk["text"] for chunk in top_chunks)

    answer = generate_answer(context, question)

    return {
        "question": question,
        "answer": answer,
        "used_chunks": [{"chunk_id": c["chunk_id"], "filename": c["filename"]} for c in top_chunks]
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
