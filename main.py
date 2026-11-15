import hashlib
import os
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, FilterSelector

# --- Configuration Constants ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "enterprise_rag_docs"
EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "qwen2.5:7b"  # Optimized for 6GB VRAM RTX 4050
EMBEDDING_DIM = 768
EMBED_BATCH = 32  # chunks embedded per Ollama call (much faster than one call per chunk)

# Chunking: ~1000 chars is roughly 250 tokens, enough for a complete thought and well inside
# nomic-embed-text's window. The old 300 chars cut sentences in half and gave the LLM fragments.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

UPLOAD_DIR = Path("uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- FastAPI & CORS Initialization ---
app = FastAPI(title="Enterprise Local RAG Engine", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Qdrant Client Initialization ---
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection() -> None:
    """Ensures the Qdrant collection exists with the proper 768-dim Cosine configuration."""
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"[Qdrant] Initialized collection '{COLLECTION_NAME}'.")


ensure_collection()


# --- Pydantic Data Models ---
class IngestRequest(BaseModel):
    pdf_path: str = Field(..., description="Path to the PDF file on disk")


class QueryRequest(BaseModel):
    query: str = Field(..., description="User prompt or technical query")
    # Optional knobs sent by the web UI sliders (defaults keep old behaviour)
    top_k: int = Field(4, ge=1, le=20, description="Number of chunks to retrieve")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum cosine score")


# --- Chunking helper ---
def clean_text(text: str) -> str:
    """PDF extractors (math fonts, ligatures) can emit lone surrogates / NUL bytes that crash
    UTF-8 encoding and JSON requests. Drop them so one odd glyph can't fail a whole upload."""
    return text.encode("utf-8", "ignore").decode("utf-8").replace("\x00", "")


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Sliding window that prefers to cut at a sentence end, then at a space, never mid-word."""
    text = " ".join(text.split())  # collapse PDF line breaks / runs of whitespace
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = start + int(size * 0.8)  # only look for a boundary in the last 20% of the window
            cut = max(text.rfind(p, window, end) for p in (". ", "? ", "! "))
            if cut != -1:
                end = cut + 1  # keep the punctuation
            else:
                space = text.rfind(" ", window, end)
                if space != -1:
                    end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        nxt = text.find(" ", start, end)  # start the next chunk on a word boundary
        if nxt != -1:
            start = nxt + 1
    return chunks


def embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    """Embeds a batch. If Ollama rejects the batch (one odd chunk can poison it), retry chunk by
    chunk and return None for the ones that fail, so a single bad chunk never kills the upload."""
    try:
        return ollama.embed(model=EMBEDDING_MODEL, input=texts)["embeddings"]
    except Exception as batch_err:
        print(f"[Ingest] batch embed failed ({type(batch_err).__name__}: {batch_err}); retrying one by one")
        out: List[Optional[List[float]]] = []
        for t in texts:
            try:
                out.append(ollama.embed(model=EMBEDDING_MODEL, input=t)["embeddings"][0])
            except Exception as e:
                print(f"[Ingest] skipped one chunk: {type(e).__name__}: {e}")
                out.append(None)
        return out


# --- Core RAG Pipeline Logic ---
class LocalRAGPipeline:
    @staticmethod
    def _deterministic_uuid(source: str, page_num: int, chunk_idx: int, text: str) -> str:
        """Generates a deterministic 32-char hex UUID to prevent duplicate vectors on re-upload."""
        raw = f"{source}:{page_num}:{chunk_idx}:{text}".encode("utf-8")
        return hashlib.md5(raw).hexdigest()

    @staticmethod
    def ingest_document(pdf_path: str) -> int:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        reader = PdfReader(pdf_path)
        chunks = []

        # 1. Sentence-aware sliding window chunking (per page, so page numbers stay accurate)
        for i, page in enumerate(reader.pages):
            text = clean_text(page.extract_text() or "")
            if not text.strip():
                continue
            for chunk_idx, chunk_text in enumerate(split_text(text)):
                if sum(ch.isalnum() for ch in chunk_text) < 20:
                    continue  # symbol-only / garbled chunk: useless for search and can break embedding
                chunks.append({
                    "chunk_id": LocalRAGPipeline._deterministic_uuid(path.name, i + 1, chunk_idx, chunk_text),
                    "text": chunk_text,
                    "metadata": {"source": path.name, "page_number": i + 1}
                })

        if not chunks:
            raise ValueError(f"No extractable text found in '{path.name}' (scanned PDF? it needs OCR).")

        # 2. Batched embedding & upserting to Qdrant
        points = []
        skipped = 0
        for b in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[b:b + EMBED_BATCH]
            vectors = embed_texts([f"search_document: {c['text']}" for c in batch])
            for chunk, vector in zip(batch, vectors):
                if vector is None:
                    skipped += 1
                    continue
                points.append(
                    PointStruct(
                        id=chunk["chunk_id"],
                        vector=vector,
                        payload={"text": chunk["text"], **chunk["metadata"]}
                    )
                )

        if not points:
            raise ValueError(f"Embedding failed for every chunk of '{path.name}'. Is Ollama running with '{EMBEDDING_MODEL}' installed?")
        if skipped:
            print(f"[Ingest] {path.name}: skipped {skipped} unembeddable chunk(s), indexed {len(points)}")

        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)

    @staticmethod
    def query_pipeline(query: str, top_k: int = 4, min_score: float = 0.0) -> Dict[str, Any]:
        res = ollama.embed(model=EMBEDDING_MODEL, input=f"search_query: {query}")
        query_vector = res["embeddings"][0]

        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            score_threshold=min_score if min_score > 0 else None,
        )

        context_chunks = []
        context_str = ""
        for idx, point in enumerate(search_results.points):
            payload = point.payload
            chunk_text = payload.get("text")
            source = payload.get("source")
            page = payload.get("page_number")
            score = point.score

            context_chunks.append({
                "source": source,
                "page": page,
                "score": score,
                "text": chunk_text
            })
            context_str += f"\n[Context {idx+1}] (Source: {source}, Page {page}, Score: {score:.4f})\n{chunk_text}\n"

        # Nothing relevant retrieved: skip the LLM instead of letting it guess
        if not context_chunks:
            return {
                "query": query,
                "answer": "I cannot find the answer in the provided documentation.",
                "retrieved_contexts": [],
            }

        system_prompt = (
            "You are an enterprise AI architect. Answer the query strictly using the provided context chunks. "
            "If the answer is missing, state 'I cannot find the answer in the provided documentation.' "
            "Always cite your sources (File name and Page number)."
        )
        user_prompt = f"Context:\n{context_str}\n\nUser Query: {query}"

        llm_response = ollama.chat(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return {
            "query": query,
            "answer": llm_response["message"]["content"],
            "retrieved_contexts": context_chunks
        }


# --- FastAPI REST Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "online", "qdrant": COLLECTION_NAME, "model": GENERATION_MODEL}


@app.get("/collections")
def get_collections():
    try:
        response = qdrant_client.get_collections()
        return {"collections": [c.name for c in response.collections]}
    except Exception:
        return {"collections": [COLLECTION_NAME]}


@app.delete("/index")
def reset_index(delete_files: bool = True):
    """Removes EVERY document: wipes and recreates the vector collection and (by default)
    deletes the stored PDFs, so the document list and the index stay in sync."""
    try:
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        ensure_collection()
        removed = 0
        if delete_files:
            for f in UPLOAD_DIR.glob("*.pdf"):
                f.unlink()
                removed += 1
        return {"status": "reset", "collection": COLLECTION_NAME, "files_removed": removed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    """PDFs stored on disk, for the web UI's document list / PDF browser."""
    docs = []
    for f in sorted(UPLOAD_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True):
        docs.append({"name": f.name, "size": f.stat().st_size, "status": "indexed"})
    return docs


@app.delete("/documents/{name}")
def delete_document(name: str):
    """Removes ONE document: its chunks in Qdrant (matched by the 'source' payload) and its stored PDF."""
    safe = Path(name).name  # blocks ../ traversal
    path = UPLOAD_DIR / safe
    flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=safe))])
    try:
        chunks = qdrant_client.count(collection_name=COLLECTION_NAME, count_filter=flt, exact=True).count
        if chunks == 0 and not path.is_file():
            raise HTTPException(status_code=404, detail=f"'{safe}' not found")
        if chunks:
            qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=FilterSelector(filter=flt))
        file_removed = False
        if path.is_file():
            path.unlink()
            file_removed = True
        return {"status": "deleted", "name": safe, "chunks_removed": chunks, "file_removed": file_removed}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pdfs/{name}")
def get_pdf(name: str):
    """Serves a stored PDF inline. Path.name blocks ../ traversal."""
    path = UPLOAD_DIR / Path(name).name
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(path, media_type="application/pdf")


# Plain `def` (not `async def`): FastAPI runs it in a worker thread, so the slow, blocking
# Ollama/Qdrant calls no longer freeze the whole server (and /health) during an upload.
@app.post("/ingest")
def api_ingest(
    files: List[UploadFile] = File(...),
    collection: Optional[str] = Form(None)
):
    total_chunks = 0
    accepted = 0
    try:
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                continue
            file_path = UPLOAD_DIR / Path(file.filename).name
            with open(file_path, "wb") as buffer:
                buffer.write(file.file.read())
            total_chunks += LocalRAGPipeline.ingest_document(str(file_path))
            accepted += 1
        if accepted == 0:
            raise ValueError("No PDF files received. Only .pdf uploads are supported.")
        return {"status": "success", "files_indexed": accepted, "chunks_indexed": total_chunks}
    except Exception as e:
        traceback.print_exc()  # full error in the uvicorn terminal
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")


@app.post("/query")
def api_query(payload: QueryRequest):
    try:
        return LocalRAGPipeline.query_pipeline(payload.query, payload.top_k, payload.min_score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)