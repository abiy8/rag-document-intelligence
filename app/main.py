import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
import httpx
from fastapi import FastAPI, File, Header, HTTPException, UploadFile, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from .engine import Engine

def authorize(x_api_key: str | None = Header(default=None)):
    key = os.getenv("API_KEY", "")
    if key:
        import hmac
        if not x_api_key or not hmac.compare_digest(key, x_api_key):
            raise HTTPException(401, "Invalid API key")

def create_app(store=None, backend=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = Engine(store or os.getenv("RAG_STORE", "data/qdrant"), backend or os.getenv("EMBEDDING_BACKEND", "hash"), os.getenv("GENERATION_BACKEND", "extractive"), os.getenv("OLLAMA_URL", "http://localhost:11434"), os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
        yield
        app.state.engine.close()
    app = FastAPI(title="Document Intelligence", lifespan=lifespan)

    class Document(BaseModel):
        name: str = Field(min_length=1, max_length=200)
        text: str = Field(min_length=1, max_length=500000)
    class Query(BaseModel):
        question: str = Field(min_length=2, max_length=2000)
        k: int = Field(default=4, ge=1, le=10)

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "web.html")
    @app.get("/health")
    def health():
        return {"status":"ok", "embedding_backend":app.state.engine.embedder.backend, "generation_backend":app.state.engine.generation}
    @app.post("/documents", dependencies=[Depends(authorize)])
    def ingest(document: Document):
        try:
            return app.state.engine.ingest(document.name, document.text)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    @app.post("/upload", dependencies=[Depends(authorize)])
    def upload(file: UploadFile = File(...)):
        data = file.file.read(5_000_001)
        if len(data) > 5_000_000:
            raise HTTPException(413, "Maximum file size is 5 MB")
        name = Path(file.filename or "document.txt").name
        try:
            if name.lower().endswith(".pdf"):
                text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
            elif name.lower().endswith((".txt", ".md")):
                text = data.decode("utf-8")
            else:
                raise HTTPException(415, "Use PDF, TXT or Markdown")
            return app.state.engine.ingest(name, text)
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(422, "Could not extract document text") from exc
    @app.post("/query", dependencies=[Depends(authorize)])
    def query(q: Query):
        try:
            return app.state.engine.answer(q.question, q.k)
        except httpx.HTTPError as exc:
            raise HTTPException(502, "Local LLM is unavailable; check Ollama configuration") from exc
    @app.delete("/documents/{document_id}", dependencies=[Depends(authorize)])
    def delete(document_id: str):
        app.state.engine.delete(document_id)
        return {"deleted": document_id}
    return app

app = create_app()
