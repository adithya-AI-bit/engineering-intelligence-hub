import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from hub.index import VectorIndex
from hub.embed import embed_texts
from hub.answer import answer_question, default_llm, extractive_llm
from hub.config import INDEX_PATH, LOCAL_EMBEDDER

STATE = {"index": None, "embed_fn": embed_texts, "llm_fn": default_llm}


def _configure_for(index):
    """Match the query path to whichever embedder built the index.

    Querying a local index with OpenAI vectors is a dimension mismatch, so the
    index carries its embedder and the server follows it instead of guessing.
    """
    if index.embedder == LOCAL_EMBEDDER:
        from hub.embed_local import embed_texts_local
        return embed_texts_local, extractive_llm
    return embed_texts, default_llm


@asynccontextmanager
async def lifespan(app):
    if STATE["index"] is None and os.path.exists(INDEX_PATH):
        index = VectorIndex.load(INDEX_PATH)
        STATE["index"] = index
        STATE["embed_fn"], STATE["llm_fn"] = _configure_for(index)
        print(f"Loaded {len(index.chunks)} chunks from {INDEX_PATH} "
              f"({index.embedder} embedder)")
    yield


app = FastAPI(title="Engineering Intelligence Hub", lifespan=lifespan)


class Question(BaseModel):
    question: str


@app.get("/")
def ui():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/health")
def health():
    index = STATE["index"]
    return {
        "status": "ok",
        "chunks": len(index.chunks) if index else 0,
        "embedder": index.embedder if index else None,
    }


@app.post("/ask")
def ask(payload: Question):
    if not payload.question.strip():
        return JSONResponse({"error": "Ask a question first."}, status_code=400)

    index = STATE["index"]
    if index is None:
        return JSONResponse({"error": "No index loaded. Run ingest.py first."}, status_code=503)

    result = answer_question(payload.question, index, STATE["embed_fn"], STATE["llm_fn"])
    return {**result, "used_citations": sorted(result["used_citations"])}
