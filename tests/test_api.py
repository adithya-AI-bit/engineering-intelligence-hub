import numpy as np
import pytest
from fastapi.testclient import TestClient
from hub.chunking import Chunk
from hub.index import VectorIndex
import api.main as main


@pytest.fixture
def client():
    index = VectorIndex()
    index.add([Chunk("Retries back off.", "docs/retry.md", 1, 2, "Retry")], np.eye(1, dtype=np.float32))
    main.STATE["index"] = index
    main.STATE["embed_fn"] = lambda texts: np.array([[1.0]], dtype=np.float32)
    main.STATE["llm_fn"] = lambda system, user: "Retries back off [1]."
    with TestClient(main.app) as test_client:
        yield test_client
    main.STATE["index"] = None


def test_health_reports_chunk_count(client):
    assert client.get("/health").json()["chunks"] == 1


def test_ask_returns_answer_and_sources(client):
    body = client.post("/ask", json={"question": "how do retries work?"}).json()
    assert body["answer"] == "Retries back off [1]."
    assert body["sources"][0]["source"] == "docs/retry.md"
    assert body["used_citations"] == [1]


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": "   "}).status_code == 400


def test_ask_rejects_missing_field(client):
    assert client.post("/ask", json={}).status_code == 422


def test_ask_without_an_index_returns_503(client):
    main.STATE["index"] = None
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 503
    assert "ingest.py" in response.json()["error"]


def test_root_serves_the_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Engineering Intelligence Hub" in response.content


def test_local_index_is_served_with_the_matching_embedder(tmp_path):
    """A --local index must be queryable without a key, not crash on dimensions."""
    from hub.config import LOCAL_EMBEDDER
    from hub.embed_local import embed_texts_local
    from hub.answer import extractive_llm

    index = VectorIndex(embedder=LOCAL_EMBEDDER)
    chunks = [Chunk("retries use exponential backoff", "docs/retry.md", 1, 2, "Retry policy")]
    index.add(chunks, embed_texts_local([c.text for c in chunks]))

    embed_fn, llm_fn = main._configure_for(index)
    assert embed_fn is embed_texts_local
    assert llm_fn is extractive_llm

    main.STATE["index"] = index
    main.STATE["embed_fn"] = embed_fn
    main.STATE["llm_fn"] = llm_fn
    with TestClient(main.app) as c:
        body = c.post("/ask", json={"question": "how do retries work?"}).json()
    main.STATE["index"] = None

    assert "exponential backoff" in body["answer"]
    assert body["sources"][0]["source"] == "docs/retry.md"
    assert body["used_citations"] == [1]


def test_health_reports_the_embedder(client):
    assert client.get("/health").json()["embedder"] == "openai"


def test_mismatched_query_dimensions_raise_a_readable_error():
    import numpy as np
    from hub.config import LOCAL_EMBEDDER
    index = VectorIndex(embedder=LOCAL_EMBEDDER)
    index.add([Chunk("x", "a.md", 1, 1, "x")], np.ones((1, 512), dtype=np.float32))
    with pytest.raises(ValueError, match="built with the 'local' embedder"):
        index.search(np.ones(1536, dtype=np.float32), k=1)
