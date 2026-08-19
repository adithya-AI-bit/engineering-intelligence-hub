import numpy as np
import pytest
from hub.chunking import Chunk
from hub.index import VectorIndex


def _chunk(name):
    return Chunk(text=f"body of {name}", source=f"{name}.md", start_line=1, end_line=2, context=name)


def test_search_returns_nearest_chunk_first():
    index = VectorIndex()
    index.add([_chunk("a"), _chunk("b")], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    results = index.search(np.array([0.9, 0.1], dtype=np.float32), k=2)
    assert results[0][0].context == "a"
    assert results[0][1] > results[1][1]


def test_search_respects_k():
    index = VectorIndex()
    index.add([_chunk(str(i)) for i in range(5)], np.eye(5, dtype=np.float32))
    assert len(index.search(np.eye(5, dtype=np.float32)[0], k=3)) == 3


def test_search_on_empty_index_returns_empty():
    assert VectorIndex().search(np.array([1.0, 0.0], dtype=np.float32), k=3) == []


def test_add_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        VectorIndex().add([_chunk("a")], np.eye(2, dtype=np.float32))


def test_save_and_load_roundtrip(tmp_path):
    index = VectorIndex()
    index.add([_chunk("a"), _chunk("b")], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    path = tmp_path / "index.npz"
    index.save(path)

    loaded = VectorIndex.load(str(path))
    assert len(loaded.chunks) == 2
    assert loaded.chunks[0].start_line == 1
    assert loaded.search(np.array([1.0, 0.0], dtype=np.float32), k=1)[0][0].context == "a"


def test_save_and_load_empty_index(tmp_path):
    path = tmp_path / "empty.npz"
    VectorIndex().save(path)
    loaded = VectorIndex.load(str(path))
    assert loaded.chunks == []
    assert loaded.search(np.array([1.0], dtype=np.float32), k=3) == []


def test_add_twice_stacks_vectors():
    index = VectorIndex()
    index.add([_chunk("a")], np.array([[1.0, 0.0]], dtype=np.float32))
    index.add([_chunk("b")], np.array([[0.0, 1.0]], dtype=np.float32))
    assert index.vectors.shape == (2, 2)
    assert len(index.search(np.array([0.0, 1.0], dtype=np.float32), k=2)) == 2
