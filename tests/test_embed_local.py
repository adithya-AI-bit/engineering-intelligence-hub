import numpy as np
from hub.embed_local import embed_texts_local, LOCAL_DIM


def test_shape_and_normalisation():
    vectors = embed_texts_local(["retry policy backoff", "network connection"])
    assert vectors.shape == (2, LOCAL_DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_identical_text_gives_identical_vectors():
    a, b = embed_texts_local(["same text here", "same text here"])
    assert np.allclose(a, b)


def test_related_text_scores_higher_than_unrelated():
    docs = embed_texts_local(["the retry policy uses exponential backoff",
                              "the parser reads markdown headers"])
    query = embed_texts_local(["how does retry backoff work"])[0]
    assert float(docs[0] @ query) > float(docs[1] @ query)


def test_snake_case_identifier_matches_its_parts():
    docs = embed_texts_local(["def retry_policy(): pass", "def unrelated(): pass"])
    query = embed_texts_local(["retry"])[0]
    assert float(docs[0] @ query) > float(docs[1] @ query)


def test_empty_text_does_not_divide_by_zero():
    vectors = embed_texts_local([""])
    assert vectors.shape == (1, LOCAL_DIM)
    assert not np.isnan(vectors).any()


def test_bucket_assignment_is_stable_across_processes():
    """Built-in hash() is randomised per process; a saved index must survive reload."""
    import subprocess
    import sys

    script = (
        "from hub.embed_local import embed_texts_local; import numpy as np; "
        "print(list(np.nonzero(embed_texts_local(['retry policy backoff'])[0])[0]))"
    )
    runs = [
        subprocess.run([sys.executable, "-c", script], capture_output=True, text=True).stdout.strip()
        for _ in range(2)
    ]
    assert runs[0] == runs[1], f"vectors differ across processes: {runs}"
