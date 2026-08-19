import numpy as np
from hub.chunking import Chunk
from hub.index import VectorIndex
from evaluation.run_eval import load_golden, evaluate


def _index():
    index = VectorIndex()
    index.add(
        [Chunk("retry text", "docs/retry.md", 1, 2, "Retry"),
         Chunk("net text", "net.py", 1, 2, "net")],
        np.eye(2, dtype=np.float32),
    )
    return index


def test_load_golden_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('# a comment\n\n{"question": "q", "expected_source": "a.md"}\n')
    cases = load_golden(str(path))
    assert len(cases) == 1
    assert cases[0]["expected_source"] == "a.md"


def test_evaluate_reports_hit_and_perfect_groundedness():
    report = evaluate(
        [{"question": "how do retries work?", "expected_source": "docs/retry.md"}],
        _index(), k=1,
        embed_fn=lambda t: np.array([[1.0, 0.0]], dtype=np.float32),
        llm_fn=lambda s, u: "Retries back off [1].",
    )
    assert report["recall_at_k"] == 1.0
    assert report["mean_groundedness"] == 1.0
    assert report["invalid_citations"] == 0
    assert report["misses"] == []


def test_evaluate_records_a_retrieval_miss_with_detail():
    report = evaluate(
        [{"question": "how do retries work?", "expected_source": "does-not-exist.md"}],
        _index(), k=1,
        embed_fn=lambda t: np.array([[1.0, 0.0]], dtype=np.float32),
        llm_fn=lambda s, u: "Something [1].",
    )
    assert report["recall_at_k"] == 0.0
    assert report["misses"][0]["expected"] == "does-not-exist.md"
    assert report["misses"][0]["retrieved"] == ["docs/retry.md"]


def test_evaluate_counts_invalid_citations():
    report = evaluate(
        [{"question": "q", "expected_source": "docs/retry.md"}],
        _index(), k=1,
        embed_fn=lambda t: np.array([[1.0, 0.0]], dtype=np.float32),
        llm_fn=lambda s, u: "Claim [9].",
    )
    assert report["invalid_citations"] == 1


def test_evaluate_on_no_cases_does_not_divide_by_zero():
    report = evaluate([], _index(), k=1,
                      embed_fn=lambda t: np.array([[1.0, 0.0]], dtype=np.float32),
                      llm_fn=lambda s, u: "x")
    assert report["recall_at_k"] == 0.0
    assert report["mean_groundedness"] == 0.0
